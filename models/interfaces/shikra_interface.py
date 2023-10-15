from .shikra.mllm.dataset.process_function import PlainBoxFormatter
from .shikra.mllm.dataset.builder import prepare_interactive
from .shikra.mllm.models.builder.build_shikra import load_pretrained_shikra
from .shikra.mllm.dataset.utils.transform import expand2square, box_xyxy_expand2square
from mmengine import Config
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
from .utils import get_image
import torch
import torch.nn as nn
from utils.preprocessors import ShikraProcessor

# notice that miniGPT-4 requires relative import since it does not enable absolute installation
class Shikra_Interface(nn.Module):
    def __init__(self, model_path="facebook/opt-350m", device=None, half=False, inference_method='generation') -> None:
        super(Shikra_Interface, self).__init__()

        # setup the cuda device
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # model config
        model_args = Config(dict(
            type='shikra',
            version='v1',

            # checkpoint config
            cache_dir=None,
            model_name_or_path=model_path,
            vision_tower=r'openai/clip-vit-large-patch14',
            pretrain_mm_mlp_adapter=None,

            # model config
            mm_vision_select_layer=-2,
            model_max_length=2048,

            # finetune config
            freeze_backbone=False,
            tune_mm_mlp_adapter=False,
            freeze_mm_mlp_adapter=False,

            # data process config
            is_multimodal=True,
            sep_image_conv_front=False,
            image_token_len=256,
            mm_use_im_start_end=True,

            target_processor=dict(
                boxes=dict(type='PlainBoxFormatter'),
            ),

            process_func_args=dict(
                conv=dict(type='ShikraConvProcess'),
                target=dict(type='BoxFormatProcess'),
                text=dict(type='ShikraTextProcess'),
                image=dict(type='ShikraImageProcessor'),
            ),

            conv_args=dict(
                conv_template='vicuna_v1.1',
                transforms=dict(type='Expand2square'),
                tokenize_kwargs=dict(truncation_size=None),
            ),

            gen_kwargs_set_pad_token_id=True,
            gen_kwargs_set_bos_token_id=True,
            gen_kwargs_set_eos_token_id=True,
        ))
        training_args = Config(dict(
            bf16=False,
            fp16=True,
            device='cuda',
            fsdp=None,
        ))

        
        # load the model!
        self.model, self.preprocessor = load_pretrained_shikra(model_args, training_args)
        
        # convert to half precision if needed
        self.half_precision = half
        if half:
            self.dtype = torch.float16
            self.model.to(dtype=torch.float16, device='cuda')
            self.model.model.vision_tower[0].to(dtype=torch.float16, device=self.device)
        else:
            self.dtype = torch.float32
            self.model.to(device=self.device)
            self.model.model.vision_tower[0].to(device=self.device)

        # load box processor
        self.preprocessor['target'] = {'boxes': PlainBoxFormatter()}
        self.tokenizer = self.preprocessor['text']
        # setup the inference method
        self.inference_method = inference_method
        
        # setup the ds template
        self.ds_template = prepare_interactive(model_args=model_args, preprocessor=self.preprocessor)


    @torch.no_grad()
    def raw_generate(self, model_inputs, max_new_tokens=512, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        # in Shikra, no need to process the image, the image is processed by the preprocessor
        model_inputs['images'] = model_inputs['images'].to(dtype=self.dtype, device=self.device)
        model_inputs['input_ids'] = model_inputs['input_ids'].to(device=self.device)
        input_ids = model_inputs['input_ids']
        
        # the generation arguments
        gen_kwargs = dict(
            use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=float(temperature),
        )
        
        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                output_ids = self.model.generate(input_ids=model_inputs['input_ids'],
                                                 images=model_inputs['images'],
                                                 **gen_kwargs)
        
        input_token_len = input_ids.shape[-1]
        response = self.tokenizer.batch_decode(output_ids[:, input_token_len:])[0]

        return response
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, inputs_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(inputs, temperature=temperature, max_new_tokens=max_new_tokens) for img, inputs in zip(image_list, inputs_list)]
        return outputs
    
    @torch.no_grad()
    def raw_predict(self, model_inputs, candidates, likelihood_reduction='sum'):
        # in Shikra, no need to process the image, the image is processed by the preprocessor
        model_inputs['images'] = model_inputs['images'].to(dtype=self.dtype, device=self.device)
        model_inputs['input_ids'] = model_inputs['input_ids'].to(device=self.device)
        input_ids = model_inputs['input_ids']
        
        # get the embedding from the input
        num_cand = len(candidates)
        input_seq_len = input_ids.shape[1]
        attention_mask = torch.ones(num_cand, input_seq_len, dtype=torch.long, device=self.device)

        # tokenize the candidates
        current_padding_side = self.tokenizer.padding_side
        current_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        candidates_tokens = self.tokenizer(
            candidates,
            return_tensors='pt',
            padding='longest'
        ).to(self.device)
        self.tokenizer.padding_side = current_padding_side
        self.tokenizer.truncation_side = current_truncation_side

        # construct the inputs_ids and LM targets
        candidates_ids = candidates_tokens.input_ids[:, 1:] # remove the <s> token
        candidates_att = candidates_tokens.attention_mask[:, 1:] # remove the <s> token
        # mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_token_id, -100)
        # mask the targets for inputs part
        targets = torch.cat([-100*torch.ones(num_cand, input_seq_len, dtype=torch.long, device=self.device), \
                             cand_targets], dim=1)
        # concatenate the inputs for the model
        attention_mask = torch.cat([attention_mask, candidates_att], dim=1)
        full_input_ids = torch.cat([input_ids.repeat_interleave(num_cand, dim=0), candidates_ids], dim=1)

        with torch.inference_mode():
            outputs = self.model.forward_likelihood(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                labels=targets,
                images=model_inputs['images'].repeat_interleave(num_cand, dim=0),
                return_dict=True,
                likelihood_reduction=likelihood_reduction
            )
        neg_likelihood = outputs.loss
        if likelihood_reduction == 'none':
            return neg_likelihood
        output_class_ranks = torch.argsort(neg_likelihood, dim=-1)[0].item()

        return output_class_ranks
    
    @torch.no_grad()
    def raw_batch_predict(self, image_list, question_list, candidates, likelihood_reduction='sum'):
        preds = [self.raw_predict(question, cands, likelihood_reduction=likelihood_reduction) for image, question, cands in zip(image_list, question_list, candidates)]

        return preds
    
    def forward(self, image, prompt, candidates=None, temperature=0.2, max_new_tokens=30, likelihood_reduction='sum'):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates, likelihood_reduction=likelihood_reduction)
        else:
            raise NotImplementedError
    
def get_shikra(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_name', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = Shikra_Interface(**model_args)
    ds_template = model.ds_template
    proc = ShikraProcessor(ds_template, infer_method=model_args['inference_method'], answer_prefix='The answer is')
    return model, proc

if __name__=='__main__':
    model = Shikra_Interface()