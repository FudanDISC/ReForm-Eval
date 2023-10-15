# from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
# from transformers import MplugOwlProcessor, MplugOwlForConditionalGeneration
from .mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor, tokenize_prompts

import torch
from PIL import Image
import torch.nn as nn
from utils.preprocessors import ConvSingleChoiceProcessor
import contextlib
from types import MethodType
from .utils import get_image

class mPLUG_Owl_Interface(nn.Module):
    def __init__(self, model_name='mplugowl', model_path='MAGAer13/mplug-owl-llama-7b', device=None, half=False, inference_method='generation') -> None:
        super(mPLUG_Owl_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.pretrained_ckpt = model_path
        self.prec_half = half

        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            model_path # 'MAGAer13/mplug-owl-llama-7b',
            )
        
        # if self.prec_half:
        #     if torch.cuda.is_bf16_supported():
        #         self.model.to(dtype=torch.bfloat16)
        #     else:
        #         self.model.to(dtype=torch.float16)
        if self.prec_half:
            self.model = self.model.half()

        self.model.to(self.device)

        self.vis_processors = MplugOwlImageProcessor.from_pretrained(self.pretrained_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.vis_processors, self.tokenizer) #可以只输入图片，但是比vis_processors多操作了一步，就是images = [_.resize((224, 224), 3) for _ in images]，导致和vis_processors输出的结果不一样

        self.inference_method = inference_method
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=0.2, max_new_tokens=30):
        image = [get_image(image)]
        inputs = self.processor(text=prompt, images=image, return_tensors='pt')
        # print(inputs['input_ids'])
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     inputs = {k: v.to(dtype=torch.bfloat16) if v.dtype == torch.float else v for k, v in inputs.items()}
            # else:
            #     inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float else v for k, v in inputs.items()} 

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens,
            'temperature': temperature
            }
        
        with torch.inference_mode():
            res = self.model.generate(**inputs, **generate_kwargs)
        sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        
        return sentence
  
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(img, [question],temperature=temperature, max_new_tokens=max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs


    @torch.no_grad()
    def raw_predict(self, images, prompts, candidates, likelihood_reduction='sum'):
        if not isinstance(images, list):
            images = [images]
        images = [get_image(image) for image in images]
        images = [_.resize((224,224),3) for _ in images]
        image_tensors = self.vis_processors(images=images, return_tensors='pt')['pixel_values']
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     image_tensors = image_tensors.to(dtype=torch.bfloat16)
            # else:
            #     image_tensors = image_tensors.to(dtype=torch.float16)
            image_tensors = image_tensors.to(dtype=torch.float16)
            
        image_tensors = image_tensors.to(self.device)

        num_cand = len(candidates)
        
        # changed add_BOS=True
        context_tokens_tensor = tokenize_prompts(prompts=[prompts], tokens_to_generate=0, add_BOS=True, tokenizer=self.tokenizer, ignore_dist=True)
        input_ids = context_tokens_tensor['input_ids'].to(self.device)
        # print(input_ids)
        input_ids = input_ids.repeat_interleave(len(candidates), dim=0)
        
        attention_mask = context_tokens_tensor['attention_mask'].to(self.device)
        attention_mask = attention_mask.repeat_interleave(len(candidates), dim=0)
        
        #tokenize the candidates
        current_padding_side = self.tokenizer.padding_side
        current_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        candidates_tokens = self.tokenizer(
                [cand for cand in candidates],
                return_tensors='pt',
                padding='longest'
            ).to(self.device)
        self.tokenizer.padding_side = current_padding_side
        self.tokenizer.truncation_side = current_truncation_side
        
        # construct the inputs_ids and LM targets
        candidates_ids = candidates_tokens.input_ids[:, 1:] # remove the <s> token
        candidates_att = candidates_tokens.attention_mask[:, 1:] # remove the <s> token
        
        
        #mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_token_id, -100)
        
        #mask the targets for inputs part
        targets = torch.cat([-100*torch.ones_like(input_ids),cand_targets], dim=1)
        # print(targets.shape)
        
        #prompt_mask before the concatenation
        prompt_mask = torch.cat([0*torch.ones_like(input_ids), candidates_att], dim=-1)[:,1:]
        
        #concatenate the inputs for the model
        input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        attention_mask = torch.cat([attention_mask, candidates_att], dim=1)
        
        #non_padding_mask
        # non_padding_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)[:, 1:]
        non_padding_mask = (input_ids != self.tokenizer.pad_token_id).long()[:, 1:]
        
        # #non_media_mask
        tmp_enc_chunk = input_ids[:, :-1].clone()
        non_media_mask = torch.ones_like(tmp_enc_chunk)
        non_media_mask = non_media_mask.masked_fill(tmp_enc_chunk < 0, 0)
        # print(non_media_mask.shape)
        
        num_images=torch.tensor([1]*num_cand)
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, pixel_values=image_tensors.repeat_interleave(len(candidates), dim=0), \
                            labels=targets, num_images=num_images, non_padding_mask=non_padding_mask, \
                            non_media_mask=non_media_mask,prompt_mask=prompt_mask)
        
        logits = outputs.logits
        labels = targets
        # loss_mask = torch.ones(input_ids.size(), dtype=torch.float, device=logits.device)
        # loss_mask = loss_mask[:, :-1] * non_padding_mask * non_media_mask * prompt_mask
        # labels[:, 1:][loss_mask != 1] = -100
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           bnnnnnnnnnnnnnn                                                                                                    bnvv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss(reduction='none')
        vocab_size= logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(logits.size(0), -1)
        if likelihood_reduction == 'sum':
            loss = loss.sum(1)
        elif likelihood_reduction == 'mean':
            valid_num_targets = (loss > 0).sum(1)
            loss = loss.sum(1) / valid_num_targets
        elif likelihood_reduction == 'none':
            loss = loss
            return loss
        else:
            raise ValueError
        output_class_ranks = torch.argsort(loss, dim=-1)[0].item()
        return output_class_ranks

    @torch.no_grad()
    def raw_batch_predict(self, image_list, question_list, candidates):
        preds = [self.raw_predict(image, question, cands) for image, question, cands in zip(image_list, question_list, candidates)]
        return preds

    def forward(self, image, prompt, candidates=None, temperature=0.2, max_new_tokens=30):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates)
        else:
            raise NotImplementedError
    
def get_mPLUG_Owl(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_type', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    # print(model_args)
    model = mPLUG_Owl_Interface(**model_args)
    preprocessor = ConvSingleChoiceProcessor(sep='\n', infer_method=model_args['inference_method'],\
                           system_msg="The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", 
                           init_conv=[['Human', "<image>"]], roles=['Human', 'AI'], sep_style="one", response_prefix='The answer is')
    return model, preprocessor
if __name__=='__main__':
    model = mPLUG_Owl_Interface()