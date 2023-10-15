from llava import LlavaLlamaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
from .utils import get_image
import torch
import torch.nn as nn
from utils.preprocessors import BaseProcessor, SingleChoiceProcessor, ConvSingleChoiceProcessor


class LLaVA_Interface(nn.Module):
    def __init__(self, model_base=None, model_path="facebook/opt-350m", device=None, half=False, inference_method='generation') -> None:
        super(LLaVA_Interface, self).__init__()
        model_name = get_model_name_from_path(model_path)
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, \
                                                                                model_name, device_map=device)
        
        # add pad to tokenizer if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # convert to the device
        self.model.to(self.device)
        
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        self.conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = self.conv.roles
        
        self.roles = roles

        # setup the inference method
        self.inference_method = inference_method

    def get_conv(self):
        return self.conv
    
    def get_first_query_process(self):
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            return lambda qs: DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            return lambda qs: DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=0.2, max_new_tokens=30):
        image = get_image(image)
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(self.device)
        # if getattr(self.model.config, 'mm_use_im_start_end', False):
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # setup the stopping criteria
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature, max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
        image = get_image(image)
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(self.device)
        # if getattr(self.model.config, 'mm_use_im_start_end', False):
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs/root/LLM-V-Bench/models/build_scripts
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # prepare inputs for the input part
        input_ids = input_ids.repeat_interleave(len(candidates), dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype, device=input_ids.device)
        
        # tokenize the candidates
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
        # mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_token_id, -100)
        # mask the targets for inputs part
        targets = torch.cat([-100*torch.ones_like(input_ids), cand_targets], dim=1)
        # concatenate the inputs for the model
        input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        attention_mask = torch.cat([attention_mask, candidates_att], dim=1)

        with torch.inference_mode():
            outputs = self.model.forward_likelihood(
                input_ids,
                images=image.repeat_interleave(len(candidates), dim=0),
                attention_mask=attention_mask,
                labels=targets,
                return_dict=True,
                likelihood_reduction=likelihood_reduction
            )
        neg_likelihood = outputs.loss
        if likelihood_reduction == 'none':
            return input_ids, neg_likelihood
        output_class_ranks = torch.argsort(neg_likelihood, dim=-1)[0].item()

        return output_class_ranks
    
    @torch.no_grad()
    def raw_batch_predict(self, image_list, question_list, candidates, likelihood_reduction='sum'):
        preds = [self.raw_predict(image, question, cands, likelihood_reduction=likelihood_reduction) for image, question, cands in zip(image_list, question_list, candidates)]

        return preds
    
    def forward(self, image, prompt, candidates=None, temperature=0.2, max_new_tokens=30, likelihood_reduction='sum'):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates, likelihood_reduction=likelihood_reduction)
        else:
            raise NotImplementedError
    
def get_llava(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_name', 'model_type', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'model_base', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = LLaVA_Interface(**model_args)
    conv = model.get_conv()
    first_query_process_fn = model.get_first_query_process()
    if conv.sep_style.name == 'SINGLE':
        sep_style = 'one'
    elif conv.sep_style.name == 'TWO':
        sep_style = 'two'
    elif conv.sep_style.name == 'LLAMA_2':
        sep_style = 'llama_2'
    else:
        raise NotImplementedError
    proc = ConvSingleChoiceProcessor(conv.sep, sep2=conv.sep2, roles=conv.roles, system_msg=conv.system, \
                                     first_query_fn=first_query_process_fn, init_conv=conv.messages, \
                                     sep_style=sep_style, infer_method=model_args['inference_method'],
                                     response_prefix='The answer is')
    return model, proc

if __name__=='__main__':
    model = LLaVA_Interface()