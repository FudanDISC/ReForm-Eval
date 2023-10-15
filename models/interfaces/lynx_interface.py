from .lynx.models.lynx import LynxBase
from .lynx.dataset import get_image_transform, get_video_transform
from .lynx.dataset.tokenizers import build_tokenizer
import torch
from PIL import Image
import torch.nn as nn
from utils.preprocessors import BaseProcessor, SingleChoiceProcessor, ConvSingleChoiceProcessor
import contextlib
from types import MethodType
from .utils import get_image
import yaml
import os, io
from base64 import b64decode
import json
import numpy as np


class Lynx_Interface(nn.Module):
    def __init__(self, model_config=None, device=None, half=False, inference_method='generation') -> None:
        super(Lynx_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # loading the model
        self.config = yaml.load(open(model_config, 'r'), Loader=yaml.Loader)
        self.model = LynxBase(config=self.config, freeze_vit=self.config['freeze_vit'], freeze_llm=self.config['freeze_llm'], load_bridge=False)

        self.prec_half = half
        if self.prec_half:
            self.model = self.model.half()
        self.model = self.model.to(self.device)

        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self.model.eval()

        # setup text process
        self.use_left_pad = self.config['use_left_pad']
        self.tokenizer = self.model.tokenizer
        # self.tokenizer, _ = build_tokenizer(self.config, self.config['LLM'], use_left_pad=self.use_left_pad)
        self.max_input_tokens = self.config['max_input_tokens']

        self.lower_text = self.config['lower_text']

        # setup the visual data preprocessor
        _, self.img_transform = get_image_transform(self.config)
        _, self.video_transform = get_video_transform(self.config)

        # setup the inference method
        self.inference_method = inference_method

    def _get_frames(self, vision_inp):
        assert isinstance(vision_inp, list) # or isinstance(vision_inp, np.ndarray)
        assert isinstance(vision_inp[0], str)

        frames = []
        for i in range(len(vision_inp)):
            img = Image.open(io.BytesIO(b64decode(vision_inp[i]))).convert("RGB")
            frames.append(self.video_transform(img))

        return frames
    
    def load_vision_inp(self, vision_inp):
        if vision_inp is None:
            return None

        elif isinstance(vision_inp, list) or isinstance(vision_inp, np.ndarray):
            return self._get_frames(vision_inp)

        elif isinstance(vision_inp, str):

            if os.path.exists(vision_inp):
                image = Image.open(vision_inp).convert('RGB')

            else:  # base64 encoding
                try:
                    image = Image.open(io.BytesIO(b64decode(vision_inp))).convert("RGB")
                except Exception as e:
                    raise ValueError(f"check whether it is a rpath (and not exist)?: {vision_inp} {e}")
        else:
            image = vision_inp
        
        image = self.img_transform(image)

        return image.to(self.device)
    
    def process_text(self, text):
        text = text.strip()
        if self.lower_text:
            text = text.lower()
        input_ids = [self.tokenizer.bos_token] + self.tokenizer.tokenize(text)
        # print(input_ids)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        input_atts = torch.LongTensor([[1]*len(input_ids)])
        input_ids = torch.LongTensor([input_ids])
        return input_ids.to(self.device), input_atts.to(self.device)
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=1, max_new_tokens=30):
        vision_input = self.load_vision_inp(image).unsqueeze(0)
        if self.prec_half:
            vision_input = vision_input.to(torch.float16)
        
        input_ids, input_atts = self.process_text(prompt)
        
        answer = self.model.generate(
            vision_input=vision_input,
            input_ids=input_ids, input_atts=input_atts,
            use_nucleus_sampling=self.config.get('use_nucleus_sampling', False),
            apply_lemmatizer=self.config['apply_lemmatizer'],
            num_beams=3, # self.config['num_beams'],
            min_length=self.config['min_length'],
            length_penalty=self.config.get('length_penalty', 1.0),
            no_repeat_ngram_size=self.config.get('no_repeat_ngram_size', -1),
            top_p=self.config.get('top_p', 0.9),
            top_k=self.config.get('top_k', 3),
            max_new_tokens=max_new_tokens,
            temperature=temperature)

        return answer[0]
    
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=1, max_new_tokens=30):
        output = [self.raw_generate(image_list[i], question_list[i], temperature=temperature, max_new_tokens=max_new_tokens) for i in range(len(image_list))]
        return output
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
        vision_input = self.load_vision_inp(image).unsqueeze(0)
        if self.prec_half:
            vision_input = vision_input.to(torch.float16)
        
        input_ids, attention_mask = self.process_text(prompt)
        
        # get the embedding from the input
        num_cand = len(candidates)
        input_seq_len = input_ids.shape[1]

        # tokenize the candidates
        current_padding_side = self.tokenizer.padding_side
        current_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        if self.lower_text:
            candidates = [cand.lower() for cand in candidates]
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
        targets = torch.cat([-100*torch.ones(num_cand, input_seq_len+self.config["num_bridge_tokens"], dtype=torch.long, device=self.device), \
                             cand_targets], dim=1)
        # concatenate the inputs for the model
        attention_mask = torch.cat([attention_mask.repeat_interleave(num_cand, dim=0), candidates_att], dim=1)
        full_input_ids = torch.cat([input_ids.repeat_interleave(num_cand, dim=0), candidates_ids], dim=1)

        with torch.inference_mode():
            outputs = self.model.forward_likelihood(
                vision_input=vision_input.repeat_interleave(num_cand, dim=0),
                input_ids=full_input_ids,
                input_atts=attention_mask,
                labels=targets,
                likelihood_reduction=likelihood_reduction
            )
        neg_likelihood = outputs
        if likelihood_reduction == 'none':
            return neg_likelihood
        output_class_ranks = torch.argsort(neg_likelihood, dim=-1)[0].item()

        return output_class_ranks
    
    @torch.no_grad()
    def raw_batch_predict(self, image_list, question_list, candidates, likelihood_reduction='sum'):
        preds = [self.raw_predict(image=image, prompt=question, candidates=cands, likelihood_reduction=likelihood_reduction) for image, question, cands in zip(image_list, question_list, candidates)]

        return preds
    
    def forward(self, image, prompt, candidates=None, temperature=1, max_new_tokens=30):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates)
        else:
            raise NotImplementedError
    
def get_lynx(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_name', 'device', 'half', 'inference_method']
        target_args = ['model_config', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    # proc = SingleChoiceProcessor(' ', '\n', roles=['Question', 'Answer'], infer_method=model_args['inference_method'])
    proc = ConvSingleChoiceProcessor('\n', roles=['User', 'Bot'], \
                                     sep_style='one', infer_method=model_args['inference_method'], response_prefix='The answer is')
    return Lynx_Interface(**model_args), proc
