from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import torch.nn as nn
from utils.preprocessors import BaseProcessor, SingleChoiceProcessor
import contextlib
from types import MethodType
from .utils import get_image

# def get_image(image):
#     image = Image.open(image)
#     return image.convert('RGB')

def new_maybe_autocast(self, dtype=None):
    enable_autocast = self.device != torch.device("cpu")
    if enable_autocast:
        if dtype is torch.bfloat16:
            if torch.cuda.is_bf16_supported():
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
            else:
                return contextlib.nullcontext()
        else:
            return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()
    
def half_new_maybe_autocast(self, dtype=None):
    enable_autocast = self.device != torch.device("cpu")
    if enable_autocast:
        if dtype is torch.bfloat16:
            if torch.cuda.is_bf16_supported():
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
            else:
                return torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()

class BLIP2_Interface(nn.Module):
    def __init__(self, model_name='blip2_t5', model_type='pretrain_flant5xl', device=None, half=False, inference_method='generation') -> None:
        super(BLIP2_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=self.device
        )
        self.prec_half = half
        if self.prec_half:
            self.model.maybe_autocast = MethodType(half_new_maybe_autocast, self.model)
        else:
            self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)

        # setup the inference method
        self.inference_method = inference_method

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Answer:"
        }, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, max_new_tokens=30):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        if self.prec_half:
            image = image.to(torch.float16)
        answer = self.model.generate({
            "image": image, "prompt": prompt}, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        if self.prec_half:
            imgs = imgs.to(torch.float16)
        prompts = [f"Question: {question} Answer:" for question in question_list]
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)

        return output
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=1, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0)
        if self.prec_half:
            imgs = imgs.to(torch.float16)
        imgs = imgs.to(self.device)
        prompts = question_list
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens, temperature=temperature)

        return output
    
    @torch.no_grad()
    def raw_batch_predict(self, image_list, question_list, candidates, likelihood_reduction='sum'):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0)
        if self.prec_half:
            imgs = imgs.to(torch.float16)
        imgs = imgs.to(self.device)
        prompts = question_list
        output = self.model.predict_class({"image": imgs, "prompt": prompts}, candidates, likelihood_reduction=likelihood_reduction)
        
        # transfer the rank list to the predict class
        # print(type(output))
        # print(output.shape)
        if isinstance(output, list):
            pred = [each[0] for each in output]
            return pred
        
        pred = output[:, 0].tolist()

        return pred
    
    def forward(self, image, prompt, candidates=None, temperature=1, max_new_tokens=30, likelihood_reduction='sum'):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates, likelihood_reduction=likelihood_reduction)
        else:
            raise NotImplementedError
    
def get_blip2(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_name', 'model_type', 'device', 'half', 'inference_method']
        for arg in valid_args:
            if arg in model_config:
                model_args[arg] = model_config[arg]
    proc = SingleChoiceProcessor(' ', '\n', roles=['Question', 'Answer'], infer_method=model_args['inference_method'])
    return BLIP2_Interface(**model_args), proc


def get_instructblip(model_config):
    pass