from mmgpt.models.builder import create_model_and_transforms
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
from .utils import get_image
import torch
import torch.nn as nn
from utils.preprocessors import BaseProcessor, MMGPTSingleChoiceProcessor, ConvSingleChoiceProcessor

mmGPT_config = {
    "Multimodal-GPT": {
        "finetune_path": "/remote-home/share/multimodal-models/mmgpt/mmgpt-lora-v0-release.pt",
        "open_flamingo_path": "/remote-home/share/multimodal-models/mmgpt/OpenFlamingo-9B/checkpoint.pt",
        "llama_path": "/remote-home/share/LLM_CKPT/llama-7b-hf/"
    }
}
response_split = "### Response:"

class MultimodalGPT_Interface(nn.Module):
    def __init__(self, finetune_path=None, llama_path=None, open_flamingo_path=None, device=None, half=False, inference_method='generation') -> None:
        super(MultimodalGPT_Interface, self).__init__()
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)

        # load the fine-tuning ckpt
        ckpt = torch.load(finetune_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # remove the "module." prefix
            state_dict = {
                k[7:]: v
                for k, v in state_dict.items() if k.startswith("module.")
            }
        else:
            state_dict = ckpt
        tuning_config = ckpt.get("tuning_config")
        if tuning_config is None:
            print("tuning_config not found in checkpoint")
        else:
            print("tuning_config found in checkpoint: ", tuning_config)

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            model_name="open_flamingo",
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            pretrained_model_path=open_flamingo_path,
            tuning_config=tuning_config,
        )
        self.model.load_state_dict(state_dict, strict=False)
        
        # convert to the device
        self.model.half()
        self.model.eval()
        self.model.to(self.device)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_eos_token = False

        # setup the inference method
        self.inference_method = inference_method

    def get_conv(self):
        raise NotImplementedError
    
    def get_first_query_process(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=1.0, max_new_tokens=30):
        # preprocess the image
        if not isinstance(image, list):
            image = [image]
        image = [get_image(img) for img in image]
        vision_x = [self.image_processor(im).unsqueeze(0) for im in image]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).half().to(self.device)
        
        # preprocess the language
        lang_x = self.tokenizer([prompt], return_tensors='pt').to(self.device)

        # generation
        with torch.inference_mode():
            output_ids = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                temperature=temperature,
                # top_k=top_k, check later
                top_p=1.0,
                do_sample=True,
            )[0]
        generated_text = self.tokenizer.decode(
            output_ids, skip_special_tokens=True)
        result = generated_text.split(response_split)[-1].strip()
        return result
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature, max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
         # preprocess the image
        if not isinstance(image, list):
            image = [image]
        image = [get_image(img) for img in image]
        vision_x = [self.image_processor(im).unsqueeze(0) for im in image]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).half().to(self.device)
        
        # preprocess the language
        lang_x = self.tokenizer([prompt], return_tensors='pt').to(self.device)
        input_ids = lang_x.input_ids
        attention_mask = lang_x.attention_mask
        # prepare inputs for the input part
        input_ids = input_ids.repeat_interleave(len(candidates), dim=0)
        attention_mask = attention_mask.repeat_interleave(len(candidates), dim=0)
        
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
                lang_x=input_ids,
                vision_x=vision_x.repeat_interleave(len(candidates), dim=0),
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
    
def get_mmgpt(model_config=None):
    assert model_config['model_name'] in mmGPT_config
    model_args = mmGPT_config[model_config['model_name']]
    if model_config is not None:
        valid_args = ['device', 'half', 'inference_method']
        target_args = ['device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = MultimodalGPT_Interface(**model_args)
    proc = MMGPTSingleChoiceProcessor("\n\n### ", roles=["Instruction", "Response"], \
                                     sep_style="one", infer_method=model_args['inference_method'], response_prefix='The answer is')
    return model, proc

if __name__=='__main__':
    model = MultimodalGPT_Interface()