from .minigpt4.common.config import Config
from .minigpt4.common.dist_utils import get_rank
from .minigpt4.common.registry import registry
from .minigpt4.conversation.conversation import Chat, CONV_VISION, StoppingCriteriaSub
from transformers import StoppingCriteria, StoppingCriteriaList
from omegaconf import OmegaConf
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
from .utils import get_image
import torch
import torch.nn as nn
from utils.preprocessors import BaseProcessor, SingleChoiceProcessor, ConvSingleChoiceProcessor

# notice that miniGPT-4 requires relative import since it does not enable absolute installation
class MiniGPT4_Interface(nn.Module):
    def __init__(self, model_path="facebook/opt-350m", model_type=None, device=None, half=False, inference_method='generation') -> None:
        super(MiniGPT4_Interface, self).__init__()
        # get the model config
        config = OmegaConf.load(model_path)
        tmp_model_config = config.get('model', None)
        assert tmp_model_config is not None, "Missing model configuration file."

        model_cls = registry.get_model_class(tmp_model_config.arch)
        assert model_cls is not None, f"Model '{tmp_model_config.arch}' has not been registered."

        if model_type is None:
            model_type = tmp_model_config.get("model_type", None)
        assert model_type is not None, "Missing model_type."

        # get the default config path
        model_config_path = model_cls.default_config_path(model_type=model_type)

        # get the final model config
        model_config = OmegaConf.create()
        # hierarchy override, customized config > default config
        model_config = OmegaConf.merge(
            model_config,
            OmegaConf.load(model_config_path),
            {"model": config["model"]},
        ).model
        model_config.low_resource = False
        model_config.prompt_path = None

        # setup the cuda device
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # load the model!
        self.model = model_cls.from_config(model_config).to(self.device)

        # load visual processor
        vis_processor_cfg = config.datasets.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # setup the inference method
        self.inference_method = inference_method

        # setup the conversation template
        self.conv = CONV_VISION.copy()

    def get_conv(self):
        return self.conv

    @torch.no_grad()
    def raw_generate(self, image_list, conv, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        # conv.append_message(conv.roles[1], None)
        if not isinstance(image_list, list):
            image_list = [image_list]
        raw_images = [get_image(image) for image in image_list] 
        img_list = [self.vis_processor(raw_image).unsqueeze(0).to(self.device) for raw_image in raw_images]
        img_emb_list = [self.model.encode_img(image)[0] for image in img_list]
        embs = self.get_context_emb(conv, img_emb_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        # conv.messages[-1][1] = output_text
        return output_text
    
    def get_context_emb(self, prompt, img_list):
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def get_first_query_process(self):
        return lambda qs: "<Img><ImageHere></Img>" + ' ' + qs
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature=temperature, max_new_tokens=max_new_tokens) for img, question in zip(image_list, question_list)]
        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image_list, prompt, candidates, likelihood_reduction='sum'):
        if not isinstance(image_list, list):
            image_list = [image_list]
        raw_images = [get_image(image) for image in image_list] 
        img_list = [self.vis_processor(raw_image).unsqueeze(0).to(self.device) for raw_image in raw_images]
        img_emb_list = [self.model.encode_img(image)[0] for image in img_list]
        
        # get the embedding from the input
        num_cand = len(candidates)
        embs = self.get_context_emb(prompt, img_emb_list)
        input_seq_len = embs.shape[1]
        attention_mask = torch.ones(num_cand, input_seq_len, dtype=torch.long, device=self.device)

        # tokenize the candidates
        current_padding_side = self.model.llama_tokenizer.padding_side
        current_truncation_side = self.model.llama_tokenizer.truncation_side
        self.model.llama_tokenizer.padding_side = 'right'
        self.model.llama_tokenizer.truncation_side = 'right'
        candidates_tokens = self.model.llama_tokenizer(
            candidates,
            return_tensors='pt',
            padding='longest'
        ).to(self.device)
        self.model.llama_tokenizer.padding_side = current_padding_side
        self.model.llama_tokenizer.truncation_side = current_truncation_side

        # construct the inputs_ids and LM targets
        candidates_ids = candidates_tokens.input_ids[:, 1:] # remove the <s> token
        candidates_att = candidates_tokens.attention_mask[:, 1:] # remove the <s> token
        # mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.model.llama_tokenizer.pad_token_id, -100)
        # mask the targets for inputs part
        targets = torch.cat([-100*torch.ones(num_cand, input_seq_len, dtype=torch.long, device=self.device), \
                             cand_targets], dim=1)
        # concatenate the inputs for the model
        candidates_emb = self.model.llama_model.model.embed_tokens(candidates_ids)
        attention_mask = torch.cat([attention_mask, candidates_att], dim=1)
        embs = torch.cat([embs.repeat_interleave(num_cand, dim=0), candidates_emb], dim=1)

        with torch.inference_mode():
            outputs = self.model.llama_model.forward_likelihood(
                inputs_embeds=embs,
                attention_mask=attention_mask,
                labels=targets,
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
        preds = [self.raw_predict(image, question, cands, likelihood_reduction=likelihood_reduction) for image, question, cands in zip(image_list, question_list, candidates)]

        return preds
    
    def forward(self, image, prompt, candidates=None, temperature=0.2, max_new_tokens=30, likelihood_reduction='sum'):
        # ## 20230904 add for multiple images
        # if len(prompt[0].split('<ImageHere>')) > 2:
        #     prompt = [_.replace('###Human: <Img><ImageHere></Img>','###Human:') for _ in prompt]
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates, likelihood_reduction=likelihood_reduction)
        else:
            raise NotImplementedError
    
def get_minigpt4(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_name', 'model_type', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'model_type', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = MiniGPT4_Interface(**model_args)
    conv = model.get_conv()
    first_query_process_fn = model.get_first_query_process()
    if conv.sep_style.name == 'SINGLE':
        sep_style = 'one'
    elif conv.sep_style.name == 'TWO':
        sep_style = 'two'
    proc = ConvSingleChoiceProcessor(conv.sep, sep2=conv.sep2, roles=conv.roles, system_msg=conv.system, \
                                     first_query_fn=first_query_process_fn, init_conv=conv.messages, \
                                     sep_style=sep_style, infer_method=model_args['inference_method'], \
                                     response_prefix='The answer is')
    return model, proc

if __name__=='__main__':
    model = MiniGPT4_Interface()