from .adapterv2_llama import llama_adapter
from .adapterv2_llama.utils import format_prompt
import torch
import torch.nn as nn
from .adapterv2_llama.tokenizer import Tokenizer
from PIL import Image
import cv2
from .utils import get_image

from utils.preprocessors import ConvSingleChoiceProcessor, SingleChoiceProcessor


class llama_adapterv2_Interface(nn.Module):
    def __init__(self, model_name='llama_adapterv2', model_path='/remote-home/share/multimodal-models/pyllama_data', device=None, half=False, inference_method='generation') -> None:
        super(llama_adapterv2_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.pretrained_ckpt = model_path
        self.prec_half = half

        self.model, self.preprocess = llama_adapter.load(name='BIAS-7B',llama_dir=self.pretrained_ckpt, max_seq_len=1400)
        self.tokenizer = self.model.tokenizer # Tokenizer(model_path=model_path+'/tokenizer.model')

        # self.model = self.model.half()
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     self.model.to(dtype=torch.bfloat16)
            # else:
            #     self.model.to(dtype=torch.float16)
            self.model = self.model.half()
        # print(self.model.dtype)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #        print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

        self.model.to(self.device)

        self.inference_method = inference_method
        self.max_batch_size = 1

        # self.model.eval()

    @torch.no_grad()
    def raw_generate(self, images, prompt, temperature=0.1, max_new_tokens=30):
         
        # prompt = format_prompt(prompt)
        # print(prompt)
        # images = Image.fromarray(cv2.imread(images))
        images = get_image(images)
        # print(type(img))
        # img = Image.open("../docs/logo_v1.png")
        # print(type(img))
        # images = [Image.open(image) for image in images]
        img = self.preprocess(images).unsqueeze(0).to(self.device)
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     img = img.to(torch.bfloat16)
            # else:
            #     img = img.to(torch.float16)
            img = img.to(torch.float16)
        # print('img',len(img))
        # print('prompt',len([prompt]))

        result = self.model.generate(img, [prompt], max_gen_len=max_new_tokens, temperature=temperature)[0]

        return result

    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.1, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature=temperature, max_new_tokens=max_new_tokens) for img, question in zip(image_list, question_list)]
        return outputs
    
    @torch.no_grad()
    def raw_predict(self, images, prompts, candidates, likelihood_reduction='sum'):
        # if not isinstance(images, list):
        #     images=[images]
        # images = []
        # images = Image.fromarray(cv2.imread(images))
        images = get_image(images)
        images = self.preprocess(images).unsqueeze(0).to(self.device)
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     images = images.to(torch.bfloat16)
            # else:
            #     images = images.to(torch.float16)
            images = images.to(torch.float16)
            
        images = images.repeat_interleave(len(candidates),dim=0)

        language = torch.tensor(self.tokenizer.encode(prompts,bos=True, eos=False)).unsqueeze(0)
        language = language.to(self.device)

        #prepare inputs for the input part
        input_ids = language.repeat_interleave(len(candidates),dim=0)

        all_candidates_tokens=[]
        #tokenize the candidates
        for cand in candidates:
            candidates_tokens = torch.tensor(self.tokenizer.encode(
            cand, bos=True, eos=False)).unsqueeze(0).to(self.device)
            all_candidates_tokens.append(candidates_tokens)

        max_length = max(tensor.shape[1] for tensor in all_candidates_tokens)
        padded_candidates = torch.ones(len(candidates), max_length).long().to(self.device)*self.tokenizer.pad_id
        for i, tensor in enumerate(all_candidates_tokens):
            padded_candidates[i, :tensor.shape[1]] = tensor


        #construct the inputs_ids and LM targets
        candidates_ids = padded_candidates[:, 1:]

        #mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_id, -100)

        #mask the targets for inputs part
        targets = torch.cat([-100*torch.ones_like(input_ids), cand_targets], dim=1)

        #concatenate the inputs for the model
        candidates_ids[candidates_ids == self.tokenizer.pad_id] = 0
        input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                tokens=input_ids
                start_pos=0
                visual_query = self.model.forward_visual(images)
                labels = targets
            # results = model.forward_inference(
            # visual_query = model.forward_visual(images),
            # labels = targets,
            # tokens = input_ids,
            # start_pos = 0)

                _bsz, seqlen = tokens.shape
                h = self.model.llama.tok_embeddings(tokens)
                freqs_cis = self.model.llama.freqs_cis.to(h.device)
                freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
                mask = None
                mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

                for layer in self.model.llama.layers[:-1 * self.model.query_layer]:
                    h = layer(h, start_pos, freqs_cis, mask)

                adapter = self.model.adapter_query.weight.reshape(self.model.query_layer, self.model.query_len, -1).unsqueeze(1)
                adapter_index = 0
                for layer in self.model.llama.layers[-1 * self.model.query_layer:]:
                    dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
                    dynamic_adapter = dynamic_adapter + visual_query
                    h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
                    adapter_index = adapter_index + 1

                h = self.model.llama.norm(h)
                # print(h.shape)
                output = self.model.llama.output(h[:, :-1, :]).contiguous()
                # print(output.shape)
                labels = labels[:, 1:].contiguous()

                from torch.nn import CrossEntropyLoss
                loss_fct = CrossEntropyLoss(reduction='none')
                vocab_size = output.shape[-1]
                shift_logits = output.view(-1, vocab_size)
                shift_labels_ids = labels.view(-1)
                    
                shift_labels_ids = shift_labels_ids.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels_ids)
                loss = loss.view(output.size(0), -1)
                
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
    def raw_chunk_predict(self, images, prompts, candidates, likelihood_reduction='sum'):
        # if not isinstance(images, list):
        #     images=[images]
        # images = []
        # images = Image.fromarray(cv2.imread(images))
        images=get_image(images)
        images = self.preprocess(images).unsqueeze(0).to(self.device)
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     images = images.to(torch.bfloat16)
            # else:
            #     images = images.to(torch.float16)
            images = images.to(torch.float16)
        # images = images.repeat_interleave(len(candidates),dim=0)

        language = torch.tensor(self.tokenizer.encode(prompts,bos=True, eos=False)).unsqueeze(0)
        language = language.to(self.device)

        #prepare inputs for the input part
        input_ids = language.repeat_interleave(len(candidates),dim=0)

        all_candidates_tokens=[]
        #tokenize the candidates
        for cand in candidates:
            candidates_tokens = torch.tensor(self.tokenizer.encode(
            cand, bos=True, eos=False)).unsqueeze(0).to(self.device)
            all_candidates_tokens.append(candidates_tokens)

        max_length = max(tensor.shape[1] for tensor in all_candidates_tokens)
        padded_candidates = torch.ones(len(candidates), max_length).long().to(self.device)*self.tokenizer.pad_id
        for i, tensor in enumerate(all_candidates_tokens):
            padded_candidates[i, :tensor.shape[1]] = tensor


        #construct the inputs_ids and LM targets
        candidates_ids = padded_candidates[:, 1:]

        #mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_id, -100)

        #mask the targets for inputs part
        targets = torch.cat([-100*torch.ones_like(input_ids), cand_targets], dim=1)

        #concatenate the inputs for the model
        candidates_ids[candidates_ids == self.tokenizer.pad_id] = 0
        input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                visual_query = self.model.forward_visual(images)
                loss_across_cands = []
                assert self.max_batch_size == 1
                for i in range(len(candidates)):
                    start_index = i*self.max_batch_size
                    end_index = min(len(candidates), (i+1)*self.max_batch_size)
                    tokens=input_ids[start_index:end_index]
                    start_pos=0
                    labels = targets[start_index:end_index]
                # results = model.forward_inference(
                # visual_query = model.forward_visual(images),
                # labels = targets,
                # tokens = input_ids,
                # start_pos = 0)

                    _bsz, seqlen = tokens.shape
                    h = self.model.llama.tok_embeddings(tokens)
                    freqs_cis = self.model.llama.freqs_cis.to(h.device)
                    freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
                    mask = None
                    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
                    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

                    for layer in self.model.llama.layers[:-1 * self.model.query_layer]:
                        h = layer(h, start_pos, freqs_cis, mask)

                    adapter = self.model.adapter_query.weight.reshape(self.model.query_layer, self.model.query_len, -1).unsqueeze(1)
                    adapter_index = 0
                    for layer in self.model.llama.layers[-1 * self.model.query_layer:]:
                        dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
                        dynamic_adapter = dynamic_adapter + visual_query
                        h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
                        adapter_index = adapter_index + 1

                    h = self.model.llama.norm(h)
                    # print(h.shape)
                    output = self.model.llama.output(h[:, :-1, :]).contiguous()
                    # print(output.shape)
                    labels = labels[:, 1:].contiguous()

                    from torch.nn import CrossEntropyLoss
                    loss_fct = CrossEntropyLoss(reduction='none')
                    vocab_size = output.shape[-1]
                    shift_logits = output.view(-1, vocab_size)
                    shift_labels_ids = labels.view(-1)
                        
                    shift_labels_ids = shift_labels_ids.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels_ids)
                    loss = loss.view(output.size(0), -1)
                    loss_across_cands.append(loss)
                    
                loss = torch.cat(loss_across_cands, dim=0)
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
        preds = [self.raw_chunk_predict(image, question, cands) for image, question, cands in zip(image_list, question_list, candidates)]
        return preds
    
    def forward(self, image, prompt, candidates=None, temperature=0.1, max_new_tokens=30):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates)
        else:
            raise NotImplementedError

def get_llama_adapterv2(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_type', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    # print(model_args)
    model = llama_adapterv2_Interface(**model_args)
    preprocessor = ConvSingleChoiceProcessor(sep='\n\n### ', infer_method=model_args['inference_method'],\
                           system_msg='Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.', \
                           roles=['Input', 'Response'], sep_style="llama_adapter2", response_prefix='The answer is')
    # preprocessor = SingleChoiceProcessor(' ', '\n', roles=['Question', 'Answer'], infer_method=model_args['inference_method'])
    return model, preprocessor
if __name__=='__main__':
    model = llama_adapterv2_Interface()