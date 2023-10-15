from .ImageBind import data as data
from .imagebind_llama import *
import torch
import torch.nn as nn
from .imagebind_llama.tokenizer import Tokenizer
from torchvision import transforms

from utils.preprocessors import ConvSingleChoiceProcessor


class imagebindLLM_Interface(nn.Module):
    def __init__(self, model_name='imagebindLLM', model_path='/remote-home/share/multimodal-models/imagebindllm_ckpts', device=None, half=False, inference_method='generation') -> None:
        super(imagebindLLM_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.llama_dir = '/remote-home/share/multimodal-models/pyllama_data'
        self.pretrained_ckpt = model_path
        self.prec_half = half

        self.model = llama_adapter.load(self.pretrained_ckpt+'/7B.pth', self.llama_dir, knn=True)

        # if self.prec_half:
        #     self.model.to(dtype=torch.float16)
        # else:
        #     self.model.to(dtype=torch.bfloat16)

        if self.prec_half:
            self.model = self.model.half()
            
        self.model.to(self.device)

        self.tokenizer = Tokenizer(model_path=self.llama_dir+'/tokenizer.model')

        self.inference_method = inference_method
        self.model.eval()

    def load_and_transform_vision_data(image_paths, device):
        if image_paths is None:
            return None

        image_ouputs = []
        for image_path in image_paths:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
            if type(image_path) == str:
                with open(image_path, "rb") as fopen:
                    image = Image.open(fopen).convert("RGB")
            else:
                image = image_path

            image = data_transform(image).to(device)
            image_ouputs.append(image)
        return torch.stack(image_ouputs, dim=0)
    
    @torch.no_grad()
    def raw_generate(self,image, prompt, temperature=0.1, max_new_tokens=30):
        if not isinstance(image, list):
            image = [image]
        inputs = {}
        image = data.load_and_transform_vision_data(image, device=self.device)
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     image = image.to(torch.bfloat16)
            # else:
            #     image = image.to(torch.float16)
            image = image.to(torch.float16)
        
        inputs['Image'] = [image, 1]

        results = self.model.generate(
            inputs,
            [prompt],
            max_gen_len=max_new_tokens,
            temperature = temperature,
            top_p = 0.75
        )
        result = results[0].strip()
        return result
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.1, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature=temperature, max_new_tokens=max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs

    @torch.no_grad()
    def raw_predict(self, images, prompts, candidates, likelihood_reduction='sum'):
        if not isinstance(images, list):
            images=[images]
        # images = []
        # inputs = {}
        images = data.load_and_transform_vision_data(images, device=self.device)
        images = images.repeat_interleave(len(candidates),dim=0)
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     images = images.to(torch.bfloat16)
            # else:
            #     images = images.to(torch.float16)
            images = images.to(torch.float16)

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
        final_input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                tokens=final_input_ids
                start_pos=0
                visual_query = self.model.forward_visual({'Image': [images, 1]})
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
                    h = layer(h, 0, freqs_cis, mask)
                prefix_query = self.model.prefix_query.weight.reshape(
                    self.model.query_layer, 1, 4096).unsqueeze(1)
                prefix_index = 0
                visual_proj = visual_query
                for layer in self.model.llama.layers[-1 * self.model.query_layer:]:
                    h = layer(h, 0, freqs_cis, mask, visual_proj + prefix_query[prefix_index])
                    prefix_index = prefix_index + 1

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
        if not isinstance(images, list):
            images=[images]
        # images = []
        # inputs = {}
        images = data.load_and_transform_vision_data(images, device=self.device)
        # images = images.repeat_interleave(len(candidates),dim=0)
        if self.prec_half:
            # if torch.cuda.is_bf16_supported():
            #     images = images.to(torch.bfloat16)
            # else:
            #     images = images.to(torch.float16)
            images = images.to(torch.float16)

        language = torch.tensor(self.tokenizer.encode(prompts,bos=True, eos=False)).unsqueeze(0)
        input_ids = language.to(self.device)

        with torch.cuda.amp.autocast():
            visual_query = self.model.forward_visual({'Image': [images, 1]})

        #prepare inputs for the input part
        # input_ids = language.repeat_interleave(len(candidates),dim=0)

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
        
        loss_list=[]

        for i in range(len(candidates)):
            #construct the inputs_ids and LM targets
            candidates_ids = padded_candidates[i:i+1, 1:]

            #mask the LM targets with <pad>
            cand_targets = candidates_ids.clone()
            cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_id, -100)

            #mask the targets for inputs part
            targets = torch.cat([-100*torch.ones_like(input_ids), cand_targets], dim=1)

            #concatenate the inputs for the model
            candidates_ids[candidates_ids == self.tokenizer.pad_id] = 0
            final_input_ids = torch.cat([input_ids, candidates_ids], dim=1)
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast():
                    tokens=final_input_ids
                    start_pos=0
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
                        h = layer(h, 0, freqs_cis, mask)
                    prefix_query = self.model.prefix_query.weight.reshape(
                        self.model.query_layer, 1, 4096).unsqueeze(1)
                    prefix_index = 0
                    visual_proj = visual_query
                    for layer in self.model.llama.layers[-1 * self.model.query_layer:]:
                        h = layer(h, 0, freqs_cis, mask, visual_proj + prefix_query[prefix_index])
                        prefix_index = prefix_index + 1

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
                        # return loss
                    else:
                        raise ValueError
            
            loss_list.append(loss)
        
        loss_list = torch.cat(loss_list, dim=0)
        if likelihood_reduction == 'none':
            return loss_list
        output_class_ranks = torch.argsort(loss_list, dim=-1)[0].item()

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

def get_imagebindLLM(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_type', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    # print(model_args)
    model = imagebindLLM_Interface(**model_args)
    # preprocessor = ConvSingleChoiceProcessor(sep='\n', infer_method=model_args['inference_method'],\
    #                        system_msg='', roles=['### Input', '### Response'], sep_style="one")
    # preprocessor = ConvSingleChoiceProcessor(sep='\n\n### ', infer_method=model_args['inference_method'],\
    #                     system_msg='Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.', \
    #                     roles=['Input', 'Response'], sep_style="llama_adapter2", response_prefix='The answer is')
    preprocessor = ConvSingleChoiceProcessor(sep='\n\n### ', infer_method=model_args['inference_method'],\
                        system_msg='Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.', \
                        roles=['Input', 'Response'], sep_style="llama_adapter2", response_prefix='The answer is')
                         
    return model, preprocessor
if __name__=='__main__':
    model = imagebindLLM_Interface()