from .pandagpt_model.openllama import OpenLLAMAPEFTModel
from .pandagpt_model.modeling_llama import LlamaForCausalLM
from .pandagpt_model.header import *
from torch.nn.utils import rnn
import torch
from utils.preprocessors import ConvSingleChoiceProcessor
from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

class pandagpt_Interface(nn.Module):
    def __init__(self, model_name='pandagpt', model_path='/remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/', device=None, half=False, inference_method='generation') -> None:
        super(pandagpt_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)


        self.model_name = model_name
        self.pretrained_ckpt = model_path
        self.prec_half = half


        args = {
            'model': 'openllama_peft',
            'imagebind_ckpt_path': self.pretrained_ckpt +'imagebind_ckpt',
            'vicuna_ckpt_path': self.pretrained_ckpt + 'vicuna_ckpt/7b_v0',
            'delta_ckpt_path': self.pretrained_ckpt + 'pandagpt_ckpt/7b/pytorch_model.pt',
            'stage': 2,
            'max_tgt_len': 30,
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }

        self.model = OpenLLAMAPEFTModel(**args)

        # if self.prec_half:
        #     if torch.cuda.is_bf16_supported():
        #         self.model.to(dtype=torch.bfloat16)
        #     else:
        #         self.model.to(dtype=torch.float16)
        if self.prec_half:
            self.model = self.model.half()
                
        self.model.to(self.device)

        self.tokenizer = self.model.llama_tokenizer

        self.inference_method = inference_method

    def extract_multimodal_feature(self, inputs):
        features = []
        if inputs['image_paths']:
            image_embeds, _ = self.model.encode_image(inputs['image_paths'])
            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.model.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.model.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['thermal_paths']:
            thermal_embeds, _ = self.model.encode_thermal(inputs['thermal_paths'])
            features.append(thermal_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds

    def build_one_instance(self, tokenizer, conversation):
        text_list = []
        turn_num = len(conversation)
        input_ids, target_ids = [], []
        for i in range(turn_num):
            turn = conversation[i]
            # role = turn['from']
            if i == 0: # the first human turn
                # assert role == 'human'
                # text = '</Img> ' + turn['value'] + '\n### Assistant:'
                text = turn
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
            else:
                # if role == 'human':
                #     text = 'Human: ' + turn['value'] + '\n### Assistant:'
                    text = turn
                    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                    input_ids += one_input_id
                    target_ids += [-100]*len(one_input_id)
                # elif role == 'gpt':
                #     text = turn['value'] + '\n###'
                    # one_input_id = self.tokenizer(text, add_special_tokens=False).input_ids
                    # input_ids += one_input_id
                    # target_ids += one_input_id
                # else:
                #     raise Exception('Wrong Role!!!')
            text_list.append(text)
            assert len(input_ids) == len(target_ids)
        return text_list, input_ids, target_ids

    def process_batch_instance(self, tokenizer, batch_of_conversations, max_tgt_len):
        batch_input_ids, batch_target_ids = [], []
        for conversation in batch_of_conversations:
            _, one_input_ids, one_target_ids = self.build_one_instance(tokenizer, conversation)
            batch_input_ids.append(torch.LongTensor(one_input_ids))
            batch_target_ids.append(torch.LongTensor(one_target_ids))
        input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
        assert input_ids.size() == target_ids.size()
        input_ids = input_ids[:,:max_tgt_len]
        target_ids = target_ids[:,:max_tgt_len]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        assert attention_mask.size() == input_ids.size()
        return input_ids, target_ids, attention_mask.long()
    
    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        if len(inputs['modality_embeds']) == 1:
            feature_embeds = inputs['modality_embeds'][0]
        else:
            feature_embeds = self.extract_multimodal_feature(inputs)
            inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = '### Human: <Img>'
        p_before_tokens = self.tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.model.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        # text = '</Img> ' + prompt + '\n### Assistant:'
        text=prompt
        p_after_tokens = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.model.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.model.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        return inputs_embeds


    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = '### Human: <Img>'
        p_before_tokens = self.tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.model.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        p_after_embeds = self.model.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                            dtype=p_before_tokens.input_ids.dtype,
                            device=p_before_tokens.input_ids.device) * self.tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.model.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim

        # create targets
        empty_targets = (
            torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
        ) # bsz x (1 + s1 + 1)
        targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], dtype=torch.long).to('cuda') # bsz x (1 + s1 +1)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
        return inputs_embeds, targets, attention_mask 



    def raw_generate(self, image, prompt, temperature=0.1, max_new_tokens=30):

        inputs = {
            'prompt': prompt[16:],
            'image_paths': [image],
            'audio_paths': None,
            'video_paths': None,
            'modality_embeds': [],
            'thermal_paths': None,
            'top_p': 0.7,
            'temperature': temperature,
            'max_tgt_len': max_new_tokens,
        }

        input_embeds = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.model.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text


    @torch.no_grad()
    def raw_batch_generate(self, image_list, inputs_list, temperature=0.1, max_new_tokens=30):
        outputs = [self.raw_generate(img, inputs, temperature=temperature, max_new_tokens=max_new_tokens) for img, inputs in zip(image_list, inputs_list)]
        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
        prompt = prompt[16:]
        image, _ = self.model.encode_image([image])
        image = image
        # print(image.shape)
        # input_ids, target_ids, attention_mask = process_batch_instance(tokenizer, [prompt], len(prompt))

        # # preprocess the language
        language = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
        # # if self.prec_half:
        # #     lang_x = lang_x.to(dtype=torch.float16)
        # # elif torch.cuda.is_bf16_supported():
        # #     lang_x = lang_x.to(dtype=torch.bfloat16)

        input_ids = language["input_ids"]
        attention_mask = language["attention_mask"]
        # print(input_ids.shape)
        # print(attention_mask.shape)

        # prepare inputs for the input part
        # input_ids = input_ids.repeat_interleave(len(candidates),dim=0)
        # attention_mask = attention_mask.repeat_interleave(len(candidates),dim=0)
        target_ids = torch.ones_like(input_ids)*-100
        # print(input_ids.shape)
        
        inputs_embeds, targets, attention_mask = self.prompt_wrap(image, input_ids, target_ids, attention_mask)
        # print(targets.shape)
        # print(attention_mask.shape)

        # prepare inputs for the input part
        # input_ids = input_ids.repeat_interleave(len(candidates),dim=0).to('cuda')
        attention_mask = attention_mask.repeat_interleave(len(candidates),dim=0).to(self.device)
        inputs_embeds = inputs_embeds.repeat_interleave(len(candidates),dim=0).to(self.device)
        targets = targets.repeat_interleave(len(candidates),dim=0).to(self.device)
        

        # tokenize the candidates
        current_padding_side = self.tokenizer.padding_side
        current_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        candidates_tokens = self.tokenizer(
            [cand for cand in candidates],
            return_tensors = 'pt',
            padding = 'longest',
        ).to(self.device)


        self.tokenizer.padding_side = current_padding_side
        self.tokenizer.truncation_side = current_truncation_side

        #construct the inputs_ids and LM targets
        candidates_ids = candidates_tokens.input_ids[:, 1:]
        candidates_att = candidates_tokens.attention_mask[:, 1:]
        candidates_embeds = self.model.llama_model.model.model.embed_tokens(candidates_ids).expand(len(candidates), -1, -1)# [:,1:]

        #mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_token_id, -100)

        #mask the targets for inputs part
        targets = torch.cat([targets, cand_targets], dim=1)
        # print(targets.shape)
        # print('a')

        #concatenate the inputs for the model
        # input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        attention_mask = torch.cat([attention_mask, candidates_att], dim=1)
        # print(attention_mask.shape)
        # print(input_ids.shape)
        input_embeds = torch.cat([inputs_embeds, candidates_embeds], dim=1)
        # print(attention_mask.shape)

        with torch.inference_mode():
            output = self.model.llama_model.forward(inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # query_embeds = image.repeat_interleave(len(candidates),dim=0).to(self.device), 
            labels = targets,
            return_dict = True)
        
        logits = output.logits
        labels_ids = targets

        #Shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels_ids = labels_ids[..., 1:].contiguous()

        #Flatten the tokens
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss(reduction='none')
        vocab_size = logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels_ids = shift_labels_ids.view(-1)

        shift_labels_ids = shift_labels_ids.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels_ids)
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
    
    def forward(self, image, prompt, candidates=None, temperature=0.1, max_new_tokens=30):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates)
        else:
            raise NotImplementedError

def get_pandagpt(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_type', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    # print(model_args)
    model = pandagpt_Interface(**model_args)
    # if model_args['inference_method'] == 'likelihood':
        # preprocessor = ConvSingleChoiceProcessor(sep='\n', sep2='\n', infer_method=model_args['inference_method'],\
        #                    roles=['### Human', '### Assistant'], sep_style="two")
        # preprocessor = ConvSingleChoiceProcessor(sep='\n', sep2='\n', infer_method='likelihood',\
        #                                     first_query_fn=lambda x: "<Img></Img> " + x, \
        #                 roles=['### Human', '### Assistant'], sep_style="two") 
    # else:
        # preprocessor = ConvSingleChoiceProcessor(sep='\n', sep2='\n', infer_method=model_args['inference_method'],\
        #                    system_msg='', roles=['### Human', '### Assistant'], sep_style="two")     
    preprocessor = ConvSingleChoiceProcessor(sep='\n', sep2='\n', infer_method=model_args['inference_method'],\
                                            first_query_fn=lambda x: "<Img></Img> " + x, \
                        roles=['### Human', '### Assistant'], sep_style="two",  response_prefix='The answer is')    
    return model, preprocessor
if __name__=='__main__':
    model = pandagpt_Interface()