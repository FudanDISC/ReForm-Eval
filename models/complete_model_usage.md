### Complete Model Usage
For model-related parameters, we list required parameters of all 16 models. When running the evaluation, these commands must be applied for specific models.

**Note: Some models require additional forward_likelihood function, please refer to `Likelihood-based Black-Box Evaluation` in [Create Your Own Model Interface](#create-your-own-model-interface)**

#### BLIP-2 + InstructBLIP
```bash
# BLIP-2 flant5
--model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl
# InstructBLIP flan-t5
--model blip2  --model_name blip2_t5_instruct  --model_type flant5xl
# InstructBLIP vicuna
--model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b
```
You also have to put `bert-base-uncased` and `google/flan-t5-xl` folders on the root directory of our repository.
```
|-- ReForm-Eval
    |-- bert-base-uncased
    |-- google
        |-- flan-t5-xl
        ...
    |-- build
    |-- commands
    |-- metrics
    |-- models
    ...
```

If you load `blip2_t5`, you need to add the `predict_class` function in `blip2_t5.py`.
```python
    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - prompt: the instruction
            candidates:
                (list): A list of candidate class names;
            n_segments:
                (int): Split the candidates into n_segments and predict one by one. This is useful when the number of candidates is too large.
        Returns:
            output_class: predicted class index
        """

        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
 
        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                frame_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )

                frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            candidates, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        n_cands = len(candidates)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
            )

            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                # this_encoder_outputs = copy.deepcopy(encoder_outputs)
                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs['last_hidden_state'] = this_encoder_outputs[0].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = encoder_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
                this_output_tokens_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(this_output_tokens_ids == self.t5_tokenizer.pad_token_id, -100)

                outputs = self.t5_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks
```

Then, you should run the following command to implement the modification.
```
cd models/LAVIS
pip install e .
```

#### LLaVA
```bash
# LLaVA v0
--model llava  --model_name /path/to/LLaVA-7B-v0/
# LLaVA llama-2
--model llava  --model_name /path/to/llava-llama-2-7b-chat-lightning-lora-preview/ \
--model_type /path/to/Llama-2-7b-chat-hf/
```
#### MiniGPT-4
```bash
--model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml
```

In `PATH_TO_REFORM-EVAL/models/MiniGPT-4/eval_configs/minigpt4_eval.yaml`, you have to set:
```YAML
ckpt: '/path/to/prerained_minigpt4_7b.pth'
```

In `PATH_TO_REFORM-EVAL/models/interfaces/minigpt4/configs/models/minigpt4.yaml`, you need to set:
```YAML
# Vicuna
llama_model: "/path/to/vicuna-7B-v0/"
```
#### mPLUG-Owl
```bash
--model mplugowl  --model_name mplugowl --model_type /path/to/mplug-owl-llama-7b/
```
#### LLaMA-Adapter V2
```bash
--model llama_adapterv2  --model_name llama_adapterv2  --model_type /path/to/pyllama_data
```
#### ImageBind-LLM
```bash
--model imagebindLLM  --model_name imagebindLLM --model_type /path/to/imagebindllm_ckpts
```

You need to modify the ImageBindLLM interface:
```python
class imagebindLLM_Interface(nn.Module):
    def __init__(self, model_name='imagebindLLM', model_path='/path/to/imagebindllm_ckpts', device=None, half=False, inference_method='generation') -> None:
        super(imagebindLLM_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.llama_dir = '/path/to/pyllama_data' # The modification place
        self.pretrained_ckpt = model_path
        self.prec_half = half
```

In `/path/to/imagebindllm_ckpts`, you need to include the following ckpts:
```
|-- imagebindllm_ckpts
    |-- 7B.pth
    |-- imagebind_w3D.pth
    |-- knn.index
    `-- PointTransformer_8192point.yaml
```
#### PandaGPT
```bash
--model pandagpt  --model_name pandagpt --model_type /path/to/pandagpt_pretrained_ckpt/
```

`/path/to/pandagpt_pretrained_ckpt` includes three folders, which are `imagebind_ckpt`, `pandagpt_ckpt` and `vicuna_ckpt`. You need to put all ckpts in corresponding folders.
#### Lynx
```bash
--model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml
```

You need to modify the config file:
```YAML
checkpoint: "/path/to/finetune_lynx.pt"
```

In `PATH_TO_REFORM-EVAL/models/interfaces/lynx/configs/LYNX.yaml`, you need to set:
```YAML
LLM_base: '/path/to/vicuna-7B-v1.1/'
```
#### Cheetor
```bash
# vicuna
--model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml

# llama 2
--model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml
```

You need to modify the config file:
```YAML
# vicuna
ckpt: '/path/to/cheetah_vicuna_7b.pth'

# llama 2
ckpt: '/path/to/cheetah_llama2_7b.pth'
```

In `PATH_TO_REFORM-EVAL/models/interfaces/Cheetah/cheetah/configs/models/cheetah_vicuna(llama2).yaml`, you need to set:
```YAML
# Vicuna
llama_model: "/path/to/vicuna-7B-v0/"

# llama2
llama_model: "/path/to/Llama-2-7b-chat-hf/"
```

#### Shikra
```bash
--model shikra  --model_name /path/to/shikra-7b-v1/
```
#### Bliva
```bash
--model bliva  --model_name bliva_vicuna
```

You need to modify the config file in `PATH_TO_REFORM-EVAL/models/BLIVA/bliva/configs/models/bliva_vicuna7b.yaml`:
```YAML
finetuned: '/path/to/bliva_vicuna7b.pth'
```
#### Multimodal GPT
```bash
--model mmgpt  --model_name Multimodal-GPT
```

In the mmGPT interface, you need to modify the following config:
```python
mmGPT_config = {
    "Multimodal-GPT": {
        "finetune_path": "/path/to/mmgpt-lora-v0-release.pt",
        "open_flamingo_path": "/path/to/OpenFlamingo-9B/checkpoint.pt",
        "llama_path": "/path/to/llama-7b-hf/"
    }
}
response_split = "### Response:"

class MultimodalGPT_Interface(nn.Module):
```