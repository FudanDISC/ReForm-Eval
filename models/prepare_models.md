## 🤖 Prepare Models

### Set Up Existing Models
We conduct a comprehensive evaluation of 16 open-source LVLMs across various capability dimensions. All LVLMs are provided with checkpoints. To build the environment for each model, you can create conda envs for corresponding models and directly copy the installation command of Bash Shell in `PATH_TO_REFORM-EVAL/models/build_scripts/` and paste it in the terminal to install required packages.

| Model            | URL                                                           |
|------------------|---------------------------------------------------------------|
| BLIP-2           | [https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) |
| InstructBLIP     | [https://github.com/salesforce/LAVIS/tree/main/projects/instructblip](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) |
| LLaVA-Vicuna     | GitHub:[https://github.com/haotian-liu/LLaVA/tree/main](https://github.com/haotian-liu/LLaVA/tree/main)<br>Model:[https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0) |
| LLaVA-LLaMA2     | GitHub:[https://github.com/haotian-liu/LLaVA/tree/main](https://github.com/haotian-liu/LLaVA/tree/main)<br>Model:[https://huggingface.co/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview](https://huggingface.co/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview) |
| MiniGPT4         | GitHub:[https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)<br>Model:[https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view) |
| mPLUG-Owl        | GitHub:[https://github.com/X-PLUG/mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)<br>Model:[https://huggingface.co/MAGAer13/mplug-owl-llama-7b](https://huggingface.co/MAGAer13/mplug-owl-llama-7b) |
| PandaGPT         | [https://github.com/yxuansu/PandaGPT](https://github.com/yxuansu/PandaGPT) |
| IB-LLM           | [https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM) |
| LA-V2            | [https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) |
| mmGPT            | [https://github.com/open-mmlab/Multimodal-GPT](https://github.com/open-mmlab/Multimodal-GPT) |
| Shikra           | GitHub:[https://github.com/shikras/shikra](https://github.com/shikras/shikra)<br>Model:[https://huggingface.co/shikras/shikra-7b-delta-v1](https://huggingface.co/shikras/shikra-7b-delta-v1) |
| Lynx             | [https://github.com/bytedance/lynx-llm](https://github.com/bytedance/lynx-llm) |
| Cheetor-Vicuna   | GitHub:[https://github.com/DCDmllm/Cheetah](https://github.com/DCDmllm/Cheetah)<br>Model:[https://drive.google.com/file/d/1mBiMzyY468QWUix8CuCvuByVEs9yfYPu/view](https://drive.google.com/file/d/1mBiMzyY468QWUix8CuCvuByVEs9yfYPu/view) |
| Cheetor-LLaMA2   | GitHub:[https://github.com/DCDmllm/Cheetah](https://github.com/DCDmllm/Cheetah)<br>Model:[https://drive.google.com/file/d/1kzpbvcFdq1XxAGSPbqPMmsjwi-etJ5Yi/view](https://drive.google.com/file/d/1kzpbvcFdq1XxAGSPbqPMmsjwi-etJ5Yi/view) |
| BLIVA            | GitHub:[https://github.com/mlpc-ucsd/BLIVA](https://github.com/mlpc-ucsd/BLIVA)<br>Model:[https://huggingface.co/mlpc-lab/BLIVA_Vicuna](https://huggingface.co/mlpc-lab/BLIVA_Vicuna) |

Only when multiple models exist at the same time, both of the GitHub URL and the model URL are provided.

### Add Your Own Models
To add new models, you need to create the corresponding model interface for the unified evaluation. For a general new model interface, please refer to the interface template in `PATH_TO_REFORM-EVAL/models/interfaces/base_interface.py`. Here we provide a step-by-step guide for the convenience of your implementation (taking Lynx as an example).

#### Step 1: Configure the Code Path
Add the Lynx project as a submodule to `PATH_TO_REFORM-EVAL/models/interfaces/`:
```bash
cd models/interfaces
git submodule add https://github.com/bytedance/lynx-llm.git
```

#### Step 2: Model Loading
Refer to the code for loading the model in the original Lynx project.
```python
def main(args, config):
    print("### Evaluating", flush=True)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("config:", json.dumps(config), flush=True)
    print("output_path, ", args.output_path, flush=True)

    print("### Creating model", flush=True)
    from models.lynx import LynxBase
    model = LynxBase(config=config, freeze_vit=config['freeze_vit'], freeze_llm=config['freeze_llm'], load_bridge=False)
```

So, we can implement the `__init__` function for model loading in our interface:
```python
class Lynx_Interface(nn.Module):
    def __init__(self, model_config=None, device=None, half=False, inference_method='generation') -> None:
        super(Lynx_Interface, self).__init__()
        # setup the model device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # loading the model
        self.config = yaml.load(open(model_config, 'r'), Loader=yaml.Loader)
        self.model = LynxBase(config=self.config, freeze_vit=self.config['freeze_vit'], freeze_llm=self.config['freeze_llm'], load_bridge=False)
        
        # locate the model to half-precision and target device if needed
        self.prec_half = half
        if self.prec_half:
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        
        # setup the inference method
        self.inference_method = inference_method
```

#### Step 3: Implement the Inference Function
**Generation-based Black-Box Evaluation**
After that, find the generation-related code in the original Lynx project.
```python
@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()
    result = []

    for n, (idx, vision_input, input_ids, input_atts) in enumerate(data_loader):
        vision_input = vision_input.to(device, non_blocking=True)
        input_ids = input_ids.to(device)
        input_atts = input_atts.to(device)

        text_outputs = model.generate(
            vision_input=vision_input,
            input_ids=input_ids, input_atts=input_atts,
            use_nucleus_sampling=config.get('use_nucleus_sampling', False),
            apply_lemmatizer=config['apply_lemmatizer'],
            num_beams=config['num_beams'],
            min_length=config['min_length'],
            length_penalty=config.get('length_penalty', 1.0),
            no_repeat_ngram_size=config.get('no_repeat_ngram_size', -1),
            top_p=config.get('top_p', 0.9),
            top_k=config.get('top_k', 3),
            max_new_tokens=config.get('max_new_tokens', 64))

        for i, output in zip(idx, text_outputs):
            result.append({"index": i, "text_output": output.strip()})

    return result
```

Therefore, in `lynx_interface.py`, we can implement the generation inference function as:
```python
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
```

In this function, you have to use the internal vision processor to get the vision input (open and get the image), and the internal tokenizer to get the input_ids and input_atts. All of these codes can be directly found and implemented from the original project.
```python
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
```

**Likelihood-based Black-Box Evaluation**
To support the likelihood evaluation, we add the following function in our model file `PATH_TO_REFORM-EVAL/models/interfaces/lynx/models/lynx.py` to calculate the loss (neg-log likelihood) for each sequence.
```python
    def forward_likelihood(self, vision_input, input_ids, input_atts, labels, likelihood_reduction='sum'):
        text_embeds = self.embed_tokens(input_ids)

        if vision_input is not None:
            vision_embeds, vision_atts = self.get_vision_embeds(vision_input)
            v2t_feats, v2t_atts = self.bridge(vision_embeds=vision_embeds, vision_atts=vision_atts)

            inputs_embeds = torch.cat([v2t_feats, text_embeds], dim=1)
            attention_mask = torch.cat([v2t_atts, input_atts], dim=1)

        else:
            inputs_embeds = text_embeds
            attention_mask = input_atts

        outputs = self.LLM(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            reduction='none'
        )
        loss = outputs.loss.reshape(inputs_embeds.shape[0], -1)
        if likelihood_reduction == 'sum':
            loss = loss.sum(1)
        elif likelihood_reduction == 'mean':
            valid_num_targets = (loss > 0).sum(1)
            loss = loss.sum(1) / valid_num_targets
        elif likelihood_reduction == 'none':
            loss = loss
        else:
            raise ValueError
        return loss
```

Hence, in `lynx_interface.py`, we can use `self.model.forward_likelihood` at the `raw_predict` function.
```python
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
        # loading the image-text pair
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
        
        # calculate the loss (neg-log likelihood) for each candidate
        with torch.inference_mode():
            outputs = self.model.forward_likelihood(
                vision_input=vision_input.repeat_interleave(num_cand, dim=0),
                input_ids=full_input_ids,
                input_atts=attention_mask,
                labels=targets,
                likelihood_reduction=likelihood_reduction
            )
        neg_likelihood = outputs
        # select the one with the highest likelihood / lowest loss
        output_class_ranks = torch.argsort(neg_likelihood, dim=-1)[0].item()

        return output_class_ranks
```

#### Step 4: Implement the Preprocessor
Preprocessors are used to formulate the structural information in order to get the correct form of dialogue. Our preprocessor is in `PATH_TO_REFORM-EVAL/utils/preprocessors.py`.
```python
class ConvSingleChoiceProcessor(object):
    def __init__(self, sep, sep2=None, roles=['Question', 'Answer'], system_msg=None, first_query_fn=None, \
                 init_conv=None, sep_style='two', alphabet_choice=None, infer_method='generation', response_prefix=None):
        """
        Preprocessors to convert input information into a dialogue string
        
        Args:
            sep (str):
                The text separator-1.
            sep2 (str):
                The text separator-2.
            roles (list[str]):
                Role names of the dialogue, roles[0] is the role of users while 
                roles[1] is the name of assistants.
            system_msg (str, **optional**):
                The system message that appears at the beginning.
            first_query_fn (function, **optional**):
                The function to process the first query, mainly for adding <img> marks.
            init_conv (list[list[str]]):
                The initial conversation. Each element is a list[str, str] where the first
                is the role name and the second is the message. 
            sep_style (str):
                The dialogue style. 
            alphabet_choice (str, **optional**):
                The option mark used for multiple-choice questions, defaults to "random"
            infer_method (str, "optional"):
                The inference method ("generation" or "likelihood")
            response_prefix (str, **optional**):
                The prefix text for the response of LVLM assistants, we use "The answer is"
                to help with multiple-choice questions.
                
        Returns:
            output (str):
                The constructed dialogue text.
        """
```

Here is an example of the `\n`-separated preprocessor:
```python
proc = ConvSingleChoiceProcessor('\n', roles=['User', 'Bot'], first_query_fn=lambda x: "<image> "+x,
                                sep_style='one', infer_method=model_args['inference_method'], response_prefix='The answer is',
                                system_message="A chat between a curious human and an artificial intelligence assistant. The 
                                assistant gives helpful, detailed, and polite answers to the human's questions.")
```

The input sample is a json-style dict:
```
inputs = {'sample_id': '287626_3',
 'round_id': 3,
 'image': 'IMAGE_PATH.jpg',
 'question': 'Is there a cat in the image?',
 'answer': '2',
 'answer_options': ['yes', 'no', 'maybe'],
 'history': [{'from': 'human', 'value': 'Can you see the image? Options: (A) yes; (B) no'},
             {'from': 'assistant', 'value': 'The answer is (A) yes'}]
}
```

Therefore, the final content will be:
```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
User: <image> Can you see the image? Options: (A) yes; (B) no.\n
Bot: The answer is (A) yes\n
User: Is there a cat in the image? Options: (A) yes; (B) no; (C) maybe.\n
Bot:The answer is
```

For other supported sep_style, please refer to `PATH_TO_REFORM-EVAL/utils/preprocessors.py`.
`init_conv` can also be used to add `<image>` marks, if it is `init_conv=[['User', "<image>"]]`, this means that a new conversation will be started.

```
User: <image>
User: ......
Bot: ......
```

#### Step 5: Add Model Loader
Implement the model loading function in `PATH_TO_REFORM-EVAL/models/interfaces/lynx_interface.py`.
```python
def get_lynx(model_config=None):
    model_args = {}
    # map the general input arguments to the model-specific arguments
    if model_config is not None:
        valid_args = ['model_name', 'device', 'half', 'inference_method']
        target_args = ['model_config', 'device', 'half', 'inference_method']
        for i, arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    # configure the dialogue preprocessor
    proc = ConvSingleChoiceProcessor('\n', roles=['User', 'Bot'], \
                                     sep_style='one', infer_method=model_args['inference_method'], response_prefix='The answer is')
    return Lynx_Interface(**model_args), proc
```

Additionally, you should add the following codes in  `PATH_TO_REFORM-EVAL/models/__init__.py`.
```python
    elif model_name == 'lynx':
        from .interfaces.lynx_interface import get_lynx
        return get_lynx(model_config)
```

#### Done!
Finally, you can use the following model arguments in the main entrance to evaluate your model!
```bash
--model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml
```

If you have trouble incorporating new models into our framework, please let us know through GitHub issues or emails.

### Preprocessors
We give a brief introduction to preprocessors in order to get the correct form of dialogue. Our preprocessor is in `PATH_TO_REFORM-EVAL/utils/preprocessors.py`.
```python
class SingleChoiceProcessor(object):
    def __init__(self, sep, sep2=None, roles=['Question', 'Answer'], alphabet_choice=None, infer_method='generation'):
        self.sep = sep
        self.sep2 = sep2
        self.roles = roles
        if alphabet_choice is not None:
            if alphabet_choice == 'number':
                self.ab = alphabet[2]
            elif alphabet_choice == 'lower':
                self.ab = alphabet[0]
            elif alphabet_choice == 'upper':
                self.ab = alphabet[1]
            else:
                raise ValueError
        else:
            self.ab = alphabet
        self.infer_method = infer_method
```

There are different sep styles in these models, and we mainly classify them into three categories, which are `sep_style='one', 'two' or others`, respectively.
If you set the `sep_style='one'`, parameters should be set similar to this:
```python
proc = ConvSingleChoiceProcessor('\n', roles=['User', 'Bot'], \
                                sep_style='one', infer_method=model_args['inference_method'], response_prefix='The answer is')
```

`\n` represents whether it is a `User` or a `Bot`, the conversation will end with `\n`. 
`roles` represents whether this round of dialogue is User or Bot, and a semicolon `:` will be added after the specific role. 
`response_prefix` is that the final answer will first add the text `The answer is`.
Therefore, the final content will be:
```
User: ......
Bot: ......
User: ......
Bot:The answer is ......
```

If `sep_style='two'`, `sep=' '` and `sep2='\n'` represents when it is a User, the conversation will end with `' '` and when it is a Bot, the conversation will end with `\n`.
`system_msg` denotes that a paragraph needed to start the entire conversation, something like `Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.`
`first_query_fn` is the content only appended after the first `Human:`, which is usually used to add `<img>` or `</img>`, etc.
`init_conv` is similar to `first_query_fn`, but for example, if it is `init_conv=[['Human', "<image>"]]`, this means that a new conversation will be started.

```
Human: <image>
Human: ......
AI: ......
```

`instruct` content occasionally appears, but it will be automatically added to the final text in the preprocess function. Besides, you can also simply add your own sep styles if you need.