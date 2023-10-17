### Complete Model Usage
For model-related parameters, we list required parameters of all 16 models. When running the evaluation, these commands must be applied for specific models.

**Note: Some models require additional forward_likelihood function, please refer to `Likelihood-based Black-Box Evaluation` in [Add Your Own Models](models/prepare_models.md#add-your-own-models)**

We only list some models as examples. For the remaining existing models, please refer to the Complete Model Usage.

#### BLIP-2 + InstructBLIP
```bash
# BLIP-2 flant5
--model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl
# InstructBLIP flan-t5
--model blip2  --model_name blip2_t5_instruct  --model_type flant5xl
# InstructBLIP vicuna
--model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b


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