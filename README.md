<div align="center">
  <h1 style="display: inline-block; font-size: 48px;">ReForm-Eval</h1>
</div>

<p align="center">
<img src="https://avatars.githubusercontent.com/u/100903507?s=200&v=4" alt="Fudan Disc Logo" style="display: inline-block; vertical-align: middle; height: 48px;">
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Version-v1.0-green" />
    <img src="https://img.shields.io/github/stars/FudanDISC/ReForm-Eval?style=social" />
    <img src="https://img.shields.io/github/downloads/FudanDISC/ReForm-Eval/total?style=social" />
    <img src="https://img.shields.io/github/views/FudanDISC/ReForm-Eval?style=social" />
</p>

<div align="center">
  <h2 >ReForm-Eval: EVALUATING LARGE VISION LANGUAGE MODELS VIA UNIFIED RE-FORMULATION OF TASK-ORIENTED BENCHMARKS</h2>
</div>

<!-- <span style='font-size: 24px; font-weight: bold;'><p align='center'>ReForm-Eval: EVALUATING LARGE VISION LANGUAGE MODELS VIA UNIFIED RE-FORMULATION OF TASK-ORIENTED BENCHMARKS</p></span> -->

<p align="center"><strong>Zejun Li<sup>1</sup><sup>†</sup> , Ye Wang<sup>1</sup><sup>†</sup> , Mengfei Du<sup>1</sup><sup>†</sup> , Qingwen Liu<sup>1</sup><sup>†</sup> , Binhao Wu<sup>1</sup><sup>†</sup> , Jiwen Zhang<sup>1</sup><sup>†</sup> , Chengxing Zhou<sup>2</sup> , Zhihao Fan<sup>3</sup> , Jie Fu<sup>4</sup> , Jingjing Chen<sup>1</sup> , Xuanjing Huang<sup>1</sup> , Zhongyu Wei<sup>1</sup><sup>*</sup>.
 </strong></p>
<p align="center"><sup>1</sup>Fudan University      <sup>2</sup>Northeastern University      <sup>3</sup>Alibaba Group        <sup>4</sup>Hong Kong University of Science and Technology</p> 
<p align="center"><sup>†</sup>Equal Contribution        <sup>*</sup>Corresponding author</p> 

---

<p align="center">
  <a href="https://arxiv.org/abs/2310.02569v1">ReForm-Eval Paper</a> | <a href="https://huggingface.co/datasets/Aweminus/ReForm-Eval/tree/main">ReForm-Eval Dataset</a>
</p>

<!-- <div style="border-left: 2px solid #999; padding-left: 10px; margin-left: 10px; color: #666; font-size: 90%;">
Recent years have witnessed remarkable progress in the development of large vision-language models (LVLMs). Benefiting from the strong language backbones and efficient cross-modal alignment strategies, LVLMs exhibit surprising capabilities to perceive visual signals and perform visually grounded reasoning. However, the capabilities of LVLMs have not been comprehensively and quantitatively evaluated. Most existing multi-modal benchmarks require task-oriented input-output formats, posing great challenges to automatically assess the freeform text output of LVLMs. To effectively leverage the annotations available in existing benchmarks and reduce the manual effort required for constructing new benchmarks, we propose to re-formulate existing benchmarks into unified LVLM compatible formats. Through systematic data collection and reformulation, we present the ReForm-Eval benchmark, offering substantial data for evaluating various capabilities of LVLMs. Based on ReForm-Eval, we conduct extensive experiments, thoroughly analyze the strengths and weaknesses of existing LVLMs, and identify the underlying factors. Our benchmark and evaluation framework will be open-sourced as a cornerstone for advancing the development of LVLMs.
</div> -->

>Recent years have witnessed remarkable progress in the development of large vision-language models (LVLMs). Benefiting from the strong language backbones and efficient cross-modal alignment strategies, LVLMs exhibit surprising capabilities to perceive visual signals and perform visually grounded reasoning. However, the capabilities of LVLMs have not been comprehensively and quantitatively evaluated. Most existing multi-modal benchmarks require task-oriented input-output formats, posing great challenges to automatically assess the freeform text output of LVLMs. To effectively leverage the annotations available in existing benchmarks and reduce the manual effort required for constructing new benchmarks, we propose to re-formulate existing benchmarks into unified LVLM compatible formats. Through systematic data collection and reformulation, we present the ReForm-Eval benchmark, offering substantial data for evaluating various capabilities of LVLMs. Based on ReForm-Eval, we conduct extensive experiments, thoroughly analyze the strengths and weaknesses of existing LVLMs, and identify the underlying factors. Our benchmark and evaluation framework will be open-sourced as a cornerstone for advancing the development of LVLMs.

We explore ways of re-formulating existing benchmarks into unified formats that are compatible with LVLMs. 
<!-- Referring to the following figure, we adapt the evaluation process to the unified form shown in the lower part.  -->

<p align="center"><img src="./short.png" /></p>
<!-- Illustration of the unified re-formulation of existing benchmarks into multiple-choice problems. The text within square brackets indicates the evaluation methods, with red and green denoting incorrect and correct judgment, respectively. “EM” is short for exact match. -->
<!-- <h3 align="center"><img src="./intro-flat.pdf" /></h3> -->

<span style="font-size:larger;">**Existing LVLMs Evaluation:**</span>

- **No Quantification**: The capabilities of existing LVLMs are mainly demonstrated only by qualitative examples.
- **Task-Oriented**: Most existing multi-modal benchmarks can not be directly utilized to evaluate LVLMs since they are designed for specific tasks and rely on structured input-output formats for evaluation, even need to be fine-tuned or learn task-specific parameters.
- **Limited Samples**: Limited manual annotation such as around 100 samples per dimension in **MME** and **MMBench** could potentially introduce evaluation bias into the results.

<span style="font-size:larger;">**Based on the re-formulation framework, we present our unified multi-modal benchmark, ReForm-Eval:**</span>
- **Larger Data Scale**: ReForm-Eval provides a dataset scale almost **100 times larger** than existing benchmarks, allowing models to be comprehensively evaluated across various dimensions.

- **Without Manual Annotation**: ReForm-Eval leverages publicly open resources, reducing annotation costs while providing a larger-scale dataset.

- **Universal Evaluation**: Unlike **LVLM-ehub** which requires designing complex and dataset-specific evaluation strategies, ReForm-Eval offers greater scalability and a more universally applicable and efficient evaluation approach.

- **Comprehensive Evaluation**: We re-formulate **61 benchmark datasets** based on existing data resources, the evaluation dimensions range from basic visual perception to high-level visual reasoning and dialog.

- **Unified Re-formulation**: Multi-modal benchmark datasets are re-formulated as **multiple-choice problems** or specialized **text generation problems**. Additionally, **generation-based black-box** and **likelihood-based white-box approaches** are implemented for evaluation.

The unified formulation enables universal and comprehensive evaluation. For each formulation, we design a consistent and reliable evaluation method. As mentioned in ([Fu et al., 2023](https://arxiv.org/abs/2306.13394)), current LVLMs may struggle to follow multiple-choice instructions, we propose both black-box and white-box approaches to assist: 

(1) Guiding LVLMs to output in desired formats through in-context learning; 

(2) Directly calculating the generation probability for options and selecting the one with the highest value. 

Considering the sensitivity of LVLMs to the input prompts ([Zeng et al., 2023](https://arxiv.org/abs/2307.02469)), we additionally design an instability-aware evaluation strategy and introduce a metric to characterize such instability. 

**🔧🔧🔧ReForm-Eval serves as a reliable tool for quantitative analysis of LVLMs, aiding in the research and development of LVLMs.🔧🔧🔧**

## 📣 Update
**If you have any questions, please send us an email or leave a github issue!**
**`Email: yewang22@m.fudan.edu.cn`**

- **[2023-10]** We released a version of the original paper containing 16 models and 61 reformulated datasets!

## 📖 Contents
- [Model Performance](#🦾-model-performance)
- [Prepare Dataset](build/prepare_dataset.md#📥-prepare-dataset)
  - [Load Dataset from Hugging Face (Recommended)](build/prepare_dataset.md#load-dataset-from-hugging-face-recommended)
  - [Manually Download](build/prepare_dataset.md#manually-download)
- [Prepare Models](models/prepare_models.md#🤖-prepare-models)
  - [Set Up Existing Models](models/prepare_models.md#set-up-existing-models)
  - [Add Your Own Models](models/prepare_models.md#add-your-own-models)
  - [Preprocessors](models/prepare_models.md#preprocessors)
- [Getting Start](#🔥-getting-start)
  - [Demo](#demo)
  - [Parameters](#parameters)
  - [Model Usage](#model-usage)
  - [Data Usage](#data-usage)
- [Evaluation](#🚀-evaluation)
  - [Direct Evaluation](#direct-evaluation)
  - [Evaluation Using Our Benchmark](#evaluation-using-our-benchmark) 
- [Citation](#🖋-citation)
- [Related Projects](#🔏-related-projects)

## 🦾 Model Performance
We list the average ranking and score of the model under Generation Evaluation and Likelihood Evaluation in the table below. New models will be added soon.

| Model          | Gen-Avg-Rank | Gen-Avg-Score | Like-Avg-Rank | Like-Avg     |
|----------------|--------------|---------------|---------------|--------------|
| **BLIP-2**     | *2.3*          | **62.94**         | 4.3           | 62.92        |
| **InstructBLIP_F** | **2.0**      | *60.77*         | 4.0           | 63.48        |
| **InstructBLIP_V** | 4.4      | 52.20         | 3.0           | *64.37*        |
| **LLaVA_V**    | 11.1         | 34.24         | 8.7           | 55.49        |
| **LLaVA_L2**   | 5.9          | 45.78         | 11.2          | 52.97        |
| **MiniGPT4**   | 7.3          | 43.12         | 7.8           | 56.15        |
| **mPLUG-Owl**  | 10.6         | 37.95         | 10.3          | 53.69        |
| **PandaGPT**   | 13.9         | 26.84         | 15.8          | 41.80        |
| **IB-LLM** | 13.0       | 30.24         | 14.5          | 47.58        |
| **LA-V2**      | 12.5         | 32.60         | 12.2          | 50.00        |
| **mmGPT**      | 14.4         | 29.38         | 12.8          | 50.92        |
| **Shikra**     | 11.0         | 36.14         | 7.0           | 58.40        |
| **Lynx**       | 5.0          | 50.00         | *2.8*           | 63.93        |
| **Cheetor_V**  | 6.8          | 44.74         | 8.2           | 56.73        |
| **Cheetor_L2** | 7.9          | 41.75         | 10.7          | 52.43        |
| **BLIVA**      | 7.9          | 42.40         | **2.7**           | **64.92**        |

<!-- This table presents the comprehensive performance of each model across dimensions, from which several insights can be gleaned. 
(1) BLIP-2 and InstructBLIP continue to hold the top-2 positions in most dimensions, but in some individual dimensions, Lynx, BLIVA, and Shikra also take the lead. 
(2) It’s worth noting that the effectiveness of models like BLIVA and Lynx only becomes apparent when using likelihood evaluation. We suspect this is attributed to the instruction-following ability of models. 
(3) Compared to models based on CLIP visual encoders, PandaGPT and IB-LLM, which are based on the ImageBind encoder, exhibit relatively poorer performance in image-text tasks. Meanwhile, most top-performing models utilize Vicuna and FlanT5 as the backbone. 
(4) Apart from the architecture, a common characteristic among BLIP-2, InstructBLIP, Lynx, and BLIVA is the use of relatively high-quality data during pre-training.  -->

## 🔥 Getting Start
**Before performing the evaluation, please refer to [Prepare Dataset](build/prepare_dataset.md#prepare-dataset) and [Prepare Models](models/prepare_models.md#prepare-models).** Our benchmark supports multi-GPU evaluation. If the half evaluation is set, the evaluation can be run on a single machine within CUDA memory of 24G on a single card for 7B models under limited equipment conditions.

### Demo
We provide one example of running the benchmark test, using Lynx model for VisDial Evaluation.
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name VisDial --output_dir output/lynx/VisDial/test_generation/ \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation  --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config datasets/configs/VisDial_val_v1.2.yaml \
```

The num of `--nproc_per_node` must be equal to the num of `CUDA_VISIBLE_DEVICES`. 
`--output_dir` is the path of output result. 
`--formulation` must be `Generation` or `SingleChoice`. 
`--infer_method` must be `generation` or `likelihood`. 
If you infer in generation mode, you should use `--in_context_sample` to assist models to generate option marks for most questions. 
`--dataset_config` is the path of the dataset config file.

### Parameters
```python
def main():
    parser = argparse.ArgumentParser()
    # model-related parameters
    parser.add_argument('--model', type=str, default=None, help='the model family name')
    parser.add_argument('--model_name', type=str, default=None, help='the model name to load')
    parser.add_argument('--model_type', type=str, default=None, help='the model type to set')
    # dataset-related parameters
    parser.add_argument('--dataset_name', type=str, default=None, help='the dataset name to evaluate on')
    parser.add_argument('--formulation', type=str, default=None, help='the problem formulation to perform, must be in ("Generation", "SingleChoice")')
    parser.add_argument('--dataset_config', type=str, default=None, help='the config file path, using the default path without explicit ')
    parser.add_argument('--dataset_duplication', type=int, default=1, help='duplicate the sample for evaluating the stability')
    parser.add_argument('--in_context_sample', action='store_true', help='whether to provide in-context-learning samples')
    parser.add_argument('--capitalize', action='store_true', help='whether to capitalize the qa')
    # 0805 add
    parser.add_argument('--yesno_instruct', action='store_true', help='whether add "please answer yes or no" to the full instruct')
    parser.add_argument('--answer_space_instruct', action='store_true', help='whether add answer space to the full instruct')
    # running parameters
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=1, help='the batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='workers in dataloader')
    parser.add_argument('--half_evaluation', action='store_true', help='whether to use half precision for evluation')
    # general evaluation setup
    parser.add_argument('--do_eval', action='store_true', help='whether to evluate the output.')
    parser.add_argument('--eval_stability', action='store_true', help='whether to evaluate the stability')
    # parameters for model generation
    parser.add_argument('--temperature', type=float, default=None, help='the temperature for generation')
    parser.add_argument('--max_new_tokens', type=int, default=None, help='max new tokens to generate')
    # parameters for likelihood measurement
    parser.add_argument('--likelihood_reduction', type=str, default=None, help='the reduction method for likelihood measurement')
    # parameters for SingleChoice problem
    parser.add_argument('--infer_method', type=str, default='generation', help='the inference method to use, must be in ["generation", "likelihood"]')
    parser.add_argument('--option_mark', type=str, default=None, help='the index mark for options in single-shoice questions, \
                        "number" for (1,2,3,4), "lower" for (a,b,c,d) while "upper" for (A,B,C,D)')
    # parameters for randomness control
    parser.add_argument('--random_instruct', action='store_true', help='whether to use random instructions')
    parser.add_argument('--shuffle_options', action='store_true', help='whether to shuffle options')
    # parameters for multi-round problem
    parser.add_argument('--options_in_history', action='store_true', help='whether to put options in history.')
    parser.add_argument('--online_multi_round', action='store_true', help='make online update to the history during dialog')
    parser.add_argument('--multi_round_eval', action='store_true', help='whether to evaluate multi-round performance')
    # output setup
    parser.add_argument('--output_dir', type=str, default='./output/', help='the path to save the output')
    # debug mode
    parser.add_argument('--dataset_debug', action='store_true', help='debug on the dataset setup')
    parser.add_argument('--dataset_subsample', type=int, default=None, help='only n sub-samples of the dataset')
    # core
    parser.add_argument('--core_eval', action='store_true', help='only eval on the core datasets')
    # hugging face
    parser.add_argument('--hf', action='store_true', help='whether to load the dataset directly from Hugging Face')
    args = parser.parse_args()
```

All parameters used are listed above and you can modify any parameter to customize your evaluation settings.

### Model Usage
For model-related parameters, we list required parameters of all 16 models. When running the evaluation, these commands must be applied for specific models.

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

In `./models/MiniGPT-4/eval_configs/minigpt4_eval.yaml`, you have to set:
```YAML
ckpt: '/path/to/prerained_minigpt4_7b.pth'
```

In `./models/interfaces/minigpt4/configs/models/minigpt4.yaml`, you need to set:
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

In `./models/interfaces/lynx/configs/LYNX.yaml`, you need to set:
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

In `./models/interfaces/Cheetah/cheetah/configs/models/cheetah_vicuna(llama2).yaml`, you need to set:
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

You need to modify the config file in `./models/BLIVA/bliva/configs/models/bliva_vicuna7b.yaml`:
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

### Data Usage

#### Coarse-Grained Perception
##### Flowers102
```bash
--dataset_name Flowers102 --formulation SingleChoice --dataset_config build/configs/ImageClassification_flowers102_val.yaml
```
##### CIFAR10
```bash
--dataset_name CIFAR10 --formulation SingleChoice --dataset_config build/configs/ImageClassification_cifar10_val.yaml
```
##### ImageNet-1K
```bash
--dataset_name ImageNet-1K --formulation SingleChoice --dataset_config build/configs/ImageClassification_imagenet1k_val.yaml
```
##### Pets37
```bash
--dataset_name Pets37 --formulation SingleChoice --dataset_config build/configs/ImageClassification_pets37_val.yaml
```
##### VizWiz-yesno
```bash
--dataset_name VizWiz --formulation SingleChoice --dataset_config build/configs/ImageQuality_vizwiz_yesNo_val.yaml
```
##### VizWiz-singleChoice
```bash
--dataset_name VizWiz --formulation SingleChoice --dataset_config build/configs/ImageQuality_vizwiz_singleChoice_val.yaml
```
##### TDIUC-Sport
```bash
--dataset_name VizWiz --formulation SingleChoice --dataset_config build/configs/ImageQuality_vizwiz_singleChoice_val.yaml
```
##### TDIUC-Scene
```bash
--dataset_name TDIUC --formulation SingleChoice --dataset_config build/configs/TDIUC_scene.yaml
```
##### MEDIC

#### Fine-Grained Perception
##### MSCOCO-MCI
##### MSCOCO-GOI
##### MSCOCO-MOS

##### TDIUC-Color
```bash
--dataset_name TDIUC --formulation SingleChoice --dataset_config build/configs/TDIUC_color.yaml
```
##### TDIUC-Utility
```bash
--dataset_name TDIUC --formulation SingleChoice --dataset_config build/configs/TDIUC_utility.yaml
```
##### TDIUC-Position
```bash
--dataset_name TDIUC --formulation SingleChoice --dataset_config build/configs/TDIUC_position.yaml
```
##### TDIUC-Detection
```bash
--dataset_name TDIUC --formulation SingleChoice --dataset_config build/configs/TDIUC_detection.yaml
```
##### TDIUC-Counting
```bash
--dataset_name TDIUC --formulation SingleChoice --dataset_config build/configs/TDIUC_counting.yaml
```
##### RefCOCO
##### MSCOCO-OC
```bash
--dataset_name MSCOCO --formulation SingleChoice --dataset_config build/configs/ObjectCounting_mscoco_val.yaml
```

#### Visually Grounded Reasoning

##### VQA v2
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_vqa_v2_val.yaml
```

##### GQA
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_gqa_val_v2.0.yaml
```

##### Whoops
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_whoops_val.yaml
```
##### OK-VQA
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_okvqa_val.yaml
```

##### ScienceQA
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_scienceqa_val_v2.0.yaml
```

##### VizWiz
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_vizwiz_val_v2.0.yaml
```


##### ViQuAE
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_viquae_val.yaml
```

##### K-ViQuAE
``` bash
--dataset_name KVQA --formulation SingleChoice --dataset_config build/configs/KVQA_viquae_val.yaml
```

##### A-OKVQA
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_aokvqa_val.yaml
```

##### A-OKVQRA
``` bash
--dataset_name VQRA --formulation SingleChoice --dataset_config build/configs/VQRA_aokvqa_val.yaml
```

##### A-OKVQAR
``` bash
--dataset_name VQAR --formulation SingleChoice --dataset_config build/configs/VQAR_aokvqa_val.yaml
```

##### ImageNetVC
``` bash
--dataset_name VQA --formulation SingleChoice --dataset_config build/configs/VQA_imagenetvc_val.yaml
```

#### Cross-Modal Inference

##### Winoground
``` bash
--dataset_name CaptionSelection --formulation SingleChoice --dataset_config build/configs/CaptionSelection_winoground_val.yaml
```

##### MOCHEG
``` bash
--dataset_name MCV  --formulation SingleChoice --dataset_config build/configs/MCV_mocheg_val.yaml
```

#### Visually Scene Recognition

##### TextVQA
``` bash
--dataset_name OCR --formulation OCROpenEnded --dataset_config build/configs/OCR_textvqa_val.yaml
```

##### DocVQA
``` bash
--dataset_name OCR --formulation OCROpenEnded --dataset_config build/configs/OCR_docvqa_val.yaml
```

##### OCR-VQA
``` bash
--dataset_name OCR --formulation OCROpenEnded --dataset_config build/configs/OCR_ocrvqa_val.yaml
```

## 🚀 Evaluation

### Data Loader

ReForm-Eval provides the direct data loader if you would like to perform evaluation without our framework. Here is an example:
```python
from build import load_reform_dataset

# example for loading VQA v2
dataset = load_reform_dataset(
    # dataset config, please check Data Usage for available arguments
    dataset_name='VQA',
    formulation='SingleChoice',
    dataset_config='PATH_TO_REFORM-EVAL/build/configs/VQA_vqa_v2_val.yaml',
    inference_method='generation', # inference method, generation / likeligood
    in_context_sample=True, # whether to include in-context-sample
    random_instruct=True, # whether to use different instructions for the same sample
    data_duplication=5, # number of multiple tests for the same sample
    shuffle_options=True, # whether to shuffle the options for the same sample
    load_from_hf:Optional=True # whether to load from huggingface
)
```
Notice that each sample of the loaded dataset will be a dict containing all information like: 
```
{
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x484>,
    'question': 'Is there a cat in the image?',
    'answer': 2,
    'answer_options': ['yes', 'no', 'maybe'],
    'instruct': 'Based on the image, answer the question with the provided options.',
}
```
You may need to process them into a string with the desired format. You may be intersted in the [Preprocessors](models/prepare_models.md#preprocessors) we used in ReForm-Eval to gather the information into a dialogue-like string as the input for you model. All valid datasets and corresponding arguments are in the [Data Usage](#data-usage).

### Direct Evaluation
The output json file is generated in your `--output_dir` path, and you can dircetly look up the corresponding json file for the final result. You can also run command by ipython in the terminal:
```python
import json
res = json.load(open('/path/to/___.json')) #load the output json file
res[0] #res[n], n can be any number within the generated results
```

### Evaluation Using Our Benchmark
Our benchmark provides accuracy and instability as metrics for each task, to quantify the model performance.

**Step 1:** Use existing model interface or create a new model interface based on ReForm-Eval framework refer to [Prepare Models](models/prepare_models.md#🤖-prepare-models).

**Step 2:** Create the conda env corresponding to the model and install the necessary packages.

**Step 3:** Switch to the corresponding conda env, run run_eval.py in the root path of this repository, and add necessary parameters.

**Step 4:** Check the inference progress and results in the terminal. The accuracy, the format hit rate and instability can also be viewed in `output_dir_path/log.txt`.



## 🖋 Citation
If ReForm-Eval has been beneficial to your research and work, please cite our work using the following format:
```latex
@misc{li2023reformeval,
      title={ReForm-Eval: Evaluating Large Vision Language Models via Unified Re-Formulation of Task-Oriented Benchmarks}, 
      author={Zejun Li and Ye Wang and Mengfei Du and Qingwen Liu and Binhao Wu and Jiwen Zhang and Chengxing Zhou and Zhihao Fan and Jie Fu and Jingjing Chen and Xuanjing Huang and Zhongyu Wei},
      year={2023},
      eprint={2310.02569},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## 🤝 Acknowledgements
We thank [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), [MMBench](https://github.com/open-compass/MMBench), [LVLM-eHub](http://lvlm-ehub.opengvlab.com/index.html) and other repositories that have made great contributions to multi-modal large model evaluation. In addition, we are also very grateful that many LVLMs can be open sourced and participate in our evaluation, enriching results of our benchmarks.


## 🔏 Related Projects
- [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
- [MMBench: Is Your Multi-modal Model an All-around Player?](https://github.com/open-compass/MMBench)
- [LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models](http://lvlm-ehub.opengvlab.com/index.html)