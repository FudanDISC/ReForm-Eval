source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh

#################### ImageNet-1K
dm="ImageNet-1K"
dc="datasets/configs/ImageClassification_imagenet1k_val.yaml"

conda activate llava
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/mmgpt/imageNet1K/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/mmgpt/imageNet1K/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \


conda activate llava
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/shikra/imageNet1K/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/shikra/imageNet1K/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate llava
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/shikra/imageNet1K/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/shikra/imageNet1K/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate llava
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml \
    --dataset_name $dm --output_dir output/cheetor/imageNet1K/llama2/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc\

conda activate llava
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/imageNet1K/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/imageNet1K/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

#################### Pets37
conda activate llama_adapter_v2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name $dm --output_dir output/llama_adapter_v2/pets37/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name $dm --output_dir output/llama_adapter_v2/pets37/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate lynx
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/pets37/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/pets37/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate bliva
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/pets37/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/pets37/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

#################### VizWiz_singlechoice
dm="VizWiz"
dc="datasets/configs/ImageQuality_vizwiz_singleChoice_val.yaml"

conda activate blip2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir output/blip2/vizwiz/instruct_vicuna7b/singleChoice/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate imagebind_LLM
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name $dm --output_dir output/imagebindLLM/vizwiz/singleChoice/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name $dm --output_dir output/imagebindLLM/vizwiz/singleChoice/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate lynx
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/vizwiz/singleChoice/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/vizwiz/singleChoice/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate bliva
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/vizwiz/singleChoice/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/vizwiz/singleChoice/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

#################### VizWiz_yesNo
dm="VizWiz"
dc="datasets/configs/ImageQuality_vizwiz_yesNo_val.yaml"

conda activate blip2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir output/blip2/vizwiz/instruct_vicuna7b/yesNo/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate imagebind_LLM
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name $dm --output_dir output/imagebindLLM/vizwiz/yesNo/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name $dm --output_dir output/imagebindLLM/vizwiz/yesNo/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate lynx
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/vizwiz/yesNo/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/vizwiz/yesNo/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate bliva
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/vizwiz/yesNo/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/vizwiz/yesNo/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \


#################### MSCOCO
dm="MSCOCO"
dc="datasets/configs/ObjectCounting_mscoco_val.yaml"

conda activate mmgpt
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/mmgpt/mscoco/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate imagebind_LLM
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name $dm --output_dir output/imagebindLLM/mscoco/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name $dm --output_dir output/imagebindLLM/mscoco/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate lynx
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/mscoco/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir output/lynx/mscoco/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate bliva
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/mscoco/likelihood \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir output/bliva/mscoco/generation \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \



