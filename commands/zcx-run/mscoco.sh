source /root/anaconda3/etc/profile.d/conda.sh

dm="MSCOCO"
dc="datasets/configs/ObjectCounting_mscoco_val.yaml"

######################################  BLIP2
# conda activate blip2

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir test \
    --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --random_instruct --shuffle_options\

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
#     --dataset_name $dm --output_dir output/blip2/mscoco/flant5xl/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc\

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
#     --dataset_name $dm --output_dir output/blip2/mscoco/instruct_flant5xl/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
#     --dataset_name $dm --output_dir output/blip2/mscoco/instruct_flant5xl/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
#     --dataset_name $dm --output_dir output/blip2/mscoco/instruct_vicuna7b/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
#     --dataset_name $dm --output_dir output/blip2/mscoco/instruct_vicuna7b/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# # ####################################  LLaVA
# conda activate llava

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
#     --dataset_name $dm --output_dir output/llava/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
#     --dataset_name $dm --output_dir output/llava/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
#     --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
#     --dataset_name $dm --output_dir output/llava-llama-2/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
#     --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
#     --dataset_name $dm --output_dir output/llava-llama-2/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# # ###################################  MiniGPT4
# conda activate minigpt4

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
#     --dataset_name $dm --output_dir output/minigpt4/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \
    
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
#     --dataset_name $dm --output_dir output/minigpt4/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# # ###################################  mPLUG-owl
# conda activate mplugowl

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name $dm --output_dir output/mplug_owl/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name $dm --output_dir output/mplug_owl/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# ###################################  llama-adapter-v2
# conda activate llama_adapter_v2

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
#     --dataset_name $dm --output_dir output/llama_adapter_v2/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
#     --dataset_name $dm --output_dir output/llama_adapter_v2/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# ###################################  imageBind_LLM
# conda activate imagebind_LLM

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/imagebindLLM/mscoco/generation \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/imagebindLLM/mscoco/likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# ###################################  otter
# conda activate otter

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model otter  --model_name otter --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
#     --dataset_name $dm --output_dir output/otter/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model otter  --model_name otter --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
#     --dataset_name $dm --output_dir output/otter/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# ###################################  pandaGPT
# conda activate pandagpt

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt \
#     --dataset_name $dm --output_dir output/pandagpt/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt \
#     --dataset_name $dm --output_dir output/pandagpt/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# ####################################  lynx
# conda activate lynx

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir output/lynx/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir output/lynx/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# ####################################  cheetor
# conda activate cheetah

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
#     --dataset_name $dm --output_dir output/cheetor/mscoco/vicuna/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
#     --dataset_name $dm --output_dir output/cheetor/mscoco/vicuna/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml \
#     --dataset_name $dm --output_dir output/cheetor/mscoco/llama2/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc\

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml \
#     --dataset_name $dm --output_dir output/cheetor/mscoco/llama2/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc\

# ####################################  shikra
# conda activate shikra

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
#     --dataset_name $dm --output_dir output/shikra/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
#     --dataset_name $dm --output_dir output/shikra/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# # ####################################  bliva
# conda activate bliva

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir output/bliva/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir output/bliva/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# # ####################################  multimodal GPT
# conda activate mmgpt

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model mmgpt  --model_name Multimodal-GPT \
#     --dataset_name $dm --output_dir output/mmgpt/mscoco/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model mmgpt  --model_name Multimodal-GPT \
#     --dataset_name $dm --output_dir output/mmgpt/mscoco/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \
