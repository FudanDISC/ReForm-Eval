# visual entailment --master_port='29500'
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port='29500' run_eval.py \
#     --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
#     --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
#     --output_dir output/bhwu_output/caption/flickr30k/blip2_t5 \
#     --infer_method generation \
#     --per_gpu_eval_batch_size 1 --formulation Generation \
#     --dataset_duplication 1

source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh
# overall evaluation
# test mscoco
conda activate llm-v-bench
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2  --model_name blip2_t5 --model_type pretrain_flant5xl \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/blip2_t5/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/blip2_vicuna_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/blip2_t5_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/minigpt4/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12

conda activate llm-v-bench-llava
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/llava/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12

conda activate llm-v-bench-mmgpt
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/mmgpt/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12

conda activate mplugowl
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/mplugowl/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12



# test textcaps
conda activate llm-v-bench
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_t5/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_vicuna_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_t5_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/minigpt4/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15

conda activate llm-v-bench-llava
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/llava/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15

conda activate llm-v-bench-mmgpt
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/mmgpt/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15

conda activate mplugowl
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/mplugowl/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15



# test nocaps
conda activate llm-v-bench
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/blip2_t5/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/blip2_vicuna_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/blip2_t5_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/minigpt4/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14

conda activate llm-v-bench-llava
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/llava/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14

conda activate llm-v-bench-mmgpt
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/mmgpt/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14

conda activate mplugowl
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/mplugowl/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14




# test flickr30k
conda activate llm-v-bench
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/blip2_t5/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/blip2_vicuna_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/blip2_t5_instruct/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/minigpt4/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16

conda activate llm-v-bench-llava
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/llava/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16

conda activate llm-v-bench-mmgpt
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/mmgpt/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16

conda activate mplugowl
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/mplugowl/restrict_max_tokens \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16


