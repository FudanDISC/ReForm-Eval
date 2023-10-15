source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh
nvidia-smi

# machine=224
# machine=226
machine=235

MASTER_PORT='29501'
CUDA_DEVICE='4,5,6,7'
NPROC_PER_NODE='4'
INFER_METHOD='generation'

# overall evaluation


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench
elif [ "$machine" -eq 235 ]; then
    conda activate 235-blip2
fi
# blip2-flant5
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --output_dir output/bhwu_output/instability/blip2_t5/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --output_dir output/bhwu_output/instability/blip2_t5/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --output_dir output/bhwu_output/instability/blip2_t5/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random

# instructblip-vicuna
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --output_dir output/bhwu_output/instability/blip2_vicuna_instruct/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --output_dir output/bhwu_output/instability/blip2_vicuna_instruct/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --output_dir output/bhwu_output/instability/blip2_vicuna_instruct/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random

# instructblip-flant5
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --output_dir output/bhwu_output/instability/blip2_t5_instruct/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --output_dir output/bhwu_output/instability/blip2_t5_instruct/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --output_dir output/bhwu_output/instability/blip2_t5_instruct/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random

# minigpt4
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --output_dir output/bhwu_output/instability/minigpt4/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --output_dir output/bhwu_output/instability/minigpt4/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --output_dir output/bhwu_output/instability/minigpt4/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench-llava
elif [ "$machine" -eq 235 ]; then
    conda activate 235-llava
fi
# llava
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --output_dir output/bhwu_output/instability/llava/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --output_dir output/bhwu_output/instability/llava/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --output_dir output/bhwu_output/instability/llava/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random

# llava2
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
    --output_dir output/bhwu_output/instability/llava2/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
    --output_dir output/bhwu_output/instability/llava2/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
    --output_dir output/bhwu_output/instability/llava2/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate mplugowl
elif [ "$machine" -eq 235 ]; then
    conda activate 235-mplugowl
fi
# mplugowl
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --output_dir output/bhwu_output/instability/mplugowl/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --output_dir output/bhwu_output/instability/mplugowl/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --output_dir output/bhwu_output/instability/mplugowl/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate imagebind_LLM
elif [ "$machine" -eq 235 ]; then
    conda activate 235-imagebind_LLM
fi
# imagebindLLM
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM  --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --output_dir output/bhwu_output/instability/imagebindLLM/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM  --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --output_dir output/bhwu_output/instability/imagebindLLM/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM  --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --output_dir output/bhwu_output/instability/imagebindLLM/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llama_adapter_v2
elif [ "$machine" -eq 235 ]; then
    conda activate 235-llama_adapter_v2
fi
# llama_adapter_v2
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --output_dir output/bhwu_output/instability/llama_adapterv2/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --output_dir output/bhwu_output/instability/llama_adapterv2/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --output_dir output/bhwu_output/instability/llama_adapterv2/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench-mmgpt
elif [ "$machine" -eq 235 ]; then
    conda activate 235-mmgpt
fi
# mmgpt
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --output_dir output/bhwu_output/instability/mmgpt/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --output_dir output/bhwu_output/instability/mmgpt/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --output_dir output/bhwu_output/instability/mmgpt/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate pandagpt
elif [ "$machine" -eq 235 ]; then
    conda activate 235-pandagpt
fi
# pandagpt
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
    --output_dir output/bhwu_output/instability/mmgpt/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
    --output_dir output/bhwu_output/instability/mmgpt/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
    --output_dir output/bhwu_output/instability/mmgpt/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate shikra
elif [ "$machine" -eq 235 ]; then
    conda activate 235-shikra
fi
# shikra
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --output_dir output/bhwu_output/instability/shikra/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --output_dir output/bhwu_output/instability/shikra/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --output_dir output/bhwu_output/instability/shikra/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate cheetah
elif [ "$machine" -eq 235 ]; then
    conda activate 235-cheetah
fi
# cheetor-vicuna
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
    --output_dir output/bhwu_output/instability/cheetor_vicuna/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
    --output_dir output/bhwu_output/instability/cheetor_vicuna/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
    --output_dir output/bhwu_output/instability/cheetor_vicuna/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random

# cheetor-llama2
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml \
    --output_dir output/bhwu_output/instability/cheetor_llama2/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml \
    --output_dir output/bhwu_output/instability/cheetor_llama2/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml \
    --output_dir output/bhwu_output/instability/cheetor_llama2/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate bliva
elif [ "$machine" -eq 235 ]; then
    conda activate 235-bliva
fi
# bliva
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model bliva --model_name bliva_vicuna \
    --output_dir output/bhwu_output/instability/bliva/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model bliva --model_name bliva_vicuna \
    --output_dir output/bhwu_output/instability/bliva/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model bliva --model_name bliva_vicuna \
    --output_dir output/bhwu_output/instability/bliva/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate lynx
elif [ "$machine" -eq 235 ]; then
    conda activate 235-lynx
fi
# lynx
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --output_dir output/bhwu_output/instability/lynx/scienceqa_random/random_instruct/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --random_instruct

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --output_dir output/bhwu_output/instability/lynx/scienceqa_random/shuffle_options/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark upper --shuffle_options

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --output_dir output/bhwu_output/instability/lynx/scienceqa_random/random_mark/eval_gen \
    --dataset_name VQA_Random --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --infer_method $INFER_METHOD --in_context_sample --temperature 0.2 --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval --dataset_duplication 5 \
    --option_mark random











