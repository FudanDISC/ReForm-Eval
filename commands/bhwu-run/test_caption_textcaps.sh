# TextCaps Caption
source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh
export PYTHONWARNINGS="ignore"
export CURL_CA_BUNDLE=""
export PYTHONHTTPSVERIFY=0
nvidia-smi
# machine=224
# machine=226
machine=235

MASTER_PORT='29501'
CUDA_DEVICE='4,5,6,7'
NPROC_PER_NODE='4'
# upper 1/4 tokens 15
# max tokens 36
MAX_NEW_TOKENS='30'

# overall

if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench
elif [ "$machine" -eq 235 ]; then
    conda activate 235-blip2
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_t5/default_length \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_vicuna_instruct/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_t5_instruct/default_length \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/minigpt4/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench-llava
elif [ "$machine" -eq 235 ]; then
    conda activate 235-llava
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/llava/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/llava2/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate mplugowl
elif [ "$machine" -eq 235 ]; then
    conda activate 235-mplugowl
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/mplugowl/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate imagebind_LLM
elif [ "$machine" -eq 235 ]; then
    conda activate 235-imagebind_LLM
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/imagebindLLM/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llama_adapter_v2
elif [ "$machine" -eq 235 ]; then
    conda activate 235-llama_adapter_v2
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/llama_adapterv2/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench-mmgpt
elif [ "$machine" -eq 235 ]; then
    conda activate 235-mmgpt
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/mmgpt/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate pandagpt
elif [ "$machine" -eq 235 ]; then
    conda activate 235-pandagpt
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/pandagpt/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate shikra
elif [ "$machine" -eq 235 ]; then
    conda activate 235-shikra
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/shikra/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate cheetah
elif [ "$machine" -eq 235 ]; then
    conda activate 235-cheetah
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/cheetor_vicuna/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/cheetor_llama2/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate bliva
elif [ "$machine" -eq 235 ]; then
    conda activate 235-bliva
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model bliva --model_name bliva_vicuna \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/bliva/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS


if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate lynx
elif [ "$machine" -eq 235 ]; then
    conda activate 235-lynx
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/lynx/default_length \
    --infer_method generation --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens $MAX_NEW_TOKENS

unset PYTHONHTTPSVERIFY
