source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh
nvidia-smi
# machine=224
# machine=226
machine=235

MASTER_PORT='29500'
CUDA_DEVICE='0,1,2,3'
NPROC_PER_NODE='4'

if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench
elif [ "$machine" -eq 235 ]; then
    conda activate 235-blip2
fi
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name MSCOCO --dataset_config datasets/configs/Caption_MSCOCO_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/blip2_t5_instruct/without_half_eval_v1 \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name TextCaps --dataset_config datasets/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_t5_instruct/without_half_eval_v1 \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name NoCaps --dataset_config datasets/configs/Caption_NoCaps_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/blip2_t5_instruct/without_half_eval_v1 \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --dataset_name Flickr30K --dataset_config datasets/configs/Caption_Flickr30K_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/blip2_t5_instruct/without_half_eval_v1 \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16


