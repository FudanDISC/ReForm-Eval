source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh

nvidia-smi
# machine=224
# machine=226
machine=235

MASTER_PORT='29500'
CUDA_DEVICE='5,7'
NPROC_PER_NODE='2'
INFER_METHOD='likelihood'

# only test under BLIP2-T5
if [ "$machine" -eq 224 ] || [ "$machine" -eq 226 ]; then
    conda activate llm-v-bench
elif [ "$machine" -eq 235 ]; then
    conda activate 235-blip2
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5  --model_type pretrain_flant5xl \
    --in_context_sample --random_instruct --shuffle_options --infer_method $INFER_METHOD \
    --dataset_name MSCOCO --dataset_config build/configs/ImageTextMatching_val.yaml \
    --output_dir output/test_20231017/bhwu-test/itm/blip2_t5/eval_ll \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

