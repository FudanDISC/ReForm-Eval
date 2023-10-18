source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh

nvidia-smi
machine=224
# machine=226
# machine=235

MASTER_PORT='29500'
CUDA_DEVICE='0,1,2,3'
NPROC_PER_NODE='4'
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

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5  --model_type pretrain_flant5xl \
    --in_context_sample --random_instruct --shuffle_options --infer_method $INFER_METHOD \
    --dataset_name MSCOCO --dataset_config build/configs/ImageTextSelection_val.yaml \
    --output_dir output/test_20231017/bhwu-test/its/blip2_t5/eval_ll \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5  --model_type pretrain_flant5xl \
    --in_context_sample --random_instruct --shuffle_options --infer_method $INFER_METHOD \
    --dataset_name WikiHow --dataset_config build/configs/TemporalOrdering_val.yaml \
    --output_dir output/test_20231017/bhwu-test/wits/blip2_t5/eval_ll \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5  --model_type pretrain_flant5xl \
    --in_context_sample --random_instruct --shuffle_options --infer_method $INFER_METHOD \
    --dataset_name SNLI-VE --dataset_config build/configs/VisualEntailment_val.yaml \
    --output_dir output/test_20231017/bhwu-test/ve/blip2_t5/eval_ll \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5  --model_type pretrain_flant5xl \
    --in_context_sample --random_instruct --shuffle_options --infer_method $INFER_METHOD \
    --dataset_name MEDIC --dataset_config build/configs/DisasterType_val.yaml \
    --output_dir output/test_20231017/bhwu-test/dts/blip2_t5/eval_ll \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5  --model_type pretrain_flant5xl \
    --in_context_sample --random_instruct --shuffle_options --infer_method $INFER_METHOD \
    --dataset_name RefCOCO --dataset_config build/configs/ReferringExpression_val.yaml \
    --output_dir output/test_20231017/bhwu-test/res/blip2_t5/eval_ll \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

# caption
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --dataset_name MSCOCO --dataset_config build/configs/Caption_MSCOCO_val.yaml \
    --output_dir output/bhwu_output/caption/mscoco/blip2_t5/eval \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 12 --offline_hf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --dataset_name TextCaps --dataset_config build/configs/Caption_TextCaps_val.yaml \
    --output_dir output/bhwu_output/caption/textcaps/blip2_t5/eval \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 15 --offline_hf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --dataset_name NoCaps --dataset_config build/configs/Caption_NoCaps_val.yaml \
    --output_dir output/bhwu_output/caption/nocaps/blip2_t5/eval \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 14 --offline_hf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT run_eval.py \
    --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
    --dataset_name Flickr30K --dataset_config build/configs/Caption_Flickr30K_val.yaml \
    --output_dir output/bhwu_output/caption/flickr30k/blip2_t5/eval \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation Generation \
    --dataset_duplication 1 --max_new_tokens 16 --offline_hf

