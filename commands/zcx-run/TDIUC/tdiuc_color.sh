# 修改yaml中数据即可

source /root/anaconda3/etc/profile.d/conda.sh

dm="TDIUC"
dc="datasets/configs/TDIUC_color.yaml"
######################################  BLIP2
conda activate llava

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir test \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc\

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
#     --dataset_name $dm --output_dir output/blip2/TDIUC_color/flant5xl/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc\

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
#     --dataset_name $dm --output_dir output/blip2/TDIUC_Color/instruct_flant5xl/likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
#     --dataset_name $dm --output_dir output/blip2/TDIUC_Color/instruct_flant5xl/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
#     --dataset_name $dm --output_dir output/blip2/TDIUC_Color/instruct_vicuna7b/likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
#     --dataset_name $dm --output_dir output/blip2/TDIUC_Color/instruct_vicuna7b/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# ####################################  LLaVA

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
#     --dataset_name $dm --output_dir output/llava/TDIUC_Color/likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
#     --dataset_name $dm --output_dir output/llava/TDIUC_Color/generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \