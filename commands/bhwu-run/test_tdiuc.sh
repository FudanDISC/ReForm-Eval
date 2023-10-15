source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh
# total 48 group
nvidia-smi
dm="TDIUC"

# imagebind_LLM----------------------------------------------------------------------------------------
# conda activate 235-imagebind_LLM

# # generation
# INFER_METHOD="generation"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/imagebindLLM/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_color.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/imagebindLLM/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_counting.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/imagebindLLM/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_detection.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/imagebindLLM/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_position.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/imagebindLLM/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_scene.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/imagebindLLM/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC.yaml \


# # likelihood
# INFER_METHOD="likelihood"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/imagebindLLM/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_color.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/imagebindLLM/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_counting.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/imagebindLLM/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_detection.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/imagebindLLM/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_position.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/imagebindLLM/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_scene.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/imagebindLLM/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC.yaml \

# # pandagpt-----------------------------------------------------------------------------------------------
# conda activate 235-pandagpt
# # generation
# INFER_METHOD="generation"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/pandagpt/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_color.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/pandagpt/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_counting.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/pandagpt/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_detection.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/pandagpt/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_position.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/pandagpt/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC_scene.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/pandagpt/eval_gen_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
#     --dataset_config datasets/configs/TDIUC.yaml \


# # likelihood
# INFER_METHOD="likelihood"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/pandagpt/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_color.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/pandagpt/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_counting.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/pandagpt/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_detection.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/pandagpt/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_position.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/pandagpt/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC_scene.yaml \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
#     --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
#     --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/pandagpt/eval_ll_v1 \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
#     --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
#     --dataset_config datasets/configs/TDIUC.yaml \

# mmgpt---------------------------------------------------------------------------------------------------

conda activate 235-mmgpt
# generation
INFER_METHOD="generation"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/mmgpt/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_color.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/mmgpt/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_counting.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/mmgpt/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_detection.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/mmgpt/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_position.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/mmgpt/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_scene.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/mmgpt/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC.yaml \


# likelihood
INFER_METHOD="likelihood"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/mmgpt/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_color.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/mmgpt/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_counting.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/mmgpt/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_detection.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/mmgpt/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_position.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/mmgpt/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_scene.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model mmgpt --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/mmgpt/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC.yaml \


shikra--------------------------------------------------------------------------------------------------

conda activate 235-shikra
# generation
INFER_METHOD="generation"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/shikra/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_color.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/shikra/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_counting.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/shikra/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_detection.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/shikra/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_position.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/shikra/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC_scene.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/shikra/eval_gen_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper  \
    --dataset_config datasets/configs/TDIUC.yaml \


# likelihood
INFER_METHOD="likelihood"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_color/shikra/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_color.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_counting/shikra/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_counting.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_detection/shikra/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_detection.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_position/shikra/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_position.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC_scene/shikra/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC_scene.yaml \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name $dm --output_dir output/bhwu_output/zcx/TDIUC/shikra/eval_ll_v1 \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method $INFER_METHOD --do_eval --half_evaluation --option_mark upper \
    --dataset_config datasets/configs/TDIUC.yaml \









