source /root/anaconda3/etc/profile.d/conda.sh

#################### ImageNet1K
dm="ImageNet-1K"
dc="build/configs/ImageClassification_imagenet1k_val.yaml"
conda activate llava

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir tmp_rest_output/mmgpt/imageNet1K/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir tmp_rest_output/bliva/imageNet1K/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \


dm="TDIUC"
dc="build/configs/TDIUC_scene.yaml"
conda activate minigpt4

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir tmp_rest_output/mmgpt/new_TDIUC_Scene/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir tmp_rest_output/mmgpt/new_TDIUC_Scene/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \