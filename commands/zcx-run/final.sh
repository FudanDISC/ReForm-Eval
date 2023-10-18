dm="CIFAR10"
dc="build/configs/ImageClassification_cifar10_val.yaml"
output_dir="output/test_20231017/"

######################################  BLIP2
# conda activate llava

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir output/test_20231017/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper --shuffle_options --offline_hf \
    --dataset_config $dc \ 