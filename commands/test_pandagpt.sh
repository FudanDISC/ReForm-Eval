CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run_eval.py \
    --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
    --dataset_name VisDial --output_dir output/pandagpt/VisDial/pandagpt_likelihood_dup1_prefix \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run_eval.py \
    --model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/ \
    --dataset_name VisDial --output_dir output/pandagpt/VisDial/pandagpt_generation_dup1_prefix \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100