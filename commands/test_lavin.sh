CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model lavin  --model_name lavin  --model_type /remote-home/share/multimodal-models \
    --dataset_name VisDial --output_dir output/lavin/VisDial/lavin_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config datasets/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model lavin  --model_name lavin  --model_type /remote-home/share/multimodal-models \
    --dataset_name VisDial --output_dir output/lavin/VisDial/lavin_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config datasets/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run_eval.py \
    --model lavin  --model_name lavin  --model_type /remote-home/share/multimodal-models \
    --dataset_name VisDial --output_dir output/lavin/VisDial/lavin_generation_dup1_bs1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config datasets/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run_eval.py \
    --model lavin  --model_name lavin  --model_type /remote-home/share/multimodal-models \
    --dataset_name VisDial --output_dir output/lavin/VisDial/lavin_likelihood_dup1_bs1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config datasets/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100