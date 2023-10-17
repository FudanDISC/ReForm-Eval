CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
    --model kosmos2  --model_name kosmos2  --model_type /remote-home/share/multimodal-models/kosmos-2-patch14-224 \
    --dataset_name VisDial --output_dir output/kosmos2/VisDial/kosmos2_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
    --model kosmos2  --model_name kosmos2  --model_type /remote-home/share/multimodal-models/kosmos-2-patch14-224 \
    --dataset_name VisDial --output_dir output/kosmos2/VisDial/kosmos2_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100