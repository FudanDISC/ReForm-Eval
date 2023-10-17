CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
    --model visualglm  --model_name visualglm  --model_type /remote-home/share/multimodal-models/visualglm-6b \
    --dataset_name VisDial --output_dir output/visualglm/VisDial/visualglm_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model visualglm  --model_name visualglm  --model_type /remote-home/share/multimodal-models/visualglm-6b \
    --dataset_name VisDial --output_dir output/visualglm/VisDial/visualglm_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100
