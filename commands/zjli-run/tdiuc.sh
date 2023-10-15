for dc in "datasets/configs/TDIUC_scene.yaml" "datasets/configs/TDIUC.yaml" "datasets/configs/TDIUC_color.yaml" "datasets/configs/TDIUC_counting.yaml" "datasets/configs/TDIUC_detection.yaml" "datasets/configs/TDIUC_position.yaml"
do
echo $dc
posfix=(${dc//_/ })
posfix=${posfix[1]}
posfix=(${posfix//./ })
posfix=${posfix[0]}
echo $posfix
torchrun --nproc_per_node 8 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name TDIUC --output_dir tmp_rest_output/refined_tdiuc_part4/TDIUC_$posfix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval  --dataset_config $dc  --half_evaluation

torchrun --nproc_per_node 8 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name TDIUC --output_dir tmp_rest_output/refined_tdiuc_part4/TDIUC_$posfix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation  --in_context_sample --temperature 0.2 --do_eval --option_mark upper --dataset_config $dc  --half_evaluation
done