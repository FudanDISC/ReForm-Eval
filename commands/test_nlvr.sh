# Natural Language Visual Reasoning --master_port='29500'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --in_context_sample --option_mark upper \
    --dataset_name NLVR --dataset_config build/configs/NaturalLanguageVisualReasoning_val.yaml \
    --output_dir output/bhwu_output/nlvrm/mplugowl \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5