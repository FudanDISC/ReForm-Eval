CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/llama_adapterv2/VisDial/llama_adapterv2_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval  --dataset_duplication 1 \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/llama_adapterv2/VisDial/llama_adapterv2_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval  --dataset_duplication 1 \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/llama_adapterv2/VisDial/llama_adapterv2_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/llama_adapterv2/VisDial/llama_adapterv2_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation  --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/test_20231017/VisDial/llama_adapterv2_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval  --dataset_duplication 1 \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name MEDIC --output_dir output/test_20231017/MEDIC/llama_adapterv2_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval  --dataset_duplication 5 \
    --dataset_config build/configs/DisasterType_val.yaml \
    --do_eval --offline_hf

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --in_context_sample \
#     --dataset_name MEDIC --dataset_config build/configs/DisasterType_val.yaml \
#     --output_dir output/bhwu_output/dts/lynx/eval_ll_v1 \
#     --infer_method likelihood --half_evaluation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice \
#     --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --in_context_sample \
    --dataset_name MEDIC --dataset_config build/configs/DisasterType_val.yaml \
    --output_dir output/test_20231017/dts/blip2_t5/eval_ll_v1 \
    --infer_method likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  run_eval.py \
    --model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b \
    --in_context_sample \
    --dataset_name MEDIC --dataset_config build/configs/DisasterType_val.yaml \
    --output_dir output/test_20231017/dts/blip2_vicuna_instruct/eval_ll_v1 \
    --infer_method likelihood --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --in_context_sample \
    --dataset_name MEDIC --dataset_config build/configs/DisasterType_val.yaml \
    --output_dir output/test_20231017/dts/shikra/eval_ll_v1 \
    --infer_method likelihood --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5 --offline_hf

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model blip2 --model_name blip2_t5_instruct --model_type flant5xl \
    --in_context_sample \
    --dataset_name MEDIC --dataset_config build/configs/DisasterType_val.yaml \
    --output_dir output/test_20231017/dts/blip2_t5_instruct/eval_ll_v1 \
    --infer_method likelihood --half_evaluation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 5