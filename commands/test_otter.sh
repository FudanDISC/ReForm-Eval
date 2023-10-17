# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model otter  --model_name otter  --model_type luodian/OTTER-9B-LA-InContext \
#     --dataset_name MSCOCO --output_dir output/otter/GroundedObjIdentification/mplugowl_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 1 \
#     --dataset_config build/configs/GroundedObjIdentification_val.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
    --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-Image-LLaMA7B-LA-InContext \
    --dataset_name VisDial --output_dir output/otter/VisDial/otter_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval  --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run_eval.py \
    --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-Image-LLaMA7B-LA-InContext \
    --dataset_name VisDial --output_dir output/otter/VisDial/otter_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval  --dataset_duplication 1 \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

# CUDA_VISIBLE_DEVICES=1,5,7 torchrun --nproc_per_node=3 run_eval.py \
#     --model otter  --model_name otter --model_type luodian/OTTER-Image-LLaMA7B-LA-InContext \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/otter_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/OCR_iiit5k_val.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
#     --model otter  --model_name otter --model_type luodian/OTTER-Image-LLaMA7B-LA-InContext \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/otter_generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice \
#     --in_context_sample --option_mark upper \
#     --infer_method generation --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/OCR_iiit5k_val.yaml \


# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 run_eval.py \
#     --model otter  --model_name otter --model_type luodian/OTTER-Image-LLaMA7B-LA-InContext \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/otter_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/OCR_iiit5k_val.yaml \
#     --in_context_sample --option_mark upper \
#     --dataset_subsample 10



CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --nproc_per_node=4 run_eval.py \
    --model otter --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
    --dataset_name VisDial --output_dir output/otter/VisDial/otter_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --nproc_per_node=4 run_eval.py \
    --model otter --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-Image-LLaMA7B-LA-InContext \
    --dataset_name VisDial --output_dir output/otter/VisDial/otter_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run_eval.py \
    --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
    --dataset_name VisDial --output_dir output/otter/VisDial/otter_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
    --dataset_name VisDial --output_dir output/otter/VisDial/otter_generation_dup1_9B_LA_s100 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100 --capitalize

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-Image-LLaMA7B-LA-InContext \
    --dataset_name VisDial --output_dir output/otter/VisDial/otter_generation_dup1_s100 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100 --capitalize