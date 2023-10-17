# # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 run_eval.py \
# #     --model mplugowl --model_name mplugowl\
# #     --dataset_name MSCOCO --dataset_config build/configs/ImageTextMatching_val.yaml \
# #     --output_dir output/bhwu_output/its/mplugowl/likelihood  \
# #     --infer_method likelihood \
# #     --per_gpu_eval_batch_size 1 --formulation SingleChoice \
# #     --do_eval --dataset_duplication 5


# # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 run_eval.py \
# #     --model mplugowl  --model_name mplugowl \
# #     --dataset_name COCO_text --output_dir output/ocr_output/cocotext_gocr5000/mplugowl_likelihood \
# #     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
# #     --do_eval  --dataset_duplication 10 \
# #     --dataset_config build/configs/GroundOCR_cocotext_val.yaml

# # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
# #     --model mplugowl  --model_name mplugowl \
# #     --dataset_name COCO_text --output_dir output/ocr_output/cocotext_gocr5000mplugowl_likelihood \
# #     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
# #     --infer_method likelihood --do_eval  --dataset_duplication 10 \
# #     --dataset_config build/configs/GroundOCR_cocotext_val.yaml


# # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 run_eval.py \
# #     --model mplugowl --model_name mplugowl \
# #     --dataset_name MSCOCO --dataset_config build/configs/ImageTextMatching_val.yaml \
# #     --output_dir output/bhwu_output/its/mplugowl/likelihood  \
# #     --infer_method likelihood \
# #     --per_gpu_eval_batch_size 1 --formulation SingleChoice \
# #     --do_eval --dataset_duplication 5

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/mplugowl_likelihood_2 \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/OCR_iiit5k_val.yaml \
#     --dataset_subsample 100

# CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node=4 run_eval.py \
#     --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/mplugowl_generation_3 \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --in_context_sample --option_mark upper \
#     --infer_method generation --do_eval  --dataset_duplication 1 \
#     --dataset_config build/configs/OCR_iiit5k_val.yaml \
#     # --dataset_subsample 100 
    



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 run_eval.py \
#     --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/mplugowl_likelihood_dup5 \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 5 \
#     --dataset_config build/configs/OCR_iiit5k_val.yaml


# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name MSCOCO --output_dir output/mplug_owl/multiclass_identification/mplugowl_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/MulticlassIdentification_val.yaml


# # GroundedObjIdentification_val

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name MSCOCO --output_dir output/mplug_owl/GroundedObjIdentification/mplugowl_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 1 \
#     --dataset_config build/configs/GroundedObjIdentification_val.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
    --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/VisDial/mplugowl_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval  --dataset_duplication 1 \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --in_context_sample --option_mark upper \
    --dataset_subsample 100


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
    --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/VisDial/mplugowl_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval  --dataset_duplication 1 \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 run_eval.py \
#     --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
#     --dataset_name VisDial --output_dir output/mplug_owl/VisDial/mplugowl_likelihood_dup1 \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 1 \
#     --dataset_config build/configs/VisDial_val.yaml \
#     --dataset_subsample 10 

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/VisDial/mplugowl_generation_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --options_in_history --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 10


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_eval.py \
    --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/VisDial/mplugowl_likelihood_dup1 \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 1 \
    --in_context_sample --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --dataset_subsample 100