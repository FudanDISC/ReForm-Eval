## OCR
# on llava_vicuna_instruct 
## cocotext
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 run_eval.py \
    --model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --in_context_sample --option_mark upper \
    --dataset_name COCO_text --dataset_config datasets/configs/OCR_cocotext_val.yaml \
    --output_dir output/ocr_output/cocotext_ocr/llava_generation \
    --infer_method generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice \
    --do_eval --dataset_duplication 10

# # evaluate using the generation to predict the answer
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name COCO_text --output_dir output/ocr_output/cocotext_ocr/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_cocotext_val.yaml

# ## cocotext
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name COCO_text --output_dir output/ocr_output/cocotext_gocr/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/GroundOCR_cocotext_val.yaml

# # evaluate using the generation to predict the answer
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name COCO_text --output_dir output/ocr_output/cocotext_gocr/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/GroundOCR_cocotext_val.yaml

# ## cute80
# # evaluate using the likelihood to predict the answer 
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name CUTE80 --output_dir output/ocr_output/cute80/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_cute80_val.yaml

# # evaluate using the generation to predict the answer
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name CUTE80 --output_dir output/ocr_output/cute80/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method generation --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_cute80_val.yaml

# ## ic15
# # evaluate using the likelihood to predict the answer 
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name IC15 --output_dir output/ocr_output/ic15/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_ic15_val.yaml

# # evaluate using the generation to predict the answer
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name IC15 --output_dir output/ocr_output/ic15/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method generation --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_ic15_val.yaml

# ## iiit5k
# # evaluate using the likelihood to predict the answer 
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_iiit5k_val.yaml

# # evaluate using the generation to predict the answer
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name IIIT5K --output_dir output/ocr_output/iiit5k/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method generation --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_iiit5k_val.yaml

# ## wordart
# # evaluate using the likelihood to predict the answer 
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name WordArt --output_dir output/ocr_output/wordart/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_wordart_val.yaml

# # evaluate using the generation to predict the answer
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name WordArt --output_dir output/ocr_output/wordart/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method generation --do_eval  --dataset_duplication 10 \
#     --dataset_config datasets/configs/OCR_wordart_val.yaml