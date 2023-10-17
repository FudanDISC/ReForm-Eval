## spatial
## clevr
# ### multiple choice
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name CLEVR --output_dir output/spatial_output/clevr/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_clevr_val.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name CLEVR --output_dir output/spatial_output/clevr/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_clevr_val.yaml
## true or false
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name CLEVR --output_dir output/spatial_output/clevr_tf/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_clevr_val.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name CLEVR --output_dir output/spatial_output/clevr_tf/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_clevr_val.yaml

# ## open ended
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name CLEVR --output_dir output/spatial_output/clevr_oe/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation OCROpenEnded \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_clevr_val.yaml

## vsr
## multiple choice

### instruct_blip
# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name VSR --output_dir output/spatial_output/vsr/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_vsr_val.yaml

# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name VSR --output_dir output/spatial_output/vsr/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_vsr_val.yaml

# ### otter
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
#     --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
#     --dataset_name VSR --output_dir output/spatial_output/vsr/otter_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method generation --do_eval --half_evaluation --dataset_duplication 10 \
#     --in_context_sample --option_mark upper \
#     --dataset_config build/configs/Spatial_vsr_val.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
#     --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
#     --dataset_name VSR --output_dir output/spatial_output/vsr/otter_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 10 \
#     --in_context_sample --option_mark upper \
#     --dataset_config build/configs/Spatial_vsr_val.yaml

### llama_adapterv2
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
#     --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
#     --dataset_name VSR --output_dir output/spatial_output/vsr/llama_adapterv2_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method generation --do_eval --half_evaluation --dataset_duplication 10 \
#     --option_mark upper \
#     --dataset_config build/configs/Spatial_vsr_val.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
#     --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
#     --dataset_name VSR --output_dir output/spatial_output/vsr/llama_adapterv2_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 10 \
#     --option_mark upper \
#     --dataset_config build/configs/Spatial_vsr_val.yaml

CUDA_VISIBLE_DEVICES=0,2,3,4,6,7 torchrun --nproc_per_node=6 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name VSR --output_dir output/spatial_output/vsr/imagebindLLM \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice \
    --infer_method generation --do_eval --half_evaluation --dataset_duplication 5 \
    --option_mark upper --in_context_sample \
    --dataset_config build/configs/Spatial_vsr_val.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
#     --model mplugowl  --model_name mplugowl  --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b \
#     --dataset_name VSR --output_dir output/spatial_output/vsr/mplugowl_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval --half_evaluation --dataset_duplication 10 \
#     --option_mark upper --in_context_sample\
#     --dataset_config build/configs/Spatial_vsr_val.yaml

### true or false
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name VSR --output_dir output/spatial_output/vsr_tf/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_vsr_val.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name VSR --output_dir output/spatial_output/vsr_tf/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_vsr_val.yaml


## mp3d
### oo_relation
#### multiple_choice
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oo_relation/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oo_relation_val.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oo_relation/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oo_relation_val.yaml
#### true or false
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oo_relation_tf/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oo_relation_val.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oo_relation_tf/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oo_relation_val.yaml
### oa_depth
#### multiple_choice
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oa_depth/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oa_depth_val.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oa_depth/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oa_depth_val.yaml
#### true or false
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oa_depth_tf/blip2_vicuna_instruct_likelihood \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --infer_method likelihood --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oa_depth_val.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
#     --model blip2  --model_name blip2_vicuna_instruct --model_type vicuna7b \
#     --dataset_name MP3D --output_dir output/spatial_output/mp3d/oa_depth_tf/blip2_vicuna_instruct_generation \
#     --per_gpu_eval_batch_size 4 --formulation SingleChoice \
#     --do_eval  --dataset_duplication 10 \
#     --dataset_config build/configs/Spatial_mp3d_oa_depth_val.yaml