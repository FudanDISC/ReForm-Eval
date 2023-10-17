source /root/anaconda3/etc/profile.d/conda.sh

#### Pets37
# dm="Pets37"
# dc="build/configs/ImageClassification_pets37_val.yaml"
# conda activate llava
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir tmp_rest_output/bliva/pets37/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir tmp_rest_output/bliva/pets37/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# conda activate lynx
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/lynx/pets37/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/lynx/pets37/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# conda activate imagebind_LLM
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir tmp_rest_output/imagebindLLM/pets37/likelihood \
#     --per_gpu_eval_batch_size 8 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \


#### VizWiz_4
# dm="VizWiz"
# dc="build/configs/ImageQuality_vizwiz_singleChoice_val.yaml"
# conda activate imagebind_LLM
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir tmp_rest_output/imagebindLLM/vizwiz/singleChoice/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir tmp_rest_output/imagebindLLM/vizwiz/singleChoice/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# conda activate lynx
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/lynx/vizwiz/singleChoice/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/lynx/vizwiz/singleChoice/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \


# conda activate bliva
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir tmp_rest_output/bliva/vizwiz/singleChoice/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir tmp_rest_output/bliva/vizwiz/singleChoice/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# #### VizWiz_2
# dm="VizWiz"
# dc="build/configs/ImageQuality_vizwiz_yesNo_val.yaml"

# conda activate imagebind_LLM
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir tmp_rest_output/imagebindLLM/vizwiz/yesNo/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
#     --dataset_name $dm --output_dir tmp_rest_output/imagebindLLM/vizwiz/yesNo/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# conda activate lynx
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/lynx/vizwiz/yesNo/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/lynx/vizwiz/yesNo/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# conda activate bliva
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir tmp_rest_output/bliva/vizwiz/yesNo/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir tmp_rest_output/bliva/vizwiz/yesNo/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \


#### TDIUC_scene
dm="TDIUC"
dc="build/configs/TDIUC_scene.yaml"

# conda activate minigpt4
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/minigpt4/TDIUC_Scene/likelihood \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method likelihood --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \
    
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
#     --dataset_name $dm --output_dir tmp_rest_output/minigpt4/TDIUC_Scene/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir tmp_rest_output/mmgpt/new_TDIUC_Scene/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name $dm --output_dir tmp_rest_output/mmgpt/new_TDIUC_Scene/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

# conda activate bliva
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
#     --model bliva  --model_name bliva_vicuna \
#     --dataset_name $dm --output_dir tmp_rest_output/bliva/TDIUC_Scene/generation \
#     --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
#     --infer_method generation --do_eval --option_mark upper \
#     --dataset_config $dc --half_evaluation \


