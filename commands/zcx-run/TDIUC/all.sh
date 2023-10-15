source /root/anaconda3/etc/profile.d/conda.sh

############### Sport 
dm="TDIUC"
dc="datasets/configs/TDIUC.yaml"

conda activate blip2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Sport/flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc\

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Sport/flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Sport/instruct_flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Sport/instruct_flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Sport/instruct_vicuna7b/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Sport/instruct_vicuna7b/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

conda activate llava
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Sport/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Sport/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \


############### Scene
dm="TDIUC"
dc="datasets/configs/TDIUC_scene.yaml"
conda activate blip2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Scene/flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc\

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Scene/flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Scene/instruct_flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Scene/instruct_flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Scene/instruct_vicuna7b/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Scene/instruct_vicuna7b/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

conda activate llava
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Scene/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Scene/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \


########### color 

dm="TDIUC"
dc="datasets/configs/TDIUC_color.yaml"
conda activate blip2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Color/flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc\

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Color/flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Color/instruct_flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Color/instruct_flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Color/instruct_vicuna7b/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Color/instruct_vicuna7b/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

conda activate llava
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Color/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Color/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \


########### color 
dm="TDIUC"
dc="datasets/configs/TDIUC_position.yaml"
conda activate blip2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Position/flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc\

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Position/flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Position/instruct_flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Position/instruct_flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Position/instruct_vicuna7b/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Position/instruct_vicuna7b/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

conda activate llava
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Position/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Position/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \


############### detection
dm="TDIUC"
dc="datasets/configs/TDIUC_detection.yaml"
conda activate blip2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Detection/flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc\

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Detection/flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Detection/instruct_flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Detection/instruct_flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Detection/instruct_vicuna7b/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Detection/instruct_vicuna7b/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

conda activate llava
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Detection/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Detection/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

################ counting

source /root/anaconda3/etc/profile.d/conda.sh

dm="TDIUC"
dc="datasets/configs/TDIUC_counting.yaml"
conda activate blip2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Count/flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc\

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Count/flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Count/instruct_flant5xl/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Count/instruct_flant5xl/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Count/instruct_vicuna7b/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/blip2/TDIUC_Count/instruct_vicuna7b/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \

conda activate llava
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Count/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name $dm --output_dir tmp_rest_output/new_TDIUC/llava/TDIUC_Count/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation  \




