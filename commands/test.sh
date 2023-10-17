# for test
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name VisDial --output_dir output/blip2_visdial/ --per_gpu_eval_batch_size 1 --formulation SingleChoice

# for stability test on all 8GPUs
torchrun --nproc_per_node=8 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name VisDial --output_dir output/blip2_visdial/stability \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice  \
    --do_eval  --half_evaluation  --dataset_duplication 5

# for stability test on all 8 GPUs (notice that --half_evaluation does not suit BLIP2-FlanT5)
torchrun --nproc_per_node=8 run_eval.py \
    --model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl \
    --dataset_name VisDial --output_dir output/blip2_visdial/test_half \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice  \
    --do_eval   --dataset_duplication 5

# evaluate using the likelihood to predict the answer
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
    --model blip2  --model_name blip2_t5_instruct  --model_type flant5xl \
    --dataset_name VisDial --output_dir output/blip2_visdial/fix_4choice/ \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --do_eval \
    --infer_method likelihood --do_eval   --dataset_duplication 5 \
    --dataset_config build/configs/VisDial_val_v1.1.yaml

# evaluate using the likelihood on blip2_vicuna_instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b \
    --dataset_name VisDial --output_dir output/blip2_visdial/fix_4choice/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --infer_method likelihood --do_eval   --dataset_duplication 5 \
    --dataset_config build/configs/VisDial_val_v1.1.yaml --half_evaluation

# evluate the llava on visdial (generation)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 2 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name Flowers102 --output_dir output/llava/flowers102/test_hit/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config build/configs/ImageClassification_flowers102_val.yaml --half_evaluation

# evluate the llava on visdial (likelihood)

torchrun --nproc_per_node 8 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
    --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
    --dataset_name VisDial --output_dir output/llava/visdial/standard/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval  --multi_round_eval --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.1.yaml --half_evaluation \
    --online_multi_round --num_workers 0  --random_instruct  --shuffle_options

# standard aligned online multi-round evaluation for minigpt4 on visdial
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name VisDial --output_dir output/minigpt4/visdial/test/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --online_multi_round   --multi_round_eval  --num_workers 0 --options_in_history \
    --infer_method generation --do_eval   --dataset_duplication 5  --temperature 0.2 \
    --dataset_config build/configs/VisDial_val_v1.1.yaml --half_evaluation --dataset_subsample 20 
    
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 61234 --nproc_per_node 4 run_eval.py \
    --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name VisDial --output_dir output/minigpt4/visdial/test/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --online_multi_round   --multi_round_eval  --num_workers 0 \
    --infer_method likelihood --do_eval   --dataset_duplication 5 \
    --dataset_config build/configs/VisDial_val_v1.1.yaml --half_evaluation --dataset_subsample 20 

## test mplug_owl
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 run_eval.py \
    --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/visdial/test_mem/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval  --multi_round_eval --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml --half_evaluation \
    --online_multi_round --num_workers 0   --dataset_subsample 20

## test shikra
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 8 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name VisDial --output_dir output/shikra/visdial/test/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval  --multi_round_eval --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.1.yaml --half_evaluation \
    --online_multi_round --num_workers 0   --dataset_subsample 20  --options_in_history \
    --random_instruct  --shuffle_options

# test otter
## test mplug_owl
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 2 run_eval.py \
    --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/vqa_MR/test_run/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval  --multi_round_eval --random_instruct \
    --dataset_config build/configs/VQA_vqa_MultiRound_val.yaml --half_evaluation \
    --online_multi_round --num_workers 0   --dataset_subsample 20

torchrun --nproc_per_node 2 run_eval.py \
    --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/visdial/v1.2/standard/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval  --multi_round_eval --random_instruct \
    --dataset_config build/configs/VisDial_val_v1.2.yaml --half_evaluation --option_mark upper \
    --online_multi_round --num_workers 0   --options_in_history  --in_context_sample  --shuffle_options

torchrun --nproc_per_node 2 run_eval.py \
    --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VisDial --output_dir output/mplug_owl/visdial/v1.2/standard/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval  --multi_round_eval --random_instruct \
    --dataset_config build/configs/VisDial_val_v1.2.yaml --half_evaluation \
    --online_multi_round --num_workers 0 

# llama adapter v2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/llama_adapter_v2/visdial/v1.2/standard/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval  --multi_round_eval --random_instruct \
    --dataset_config build/configs/VisDial_val_v1.2.yaml --half_evaluation --option_mark upper \
    --online_multi_round --num_workers 0   --options_in_history  --in_context_sample  --shuffle_options

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/llama_adapter_v2/visdial/v1.2/standard/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval  --multi_round_eval --random_instruct \
    --dataset_config build/configs/VisDial_val_v1.2.yaml --half_evaluation \
    --online_multi_round --num_workers 0 

# test VQA mr
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 61234 --nproc_per_node 2 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name VisDial --output_dir output/llava/VQA_MR/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval  --multi_round_eval --option_mark upper \
    --dataset_config build/configs/VQA_vqa_MultiRound_val.yaml  --in_context_sample \
    --online_multi_round --num_workers 0  --random_instruct  --shuffle_options  --options_in_history
# naive VQA mr
torchrun --master_port 61234 --nproc_per_node 2 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/ \
    --dataset_name VQA --output_dir output/llava/VQA_MR/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample 


# test format hit with naive vqa-MR
torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VQA --output_dir output/llama_adapterv2/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VQA --output_dir output/llama_adapterv2/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method likelihood --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options --half_evaluation

# mplug_owl
torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VQA --output_dir output/imageBind_llm/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
    --dataset_name VQA --output_dir output/mplug_owl/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

# minigpt-4
torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name VQA --output_dir output/minigpt4/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
    --dataset_name VQA --output_dir output/minigpt4/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

# llava
torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/  \
    --dataset_name VQA --output_dir output/imageBind_llm/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
    --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
    --dataset_name VQA --output_dir output/imageBind_llm/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options --half_evaluation  --in_context_sample

torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
    --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/  \
    --dataset_name VQA --output_dir output/imageBind_llm/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation


# imageBind_LLM
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM  --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name VQA --output_dir output/imagebindLLM/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts \
    --dataset_name VQA --output_dir output/imagebindLLM/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

# otter
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model otter  --model_name otter  --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
    --dataset_name VQA --output_dir output/otter/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model otter  --model_name otter --model_type /remote-home/share/multimodal-models/OTTER-9B-LA-InContext \
    --dataset_name VQA --output_dir output/otter/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

# pandaGPT
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model pandagpt  --model_name pandagpt \
    --dataset_name VQA --output_dir output/pandagpt/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model pandagpt  --model_name pandagpt \
    --dataset_name VQA --output_dir output/pandagpt/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model pandagpt  --model_name pandagpt \
    --dataset_name VQA --output_dir output/pandagpt/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method likelihood --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options --half_evaluation

# lynx
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name VQA --output_dir output/lynx/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name VQA --output_dir output/lynx/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name VQA --output_dir output/lynx/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method likelihood --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options --half_evaluation

# cheetor
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
    --dataset_name VQA --output_dir output/cheetor/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
    --dataset_name VQA --output_dir output/cheetor/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml \
    --dataset_name VQA --output_dir output/cheetor/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method likelihood --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options --half_evaluation

# shikra
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name VQA --output_dir output/cheetor/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --in_context_sample --half_evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name VQA --output_dir output/cheetor/VQA_MR_naive/test_prefix/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method generation --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options  --half_evaluation

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/ \
    --dataset_name VQA --output_dir output/shikra/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method likelihood --do_eval --option_mark upper  \
    --dataset_config build/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options --half_evaluation

#bliva
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node 8 run_eval.py \
    --model bliva --model_name bliva_vicuna \
    --dataset_name VisDial --output_dir output/bliva/visdial/v1.2/standard \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --infer_method generation --dataset_duplication 5 --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml --random_instruct \
    --online_multi_round --num_workers 0  --temperature 0.2 \
    --multi_round_eval --shuffle_options --half_evaluation --options_in_history --in_context_sample

#MMGPT
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 run_eval.py \
     --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name VisDial --output_dir output/mmgpt/VisDial/likelihood_test \
    --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config build/configs/VisDial_val_v1.2.yaml \
    --num_workers 0 --half_evaluation --dataset_subsample 100

