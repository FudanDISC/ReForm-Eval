source /remote-home/mfdu/anaconda3/etc/profile.d/conda.sh
dm=TDIUC
function eval_tdiuc(){
    ## llava llama2
    conda activate /remote-home/mfdu/anaconda3/envs/llava
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
        --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/llava-llama-2/${store_name}/likelihood \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 \
        --infer_method likelihood --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation \

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
        --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/ \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/llava-llama-2/${store_name}/generation \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
        --infer_method generation --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation\

    ###################################  MiniGPT4
    conda activate /remote-home/mfdu/anaconda3/envs/minigpt4

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/minigpt4/${store_name}/likelihood \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 \
        --infer_method likelihood --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation \
        
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/minigpt4/${store_name}/generation \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
        --infer_method generation --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation\

    ###################################  mPLUG-owl
    conda activate /remote-home/mfdu/anaconda3/envs/mplugowl

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/mplug_owl/${store_name}/likelihood \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 \
        --infer_method likelihood --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation \

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/ \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/mplug_owl/${store_name}/generation \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
        --infer_method generation --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation\

    ###################################  llama-adapter-v2
    conda activate /remote-home/mfdu/anaconda3/envs/llama_adapter_v2

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/llama_adapter_v2/${store_name}/likelihood \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 \
        --infer_method likelihood --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation \

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run_eval.py \
        --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
        --dataset_name $dm --output_dir output/mfdu_output/tdiuc/llama_adapter_v2/${store_name}/generation \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
        --infer_method generation --do_eval --option_mark upper \
        --dataset_config $dc --half_evaluation
}

# dc="datasets/configs/TDIUC_color.yaml"
# store_name=TDIUC_color
# eval_tdiuc
# dc="datasets/configs/TDIUC_counting.yaml"
# store_name=TDIUC_counting
# eval_tdiuc
# dc="datasets/configs/TDIUC_detection.yaml"
# store_name=TDIUC_detection
# eval_tdiuc
# dc="datasets/configs/TDIUC_position.yaml"
# store_name=TDIUC_position
# eval_tdiuc
# dc="datasets/configs/TDIUC_scene.yaml"
# store_name=TDIUC_scene
# eval_tdiuc
dc="datasets/configs/TDIUC.yaml"
store_name=TDIUC_sport
eval_tdiuc
