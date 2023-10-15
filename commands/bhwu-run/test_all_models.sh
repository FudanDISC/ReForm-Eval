source /remote-home/bhwu/anaconda3/etc/profile.d/conda.sh

conda activate llm-v-bench
# BLIP2 flant5xl
--model blip2 --model_name blip2_t5  --model_type pretrain_flant5xl
# instructBLIP vicuna7B
--model blip2 --model_name blip2_vicuna_instruct --model_type vicuna7b
# instructBLIP flant5xl
--model blip2 --model_name blip2_t5_instruct --model_type flant5xl
# minigpt4
--model minigpt4 --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml


conda activate llm-v-bench-llava
# llava llama
--model llava --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/
# llava llama2
--model llava --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/


conda activate mplugowl
# mplugowl
--model mplugowl --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/


conda activate imagebind_LLM
# imagebind_LLM
--model imagebindLLM --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts


conda activate llama_adapter_v2
# llama_adapter_v2
--model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data


conda activate llm-v-bench-mmgpt
# mmgpt
--model mmgpt --model_name Multimodal-GPT


conda activate pandagpt
# pandagpt
--model pandagpt  --model_name pandagpt  --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/


conda activate shikra
# shikra
--model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/


conda activate cheetah
# cheetor vicuna
--model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml
# cheetor llama2
--model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml.yaml


conda activate bliva
# bliva
--model bliva --model_name bliva_vicuna


conda activate lynx
# lynx
--model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml





