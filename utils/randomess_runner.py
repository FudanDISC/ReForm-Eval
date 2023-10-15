import argparse
import os

model_to_args = {
    'blip2_t5': '--model blip2  --model_name blip2_t5  --model_type pretrain_flant5xl',
    'instructblip_t5': '--model blip2  --model_name blip2_t5_instruct  --model_type flant5xl',
    'instructblip_vicuna': '--model blip2  --model_name blip2_vicuna_instruct  --model_type vicuna7b',
    'llava_v0': '--model llava  --model_name /remote-home/share/multimodal-models/llava/LLaVA-7B-v0/',
    'llava_llama2': '--model llava  --model_name /remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ --model_type /remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/',
    'minigpt4': '--model minigpt4  --model_name models/MiniGPT-4/eval_configs/minigpt4_eval.yaml',
    'mplug_owl': '--model mplugowl  --model_name mplugowl --model_type /remote-home/share/multimodal-models/mplug-owl-llama-7b/',
    'llama_adapter_v2': '--model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data',
    'imagebind_llm': '--model imagebindLLM  --model_name imagebindLLM --model_type /remote-home/share/multimodal-models/imagebindllm_ckpts',
    'pandagpt': '--model pandagpt  --model_name pandagpt --model_type /remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/',
    'lynx': '--model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml',
    'cheetor_vicuna': '--model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml',
    'cheetor_llama2': '--model cheetor  --model_name models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml',
    'shikra': '--model shikra  --model_name /remote-home/share/multimodal-models/shikra-7b-v1/',
    'bliva': '--model bliva  --model_name bliva_vicuna',
    'mmgpt': '--model mmgpt  --model_name Multimodal-GPT'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',type=str, default=None, help='the models to test')
    parser.add_argument('--devices', type=str, default=None, help='the devices to use')
    args = parser.parse_args()
    
    model_names = args.models.split(',')
    for model_name in model_names:
        assert model_name in model_to_args, '{} not in model zoo'.format(model_name)
    
    num_devices = len(args.devices.split(','))

    base_command = 'CUDA_VISIBLE_DEVICES={cuda_devices} torchrun --master_port 61234 --nproc_per_node {num_devices} run_eval.py \
    {model_config} \
    --dataset_name VQA_Random --output_dir output/{model_name}/scienceqa_random/{randomness} \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --infer_method generation --dataset_duplication 5  --option_mark {option_mark} \
    --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    --temperature 0.2  --in_context_sample {half} {random_arg}'

    base_command_likelihood = 'CUDA_VISIBLE_DEVICES={cuda_devices} torchrun --master_port 61234 --nproc_per_node {num_devices} run_eval.py \
    {model_config} \
    --dataset_name VQA_Random --output_dir output/{model_name}/scienceqa_random/{randomness} \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --infer_method likelihood --dataset_duplication 5  \
    --dataset_config datasets/configs/VQA_scienceqa_randomness_val_v2.0.yaml \
    {half} {random_arg}'

    for model_name in model_names:
        model_config = model_to_args[model_name]
        if 't5' not in model_name:
            half = '--half_evaluation'
        else:
            half = ''
        command_args = {'model_name': model_name, 'model_config':model_config, 'half': half,
                        'cuda_devices': args.devices, 'num_devices': num_devices}
        # random instruct
        command_args_v1 = {k:v for k,v in command_args.items()}
        command_args_likelihood_v1 = {k:v for k,v in command_args.items()}
        command_args_v1.update({'randomness': 'random_instruct', 'random_arg': '--random_instruct',
                                'option_mark': 'upper'})
        command_args_likelihood_v1.update({'randomness': 'random_instruct', 'random_arg': '--random_instruct'})
        command1 = base_command.format(**command_args_v1)
        command1_likelihood = base_command_likelihood.format(**command_args_likelihood_v1)
        print(command1)
        os.system(command1)
        print(command1_likelihood)
        os.system(command1_likelihood)

        # shuffle options
        command_args_v2 = {k:v for k,v in command_args.items()}
        command_args_v2.update({'randomness': 'shuffle_options', 'random_arg': '--shuffle_options',
                                'option_mark': 'upper'})
        command2 = base_command.format(**command_args_v2)
        print(command2)
        os.system(command2)

        # option mark
        command_args_v3 = {k:v for k,v in command_args.items()}
        command_args_v3.update({'randomness': 'random_mark', 'random_arg': '',
                                'option_mark': 'random'})
        command3 = base_command.format(**command_args_v3)
        print(command3)
        os.system(command3)

if __name__=='__main__':
    main()