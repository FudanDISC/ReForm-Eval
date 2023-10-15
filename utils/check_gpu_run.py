import os
import sys
import time

cmd = 'CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data \
    --dataset_name VisDial --output_dir output/llama_adapter_v2/vqa_mr/standard_prefix \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --infer_method generation --dataset_duplication 5  --option_mark upper \
    --dataset_config datasets/configs/VQA_vqa_MultiRound_val.yaml --random_instruct \
    --online_multi_round  --num_workers 0   --temperature 0.2  --options_in_history  --in_context_sample \
    --multi_round_eval   --shuffle_options --half_evaluation'


cmd2 = 'CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 7 run_eval.py \
    --model llama_adapterv2  --model_name llama_adapterv2  --model_type /remote-home/share/multimodal-models/pyllama_data  \
    --dataset_name VisDial --output_dir output/llama_adapter_v2/vqa_mr/standard \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --do_eval \
    --infer_method likelihood --dataset_duplication 5  --option_mark upper \
    --dataset_config datasets/configs/VQA_vqa_MultiRound_val.yaml --random_instruct \
    --online_multi_round  --num_workers 0   --temperature 0.2   \
    --multi_round_eval   --shuffle_options --half_evaluation'

cmd3 = 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 61234 --nproc_per_node 8 run_eval.py \
    --model mmgpt  --model_name Multimodal-GPT \
    --dataset_name VQA --output_dir output/shikra/VQA_MR_naive/test_icl/ \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 1 \
    --infer_method likelihood --do_eval --option_mark upper  \
    --dataset_config datasets/configs/VQA_vqa_MultiRound_naive.yaml \
    --num_workers 0  --random_instruct  --shuffle_options --half_evaluation'


def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return power, memory    


def narrow_setup(interval=2):
    id = [2,3,4,5,6,7]
    flag = True
    while flag:
        print('checking gpu info at {}'.format(time.asctime()))
        for gpu_id in id:
            gpu_power, gpu_memory = gpu_info(gpu_id)
            if gpu_memory > 1000:
                flag = True
                print('GPU {} not valid, used {} memory'.format(gpu_id, gpu_memory))
                break
            else:
                flag = False
        if flag:
            print('not valid, sleep for {} seconds'.format(interval))
            time.sleep(interval)
    print('GPU available, running now!')
    os.system(cmd)
    os.system(cmd2)
    # os.system(cmd3)


if __name__ == '__main__':
    narrow_setup(600)