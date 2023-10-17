source /remote-home/mfdu/anaconda3/etc/profile.d/conda.sh
infer_method=likelihood
formulation=SingleChoice
duplication=5
batch_size=4


function test_all(){
    conda activate /remote-home/mfdu/anaconda3/envs/blip2
    model=blip2
    model_name=blip2_t5
    model_type=pretrain_flant5xl
    store_model_name=blip2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 

        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=blip2
    model_name=blip2_t5_instruct
    model_type=flant5xl
    store_model_name=instructblip2_flant5
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 

        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=blip2
    model_name=blip2_vicuna_instruct
    model_type=vicuna7b
    store_model_name=instructblip2_vicuna
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 

        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag



    conda activate /remote-home/mfdu/anaconda3/envs/cheetah
    model=cheetor
    model_name=models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml

    store_model_name=cheetor_vicuna
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=cheetor
    model_name=models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml

    store_model_name=cheetor_llama2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    conda activate /remote-home/mfdu/anaconda3/envs/mmgpt
    model=mmgpt
    model_name=Multimodal-GPT

    store_model_name=mmgpt
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/shikra
    model=shikra
    model_name=/remote-home/share/multimodal-models/shikra-7b-v1/

    store_model_name=shikra
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/lynx
    model=lynx
    model_name=models/interfaces/lynx/configs/LYNX.yaml

    store_model_name=lynx
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    conda activate /remote-home/mfdu/anaconda3/envs/pandagpt
    model=pandagpt
    model_name=pandagpt
    model_type=/remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/
    store_model_name=pandagpt
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/imagebind_LLM
    model=imagebindLLM
    model_name=imagebindLLM
    model_type=/remote-home/share/multimodal-models/imagebindllm_ckpts
    store_model_name=imagebindllm
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/llama_adapter_v2
    model=llama_adapterv2
    model_name=llama_adapterv2
    model_type=/remote-home/share/multimodal-models/pyllama_data
    store_model_name=llama_adapterv2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/mplugowl
    model=mplugowl
    model_name=mplugowl
    model_type=/remote-home/share/multimodal-models/mplug-owl-llama-7b/
    store_model_name=mplugowl
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/minigpt4
    model=minigpt4
    model_name=models/MiniGPT-4/eval_configs/minigpt4_eval.yaml

    store_model_name=minigpt4
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    conda activate /remote-home/mfdu/anaconda3/envs/llava
    model=llava
    model_name=/remote-home/share/multimodal-models/llava/LLaVA-7B-v0/

    store_model_name=llava_7B_v0
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=llava
    model_name=/remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
    model_type=/remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/
    store_model_name=llava_llama2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag



    conda activate /remote-home/mfdu/anaconda3/envs/bliva
    batch_size=1
    model=bliva
    model_name=bliva_vicuna

    store_model_name=bliva
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

 
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag
}
function test_all1(){
    conda activate /remote-home/mfdu/anaconda3/envs/blip2
    model=blip2
    model_name=blip2_t5
    model_type=pretrain_flant5xl
    store_model_name=blip2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 

        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=blip2
    model_name=blip2_t5_instruct
    model_type=flant5xl
    store_model_name=instructblip2_flant5
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 

        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=blip2
    model_name=blip2_vicuna_instruct
    model_type=vicuna7b
    store_model_name=instructblip2_vicuna
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 

        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag



    conda activate /remote-home/mfdu/anaconda3/envs/cheetah
    model=cheetor
    model_name=models/interfaces/Cheetah/eval_configs/cheetah_eval_vicuna.yaml

    store_model_name=cheetor_vicuna
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=cheetor
    model_name=models/interfaces/Cheetah/eval_configs/cheetah_eval_llama2.yaml

    store_model_name=cheetor_llama2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    conda activate /remote-home/mfdu/anaconda3/envs/mmgpt
    model=mmgpt
    model_name=Multimodal-GPT

    store_model_name=mmgpt
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/shikra
    model=shikra
    model_name=/remote-home/share/multimodal-models/shikra-7b-v1/

    store_model_name=shikra
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/lynx
    model=lynx
    model_name=models/interfaces/lynx/configs/LYNX.yaml

    store_model_name=lynx
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    conda activate /remote-home/mfdu/anaconda3/envs/pandagpt
    model=pandagpt
    model_name=pandagpt
    model_type=/remote-home/share/multimodal-models/pandagpt_pretrained_ckpt/
    store_model_name=pandagpt
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/imagebind_LLM
    model=imagebindLLM
    model_name=imagebindLLM
    model_type=/remote-home/share/multimodal-models/imagebindllm_ckpts
    store_model_name=imagebindllm
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/llama_adapter_v2
    model=llama_adapterv2
    model_name=llama_adapterv2
    model_type=/remote-home/share/multimodal-models/pyllama_data
    store_model_name=llama_adapterv2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/mplugowl
    model=mplugowl
    model_name=mplugowl
    model_type=/remote-home/share/multimodal-models/mplug-owl-llama-7b/
    store_model_name=mplugowl
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag


    conda activate /remote-home/mfdu/anaconda3/envs/minigpt4
    model=minigpt4
    model_name=models/MiniGPT-4/eval_configs/minigpt4_eval.yaml

    store_model_name=minigpt4
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    conda activate /remote-home/mfdu/anaconda3/envs/llava
    model=llava
    model_name=/remote-home/share/multimodal-models/llava/LLaVA-7B-v0/

    store_model_name=llava_7B_v0
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag

    model=llava
    model_name=/remote-home/share/multimodal-models/llava/llava-llama-2-7b-chat-lightning-lora-preview/ \
    model_type=/remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/
    store_model_name=llava_llama2
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}
        --model_type ${model_type}
        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag



    conda activate /remote-home/mfdu/anaconda3/envs/bliva
    batch_size=1
    model=bliva
    model_name=bliva_vicuna

    store_model_name=bliva
    output_dir=output/mfdu_output/rerun/${store_dataset}/${store_model_name}_${infer_method}_${formulation}
    flag=" --core_eval
        --model ${model}
        --model_name ${model_name}

        --in_context_sample
        --option_mark upper 
        --dataset_name ${dataset_name} 
        --dataset_config ${dataset_config} 
        --output_dir ${output_dir} 
        --infer_method ${infer_method} 
        --half_evaluation
        --per_gpu_eval_batch_size ${batch_size}
        --formulation ${formulation}
        --do_eval 
        --dataset_duplication ${duplication}
        "
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py $flag
}
# dataset_name=CLEVR
# dataset_config=build/configs/Spatial_clevr_val.yaml
# store_dataset=clevr
# test_all
# dataset_name=VSR
# dataset_config=build/configs/Spatial_vsr_val.yaml
# store_dataset=vsr
# test_all
# infer_method=generation
# test_all1

# dataset_name=MP3D
# dataset_config=build/configs/Spatial_mp3d_val.yaml
# store_dataset=mp3d
# test_all