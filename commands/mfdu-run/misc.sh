source /remote-home/mfdu/anaconda3/etc/profile.d/conda.sh
conda activate /remote-home/mfdu/anaconda3/envs/imagebind_LLM
infer_method=generation
formulation=OCROpenEnded
duplication=5
batch_size=1
dataset_name=COCO_text
dataset_config=datasets/configs/GroundOCR_cocotext_val.yaml
model=imagebindLLM
model_name=imagebindLLM
model_type=/remote-home/share/multimodal-models/imagebindllm_ckpts
store_model_name=imagebindllm

output_dir=output/mfdu_output/gocr_output2/cocotext/${store_model_name}_${infer_method}_${formulation}
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