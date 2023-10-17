source /root/anaconda3/etc/profile.d/conda.sh

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

#### Pets37
dm="Pets37"
dc="build/configs/ImageClassification_pets37_val.yaml"
conda activate bliva
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir tmp_rest_output/bliva/pets37/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model bliva  --model_name bliva_vicuna \
    --dataset_name $dm --output_dir tmp_rest_output/bliva/pets37/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

conda activate lynx
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir tmp_rest_output/lynx/pets37/likelihood \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method likelihood --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 run_eval.py \
    --model lynx  --model_name models/interfaces/lynx/configs/LYNX.yaml \
    --dataset_name $dm --output_dir tmp_rest_output/lynx/pets37/generation \
    --per_gpu_eval_batch_size 1 --formulation SingleChoice --dataset_duplication 5 --in_context_sample \
    --infer_method generation --do_eval --option_mark upper \
    --dataset_config $dc --half_evaluation \