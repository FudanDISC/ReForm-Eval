##################### total setting=2,4,6,7
infer_method=likelihood
formulation=SingleChoice
model=minigpt4
model_name=models/MiniGPT-4/eval_configs/minigpt4_eval.yaml
# model_type=vicuna7b
duplication=10
batch_size=4

## dataset setting

##### mci
dataset_name=MSCOCO
dataset_config=datasets/configs/MulticlassIdentification_val.yaml
output_dir=output/mci_output/${model}_${model_name}_${infer_method}_${formulation}
# output_dir=output/mci_output/${model}_${model_name}_$1{model_type}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag


##### goi
dataset_name=MSCOCO
dataset_config=datasets/configs/GroundedObjIdentification_val.yaml
output_dir=output/goi_output/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag

##### MOS
dataset_name=MSCOCO
dataset_config=datasets/configs/MissingObjectSelection_val.yaml
output_dir=output/mos_output/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag

##### TL
dataset_name=COCO_text
dataset_config=datasets/configs/TextLegibility_val.yaml
output_dir=output/tl_output/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag

##### TTC
dataset_name=COCO_text
dataset_config=datasets/configs/TextTypeClassification_val.yaml
output_dir=output/ttc_output/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
###################################Spatial
##### CLEVR
dataset_name=CLEVR
dataset_config=datasets/configs/Spatial_clevr_val.yaml
output_dir=output/spatial_output/clevr/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag

##### VSR
dataset_name=VSR
dataset_config=datasets/configs/Spatial_vsr_val.yaml
output_dir=output/spatial_output/vsr/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag

##### MP3D
dataset_name=MP3D
dataset_config=datasets/configs/Spatial_mp3d_val.yaml
output_dir=output/spatial_output/mp3d/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag

function OCR(){
####################### OCR
infer_method=generation
formulation=OCROpenEnded
### cocotext
dataset_name=COCO_text
dataset_config=datasets/configs/OCR_cocotext_val.yaml
output_dir=output/ocr_output/cocotext/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### cute80
dataset_name=CUTE80
dataset_config=datasets/configs/OCR_cute80_val.yaml
output_dir=output/ocr_output/cute80/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### ic15
dataset_name=IC15
dataset_config=datasets/configs/OCR_ic15_val.yaml
output_dir=output/ocr_output/ic15/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### iiit5k
dataset_name=IIIT5k
dataset_config=datasets/configs/OCR_iiit5k_val.yaml
output_dir=output/ocr_output/iiit5k/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### textocr
dataset_name=TextOCR
dataset_config=datasets/configs/OCR_textocr_val.yaml
output_dir=output/ocr_output/textocr/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### wordart
dataset_name=WordArt
dataset_config=datasets/configs/OCR_wordart_val.yaml
output_dir=output/ocr_output/wordart/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
###################Ground OCR
### coco text
dataset_name=COCO_text
dataset_config=datasets/configs/GroundOCR_cocotext_val.yaml
output_dir=output/gocr_output/cocotext/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### ic15
dataset_name=IC15
dataset_config=datasets/configs/GroundOCR_ic15_val.yaml
output_dir=output/gocr_output/ic15/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### textocr
dataset_name=TextOCR
dataset_config=datasets/configs/GroundOCR_textocr_val.yaml
output_dir=output/gocr_output/textocr/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
################################ KIE
formulation=KIEOpenEnded
### funsd
dataset_name=FUNSD
dataset_config=datasets/configs/KIE_funsd_val.yaml
output_dir=output/kie_output/funsd/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### funsd
dataset_name=POIE
dataset_config=datasets/configs/KIE_poie_val.yaml
output_dir=output/kie_output/poie/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
### funsd
dataset_name=SROIE
dataset_config=datasets/configs/KIE_sroie_val.yaml
output_dir=output/kie_output/sroie/${model}_${model_name}_${infer_method}_${formulation}
flag="
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
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
}

OCR

