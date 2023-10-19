dms=("CIFAR10" "Flowers102" "ImageNet-1K" "MSCOCO" "Pets37" "TDIUC" "TDIUC" "TDIUC" "TDIUC" "TDIUC" "TDIUC" "TDIUC" "VizWiz" "VizWiz")
dcs=("build/configs/ImageClassification_cifar10_val.yaml" "build/configs/ImageClassification_flowers102_val.yaml" \
    "build/configs/ImageClassification_imagenet1k_val.yaml" "build/configs/ObjectCounting_mscoco_val.yaml" \
    "build/configs/ImageClassification_pets37_val.yaml" "build/configs/TDIUC_color.yaml" "build/configs/TDIUC_counting.yaml" \
    "build/configs/TDIUC_detection.yaml" "build/configs/TDIUC_position.yaml" "build/configs/TDIUC_scene.yaml" \
    "build/configs/TDIUC_utility.yaml" "build/configs/TDIUC_sport.yaml" "build/configs/ImageQuality_vizwiz_yesNo_val.yaml" \
    "build/configs/ImageQuality_vizwiz_singleChoice_val.yaml" \
    )
saved_dir=("CIFAR10" "Flowers102" "ImageNet-1K" "MSCOCO" "Pets37" "TDIUC_color" "TDIUC_counting" "TDIUC_detection" "TDIUC_position" "TDIUC_scene" "TDIUC_utility" \
    "TDIUC_sport" "VizWiz_yesno" "VizWiz_singleChoice")

# dms=("MSCOCO" "VizWiz")
# dcs=("build/configs/ObjectCounting_mscoco_val.yaml" \
#     "build/configs/ImageQuality_vizwiz_singleChoice_val.yaml" \
#     )
# saved_dir=("MSCOCO_oc_from_raw" "VizWiz_singleChoice_from_raw")

length=${#dms[@]}
for ((i=0; i<$length; i++)); do
    # echo "${dms[$i]} , ${dcs[$i]}"
    saved_path="output/test_20231017/${saved_dir[$i]}"
    dataset_name="${dms[$i]}"
    dataset_config="${dcs[$i]}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_eval.py \
        --model blip2 --model_name blip2_t5 --model_type pretrain_flant5xl \
        --dataset_name ${dataset_name} --output_dir ${saved_path} \
        --per_gpu_eval_batch_size 4 --formulation SingleChoice --dataset_duplication 5 \
        --infer_method likelihood --do_eval --option_mark upper \
        --dataset_config ${dataset_config} --offline_hf
done