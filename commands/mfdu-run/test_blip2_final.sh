##################### total setting=2,4,6,7
function blip2_likelihood(){
    infer_method=likelihood
    formulation=SingleChoice
    model=blip2
    model_name=blip2_t5
    model_type=pretrain_flant5xl
    store_model_name=blip2
    duplication=5
    batch_size=4

    ## dataset setting

    function MCI(){
        ##### mci
        dataset_name=MSCOCO
        dataset_config=build/configs/MulticlassIdentification_val.yaml
        output_dir=output/test_20231017/test/mci_output/${store_model_name}_${infer_method}_${formulation}
        #output_dir=output/test_20231017/test/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    function GOI(){
        ##### goi
        dataset_name=MSCOCO
        dataset_config=build/configs/GroundedObjIdentification_val.yaml
        output_dir=output/test_20231017/test/goi_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MOS
    function MOS(){
        dataset_name=MSCOCO
        dataset_config=build/configs/MissingObjectSelection_val.yaml
        output_dir=output/test_20231017/test/mos_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TL
    function TL(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextLegibility_val.yaml
        output_dir=output/test_20231017/test/tl_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TTC
    function TTC(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextTypeClassification_val.yaml
        output_dir=output/test_20231017/test/ttc_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ###################################Spatial
    ##### CLEVR
    function CLEVR(){
        dataset_name=CLEVR
        dataset_config=build/configs/Spatial_clevr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/clevr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### VSR
    function VSR(){
        dataset_name=VSR
        dataset_config=build/configs/Spatial_vsr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/vsr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MP3D
    function MP3D(){
        dataset_name=MP3D
        dataset_config=build/configs/Spatial_mp3d_val.yaml
        output_dir=output/test_20231017/test/spatial_output/mp3d/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    ####################### OCR
    function OCR(){
        infer_method=generation
        formulation=OCROpenEnded
        ### cocotext
        dataset_name=COCO_text
        dataset_config=build/configs/OCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### cute80
        dataset_name=CUTE80
        dataset_config=build/configs/OCR_cute80_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cute80/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/OCR_ic15_val.yaml
        output_dir=output/test_20231017/test/ocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### iiit5k
        dataset_name=IIIT5K
        dataset_config=build/configs/OCR_iiit5k_val.yaml
        output_dir=output/test_20231017/test/ocr_output/iiit5k/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/OCR_textocr_val.yaml
        output_dir=output/test_20231017/test/ocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### wordart
        dataset_name=WordArt
        dataset_config=build/configs/OCR_wordart_val.yaml
        output_dir=output/test_20231017/test/ocr_output/wordart/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ###################Ground OCR
        ### coco text
        dataset_name=COCO_text
        dataset_config=build/configs/GroundOCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/gocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/GroundOCR_ic15_val.yaml
        output_dir=output/test_20231017/test/gocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/GroundOCR_textocr_val.yaml
        output_dir=output/test_20231017/test/gocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ################################ KIE
        formulation=KIEOpenEnded
        ### funsd
        dataset_name=FUNSD
        dataset_config=build/configs/KIE_funsd_val.yaml
        output_dir=output/test_20231017/test/kie_output/funsd/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=POIE
        dataset_config=build/configs/KIE_poie_val.yaml
        output_dir=output/test_20231017/test/kie_output/poie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=SROIE
        dataset_config=build/configs/KIE_sroie_val.yaml
        output_dir=output/test_20231017/test/kie_output/sroie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    MCI
    GOI
    MOS
    CLEVR
    VSR
    MP3D

}
function blip2_generation(){
    infer_method=generation
    formulation=SingleChoice
    model=blip2
    model_name=blip2_t5
    model_type=pretrain_flant5xl
    store_model_name=blip2
    duplication=5
    batch_size=4

    ## dataset setting

    function MCI(){
        ##### mci
        dataset_name=MSCOCO
        dataset_config=build/configs/MulticlassIdentification_val.yaml
        output_dir=output/test_20231017/test/mci_output/${store_model_name}_${infer_method}_${formulation}
        #output_dir=output/test_20231017/test/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" 
            --model ${model}
            --model_name ${model_name}
            --model_type ${model_type}
                   
            --option_mark upper 
            --dataset_name ${dataset_name} 
            --dataset_config ${dataset_config} 
            --output_dir ${output_dir} 
            --infer_method ${infer_method} 
            --shuffle_options
            --per_gpu_eval_batch_size ${batch_size}
            --formulation ${formulation}
            --do_eval 
            --dataset_duplication ${duplication}
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    function GOI(){
        ##### goi
        dataset_name=MSCOCO
        dataset_config=build/configs/GroundedObjIdentification_val.yaml
        output_dir=output/test_20231017/test/goi_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MOS
    function MOS(){
        dataset_name=MSCOCO
        dataset_config=build/configs/MissingObjectSelection_val.yaml
        output_dir=output/test_20231017/test/mos_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TL
    function TL(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextLegibility_val.yaml
        output_dir=output/test_20231017/test/tl_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TTC
    function TTC(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextTypeClassification_val.yaml
        output_dir=output/test_20231017/test/ttc_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ###################################Spatial
    ##### CLEVR
    function CLEVR(){
        dataset_name=CLEVR
        dataset_config=build/configs/Spatial_clevr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/clevr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
            --model ${model}
            --model_name ${model_name}
            --model_type ${model_type}
                   
            --option_mark upper 
            --dataset_name ${dataset_name} 
            --dataset_config ${dataset_config} 
            --output_dir ${output_dir} 
            --infer_method ${infer_method} 
            --shuffle_options
            --per_gpu_eval_batch_size ${batch_size}
            --formulation ${formulation}
            --do_eval 
            --dataset_duplication ${duplication}
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### VSR
    function VSR(){
        dataset_name=VSR
        dataset_config=build/configs/Spatial_vsr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/vsr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
            --model ${model}
            --model_name ${model_name}
            --model_type ${model_type}
                   
            --option_mark upper 
            --dataset_name ${dataset_name} 
            --dataset_config ${dataset_config} 
            --output_dir ${output_dir} 
            --infer_method ${infer_method} 
            --shuffle_options
            --per_gpu_eval_batch_size ${batch_size}
            --formulation ${formulation}
            --do_eval 
            --dataset_duplication ${duplication}
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MP3D
    function MP3D(){
        dataset_name=MP3D
        dataset_config=build/configs/Spatial_mp3d_val.yaml
        output_dir=output/test_20231017/test/spatial_output/mp3d/${store_model_name}_${infer_method}_${formulation}
        flag=" 
            --model ${model}
            --model_name ${model_name}
            --model_type ${model_type}
                   
            --option_mark upper 
            --dataset_name ${dataset_name} 
            --dataset_config ${dataset_config} 
            --output_dir ${output_dir} 
            --infer_method ${infer_method} 
            --shuffle_options
            --per_gpu_eval_batch_size ${batch_size}
            --formulation ${formulation}
            --do_eval 
            --dataset_duplication ${duplication}
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    ####################### OCR
    function OCR(){
        infer_method=generation
        formulation=OCROpenEnded
        ### cocotext
        dataset_name=COCO_text
        dataset_config=build/configs/OCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### cute80
        dataset_name=CUTE80
        dataset_config=build/configs/OCR_cute80_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cute80/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/OCR_ic15_val.yaml
        output_dir=output/test_20231017/test/ocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### iiit5k
        dataset_name=IIIT5K
        dataset_config=build/configs/OCR_iiit5k_val.yaml
        output_dir=output/test_20231017/test/ocr_output/iiit5k/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/OCR_textocr_val.yaml
        output_dir=output/test_20231017/test/ocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### wordart
        dataset_name=WordArt
        dataset_config=build/configs/OCR_wordart_val.yaml
        output_dir=output/test_20231017/test/ocr_output/wordart/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ###################Ground OCR
        ### coco text
        dataset_name=COCO_text
        dataset_config=build/configs/GroundOCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/gocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/GroundOCR_ic15_val.yaml
        output_dir=output/test_20231017/test/gocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/GroundOCR_textocr_val.yaml
        output_dir=output/test_20231017/test/gocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ################################ KIE
        formulation=KIEOpenEnded
        ### funsd
        dataset_name=FUNSD
        dataset_config=build/configs/KIE_funsd_val.yaml
        output_dir=output/test_20231017/test/kie_output/funsd/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=POIE
        dataset_config=build/configs/KIE_poie_val.yaml
        output_dir=output/test_20231017/test/kie_output/poie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=SROIE
        dataset_config=build/configs/KIE_sroie_val.yaml
        output_dir=output/test_20231017/test/kie_output/sroie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    # MCI
    # GOI
    # MOS
    # CLEVR
    # VSR
    # MP3D
    # OCR
        formulation=OCROpenEnded
        infer_method=generation
        dataset_name=COCO_text
        dataset_config=build/configs/GroundOCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/gocr_output/cocotext1/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
}

function instructblip2_flant5_likelihood(){
    infer_method=likelihood
    formulation=SingleChoice
    model=blip2
    model_name=blip2_t5_instruct
    model_type=flant5xl
    store_model_name=instructblip2_flant5
    duplication=5
    batch_size=4

    ## dataset setting

    function MCI(){
        ##### mci
        dataset_name=MSCOCO
        dataset_config=build/configs/MulticlassIdentification_val.yaml
        output_dir=output/test_20231017/test/mci_output/${store_model_name}_${infer_method}_${formulation}
        #output_dir=output/test_20231017/test/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    function GOI(){
        ##### goi
        dataset_name=MSCOCO
        dataset_config=build/configs/GroundedObjIdentification_val.yaml
        output_dir=output/test_20231017/test/goi_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MOS
    function MOS(){
        dataset_name=MSCOCO
        dataset_config=build/configs/MissingObjectSelection_val.yaml
        output_dir=output/test_20231017/test/mos_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TL
    function TL(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextLegibility_val.yaml
        output_dir=output/test_20231017/test/tl_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TTC
    function TTC(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextTypeClassification_val.yaml
        output_dir=output/test_20231017/test/ttc_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ###################################Spatial
    ##### CLEVR
    function CLEVR(){
        dataset_name=CLEVR
        dataset_config=build/configs/Spatial_clevr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/clevr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### VSR
    function VSR(){
        dataset_name=VSR
        dataset_config=build/configs/Spatial_vsr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/vsr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MP3D
    function MP3D(){
        dataset_name=MP3D
        dataset_config=build/configs/Spatial_mp3d_val.yaml
        output_dir=output/test_20231017/test/spatial_output/mp3d/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    ####################### OCR
    function OCR(){
        infer_method=generation
        formulation=OCROpenEnded
        ### cocotext
        dataset_name=COCO_text
        dataset_config=build/configs/OCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### cute80
        dataset_name=CUTE80
        dataset_config=build/configs/OCR_cute80_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cute80/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/OCR_ic15_val.yaml
        output_dir=output/test_20231017/test/ocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### iiit5k
        dataset_name=IIIT5K
        dataset_config=build/configs/OCR_iiit5k_val.yaml
        output_dir=output/test_20231017/test/ocr_output/iiit5k/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/OCR_textocr_val.yaml
        output_dir=output/test_20231017/test/ocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### wordart
        dataset_name=WordArt
        dataset_config=build/configs/OCR_wordart_val.yaml
        output_dir=output/test_20231017/test/ocr_output/wordart/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ###################Ground OCR
        ### coco text
        dataset_name=COCO_text
        dataset_config=build/configs/GroundOCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/gocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/GroundOCR_ic15_val.yaml
        output_dir=output/test_20231017/test/gocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/GroundOCR_textocr_val.yaml
        output_dir=output/test_20231017/test/gocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ################################ KIE
        formulation=KIEOpenEnded
        ### funsd
        dataset_name=FUNSD
        dataset_config=build/configs/KIE_funsd_val.yaml
        output_dir=output/test_20231017/test/kie_output/funsd/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=POIE
        dataset_config=build/configs/KIE_poie_val.yaml
        output_dir=output/test_20231017/test/kie_output/poie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=SROIE
        dataset_config=build/configs/KIE_sroie_val.yaml
        output_dir=output/test_20231017/test/kie_output/sroie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    MCI
    GOI
    # MOS
    # TL
    # TTC
    # CLEVR
    # VSR
    # MP3D
    # OCR
}
function instructblip2_flant5_generation(){
    infer_method=generation
    formulation=SingleChoice
    model=blip2
    model_name=blip2_t5_instruct
    model_type=flant5xl
    store_model_name=instructblip2_flant5
    duplication=5
    batch_size=4

    ## dataset setting

    function MCI(){
        ##### mci
        dataset_name=MSCOCO
        dataset_config=build/configs/MulticlassIdentification_val.yaml
        output_dir=output/test_20231017/test/mci_output/${store_model_name}_${infer_method}_${formulation}
        #output_dir=output/test_20231017/test/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    function GOI(){
        ##### goi
        dataset_name=MSCOCO
        dataset_config=build/configs/GroundedObjIdentification_val.yaml
        output_dir=output/test_20231017/test/goi_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MOS
    function MOS(){
        dataset_name=MSCOCO
        dataset_config=build/configs/MissingObjectSelection_val.yaml
        output_dir=output/test_20231017/test/mos_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TL
    function TL(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextLegibility_val.yaml
        output_dir=output/test_20231017/test/tl_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TTC
    function TTC(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextTypeClassification_val.yaml
        output_dir=output/test_20231017/test/ttc_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ###################################Spatial
    ##### CLEVR
    function CLEVR(){
        dataset_name=CLEVR
        dataset_config=build/configs/Spatial_clevr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/clevr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### VSR
    function VSR(){
        dataset_name=VSR
        dataset_config=build/configs/Spatial_vsr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/vsr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MP3D
    function MP3D(){
        dataset_name=MP3D
        dataset_config=build/configs/Spatial_mp3d_val.yaml
        output_dir=output/test_20231017/test/spatial_output/mp3d/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    ####################### OCR
    function OCR(){
        infer_method=generation
        formulation=OCROpenEnded
        ### cocotext
        dataset_name=COCO_text
        dataset_config=build/configs/OCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### cute80
        dataset_name=CUTE80
        dataset_config=build/configs/OCR_cute80_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cute80/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/OCR_ic15_val.yaml
        output_dir=output/test_20231017/test/ocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### iiit5k
        dataset_name=IIIT5K
        dataset_config=build/configs/OCR_iiit5k_val.yaml
        output_dir=output/test_20231017/test/ocr_output/iiit5k/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/OCR_textocr_val.yaml
        output_dir=output/test_20231017/test/ocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### wordart
        dataset_name=WordArt
        dataset_config=build/configs/OCR_wordart_val.yaml
        output_dir=output/test_20231017/test/ocr_output/wordart/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ###################Ground OCR
        ### coco text
        dataset_name=COCO_text
        dataset_config=build/configs/GroundOCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/gocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/GroundOCR_ic15_val.yaml
        output_dir=output/test_20231017/test/gocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/GroundOCR_textocr_val.yaml
        output_dir=output/test_20231017/test/gocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ################################ KIE
        formulation=KIEOpenEnded
        ### funsd
        dataset_name=FUNSD
        dataset_config=build/configs/KIE_funsd_val.yaml
        output_dir=output/test_20231017/test/kie_output/funsd/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=POIE
        dataset_config=build/configs/KIE_poie_val.yaml
        output_dir=output/test_20231017/test/kie_output/poie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=SROIE
        dataset_config=build/configs/KIE_sroie_val.yaml
        output_dir=output/test_20231017/test/kie_output/sroie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    MCI
    GOI
    MOS
    CLEVR
    VSR
    MP3D
}

function instructblip2_vicuna_likelihood(){
    infer_method=likelihood
    formulation=SingleChoice
    model=blip2
    model_name=blip2_vicuna_instruct
    model_type=vicuna7b
    store_model_name=instructblip2_vicuna
    duplication=5
    batch_size=4

    ## dataset setting

    function MCI(){
        ##### mci
        dataset_name=MSCOCO
        dataset_config=build/configs/MulticlassIdentification_val.yaml
        output_dir=output/test_20231017/test/mci_output/${store_model_name}_${infer_method}_${formulation}
        #output_dir=output/test_20231017/test/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    function GOI(){
        ##### goi
        dataset_name=MSCOCO
        dataset_config=build/configs/GroundedObjIdentification_val.yaml
        output_dir=output/test_20231017/test/goi_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MOS
    function MOS(){
        dataset_name=MSCOCO
        dataset_config=build/configs/MissingObjectSelection_val.yaml
        output_dir=output/test_20231017/test/mos_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TL
    function TL(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextLegibility_val.yaml
        output_dir=output/test_20231017/test/tl_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TTC
    function TTC(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextTypeClassification_val.yaml
        output_dir=output/test_20231017/test/ttc_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ###################################Spatial
    ##### CLEVR
    function CLEVR(){
        dataset_name=CLEVR
        dataset_config=build/configs/Spatial_clevr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/clevr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### VSR
    function VSR(){
        dataset_name=VSR
        dataset_config=build/configs/Spatial_vsr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/vsr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MP3D
    function MP3D(){
        dataset_name=MP3D
        dataset_config=build/configs/Spatial_mp3d_val.yaml
        output_dir=output/test_20231017/test/spatial_output/mp3d/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    ####################### OCR
    function OCR(){
        infer_method=generation
        formulation=OCROpenEnded
        ### cocotext
        dataset_name=COCO_text
        dataset_config=build/configs/OCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### cute80
        dataset_name=CUTE80
        dataset_config=build/configs/OCR_cute80_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cute80/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/OCR_ic15_val.yaml
        output_dir=output/test_20231017/test/ocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### iiit5k
        dataset_name=IIIT5K
        dataset_config=build/configs/OCR_iiit5k_val.yaml
        output_dir=output/test_20231017/test/ocr_output/iiit5k/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/OCR_textocr_val.yaml
        output_dir=output/test_20231017/test/ocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### wordart
        dataset_name=WordArt
        dataset_config=build/configs/OCR_wordart_val.yaml
        output_dir=output/test_20231017/test/ocr_output/wordart/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ###################Ground OCR
        ### coco text
        dataset_name=COCO_text
        dataset_config=build/configs/GroundOCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/gocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/GroundOCR_ic15_val.yaml
        output_dir=output/test_20231017/test/gocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/GroundOCR_textocr_val.yaml
        output_dir=output/test_20231017/test/gocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ################################ KIE
        formulation=KIEOpenEnded
        ### funsd
        dataset_name=FUNSD
        dataset_config=build/configs/KIE_funsd_val.yaml
        output_dir=output/test_20231017/test/kie_output/funsd/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=POIE
        dataset_config=build/configs/KIE_poie_val.yaml
        output_dir=output/test_20231017/test/kie_output/poie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=SROIE
        dataset_config=build/configs/KIE_sroie_val.yaml
        output_dir=output/test_20231017/test/kie_output/sroie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    MCI
    GOI
    MOS
    TL
    TTC
    CLEVR
    VSR
    MP3D
    OCR
}
function instructblip2_vicuna_generation(){
    infer_method=generation
    formulation=SingleChoice
    model=blip2
    model_name=blip2_vicuna_instruct
    model_type=vicuna7b
    store_model_name=instructblip2_vicuna
    duplication=5
    batch_size=4

    ## dataset setting

    function MCI(){
        ##### mci
        dataset_name=MSCOCO
        dataset_config=build/configs/MulticlassIdentification_val.yaml
        output_dir=output/test_20231017/test/mci_output/${store_model_name}_${infer_method}_${formulation}
        #output_dir=output/test_20231017/test/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    function GOI(){
        ##### goi
        dataset_name=MSCOCO
        dataset_config=build/configs/GroundedObjIdentification_val.yaml
        output_dir=output/test_20231017/test/goi_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MOS
    function MOS(){
        dataset_name=MSCOCO
        dataset_config=build/configs/MissingObjectSelection_val.yaml
        output_dir=output/test_20231017/test/mos_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TL
    function TL(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextLegibility_val.yaml
        output_dir=output/test_20231017/test/tl_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### TTC
    function TTC(){
        dataset_name=COCO_text
        dataset_config=build/configs/TextTypeClassification_val.yaml
        output_dir=output/test_20231017/test/ttc_output/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ###################################Spatial
    ##### CLEVR
    function CLEVR(){
        dataset_name=CLEVR
        dataset_config=build/configs/Spatial_clevr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/clevr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### VSR
    function VSR(){
        dataset_name=VSR
        dataset_config=build/configs/Spatial_vsr_val.yaml
        output_dir=output/test_20231017/test/spatial_output/vsr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }
    ##### MP3D
    function MP3D(){
        dataset_name=MP3D
        dataset_config=build/configs/Spatial_mp3d_val.yaml
        output_dir=output/test_20231017/test/spatial_output/mp3d/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    ####################### OCR
    function OCR(){
        infer_method=generation
        formulation=OCROpenEnded
        ### cocotext
        dataset_name=COCO_text
        dataset_config=build/configs/OCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### cute80
        dataset_name=CUTE80
        dataset_config=build/configs/OCR_cute80_val.yaml
        output_dir=output/test_20231017/test/ocr_output/cute80/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/OCR_ic15_val.yaml
        output_dir=output/test_20231017/test/ocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### iiit5k
        dataset_name=IIIT5K
        dataset_config=build/configs/OCR_iiit5k_val.yaml
        output_dir=output/test_20231017/test/ocr_output/iiit5k/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/OCR_textocr_val.yaml
        output_dir=output/test_20231017/test/ocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### wordart
        dataset_name=WordArt
        dataset_config=build/configs/OCR_wordart_val.yaml
        output_dir=output/test_20231017/test/ocr_output/wordart/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ###################Ground OCR
        ### coco text
        dataset_name=COCO_text
        dataset_config=build/configs/GroundOCR_cocotext_val.yaml
        output_dir=output/test_20231017/test/gocr_output/cocotext/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### ic15
        dataset_name=IC15
        dataset_config=build/configs/GroundOCR_ic15_val.yaml
        output_dir=output/test_20231017/test/gocr_output/ic15/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### textocr
        dataset_name=TextOCR
        dataset_config=build/configs/GroundOCR_textocr_val.yaml
        output_dir=output/test_20231017/test/gocr_output/textocr/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ################################ KIE
        formulation=KIEOpenEnded
        ### funsd
        dataset_name=FUNSD
        dataset_config=build/configs/KIE_funsd_val.yaml
        output_dir=output/test_20231017/test/kie_output/funsd/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=POIE
        dataset_config=build/configs/KIE_poie_val.yaml
        output_dir=output/test_20231017/test/kie_output/poie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
        ### funsd
        dataset_name=SROIE
        dataset_config=build/configs/KIE_sroie_val.yaml
        output_dir=output/test_20231017/test/kie_output/sroie/${store_model_name}_${infer_method}_${formulation}
        flag=" 
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
            --offline_hf"
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_eval.py $flag
    }

    MCI
    GOI
    MOS
    TL
    TTC
    CLEVR
    VSR
    MP3D
}


# blip2_likelihood
blip2_generation
# instructblip2_flant5_likelihood
# instructblip2_flant5_generation
# instructblip2_vicuna_likelihood
# instructblip2_vicuna_generation

