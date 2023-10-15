# from .VisDial.visual_dialog_dataset import get_visdial
from .VisDial import get_visdial
from .MSCOCO import get_mscoco
from .COCO_text import get_cocotext
# #from .ImageTextMatching.image_text_matching_dataset import get_itm
# from .VQA.vqa_dataset import get_vqa
# from .VQA.kvqa_dataset import get_kvqa
# from .VQA.vqar_dataset import get_vqa_random
from .WikiHow import get_wikihow
from .SNLI_VE import get_snli_ve
from .NLVR import get_nlvr
from .TextCaps import get_textcaps
from .NoCaps import get_nocaps
from .Flickr30K import get_flickr30k
from .MEDIC import get_medic
from .RefCOCO import get_refcoco
from .CUTE80 import get_cute80
from .IC15 import get_ic15
from .IIIT5K import get_iiit5k
from .WordArt import get_wordart
from .textOCR import get_textocr
from .TDIUC import get_tdiuc
from .FUNSD import get_funsd
from .POIE import get_poie
from .SROIE import get_sroie
from .CLEVR import get_clevr
from .VSR import get_vsr
# from .MOCHEG.mcv_dataset import get_mcv
# # from .VizWiz.iqa_dataset import get_vizwiz
# from .VizWiz.vizwiz_dataset_hf import get_vizwiz
# # from .ImageNet1K.dataset import get_imagenet1k
# from .ImageNet1K.imagenet1k_dataset_hf import get_imagenet1k
# # from .CIFAR10.dataset import get_cifar10
# from .CIFAR10.cifar10_dataset_hf import get_cifar10
# # from .Pets37.dataset import get_pets37
# from .Pets37.pets37_dataset_hf import get_pets37
# # from .Flowers102.dataset import get_flowers102
# from .Flowers102.flowers102_dataset_hf import get_flowers102
# from .AOKVQA.qra_dataset import get_vqra
# from .AOKVQA.qar_dataset import get_vqar
# from .Winoground.caption_selection_dataset import get_caption_selection
# from .Winoground.matching_dataset import get_matching
from .MP3D import get_mp3d
from .OCR.ocr_dataset import get_ocr
from typing import Optional

def build_dataset(args, dataset_name:str, formulation:str, dataset_config:Optional[dict]=None, preprocessor=None):
    """
    Return the constructed dataset
    Parameters:
        args: the arguments claimed in the runner
        dataset_name: the dataset name to load
        formulation: the problem formulation
        dataset_config: the path to the config file, using the default path if not specified
        preprocessor: Optional, the model processor to process
    Return:
        dataset: the constructed dataset
    Usage:
        >>> from datasets import build_dataset
        >>> dataset = build_dataset("VisDial", "SingleChoice")
    """
    if args.hf == True:
        from .VizWiz.vizwiz_dataset_hf import get_vizwiz as get_vizwiz_hf
        #后面加dataset_hf.py的函数
    else:
        from .VizWiz.iqa_dataset import get_vizwiz
        #后面加dataset.py的函数


    if dataset_name == 'VQA':
        return get_vqa(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VQA_Random':
        return get_vqa_random(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'KVQA':
        return get_kvqa(args, dataset_config, formulation, preprocessor) 
    elif dataset_name == 'VisDial':
        return get_visdial(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MSCOCO':
        return get_mscoco(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'COCO_text':
        return get_cocotext(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'WikiHow':
        return get_wikihow(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'SNLI-VE':
        return get_snli_ve(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'NLVR':
        return get_nlvr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'TextCaps':
        return get_textcaps(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'NoCaps':
        return get_nocaps(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'Flickr30K':
        return get_flickr30k(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MEDIC':
        return get_medic(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'RefCOCO':
        return get_refcoco(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CUTE80':
        return get_cute80(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'IC15':
        return get_ic15(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'IIIT5K':
        return get_iiit5k(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'WordArt':
        return get_wordart(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'TDIUC':
        return get_tdiuc(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'TextOCR':
        return get_textocr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'FUNSD':
        return get_funsd(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'POIE':
        return get_poie(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'SROIE':
        return get_sroie(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CLEVR':
        return get_clevr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VSR':
        return get_vsr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MCV':
        return get_mcv(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VizWiz':
        return get_vizwiz(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'ImageNet-1K':
        return get_imagenet1k(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CIFAR10':
        return get_cifar10(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'Pets37':
        return get_pets37(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VQRA':
        return get_vqra(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VQAR':
        return get_vqar(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CaptionSelection':
        return get_caption_selection(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MP3D':
        return get_mp3d(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'Flowers102':
        return get_flowers102(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'OCR':
        return get_ocr(args, dataset_config, formulation, preprocessor)
    else:
        raise NotImplementedError('current dataset {} is not supported yet.'.format(dataset_name))

def load_reform_dataset(dataset_name:str, 
                 formulation:str, 
                 dataset_config:Optional[dict]=None,
                 inference_method:Optional[str]='generation',
                 in_context_sample:Optional[bool]=True,
                 random_instruct:Optional[bool]=True,
                 data_duplication:Optional[int]=5,
                 shuffle_options:Optional[bool]=True,
                 load_from_hf:Optional[bool]=True, 
                 preprocessor=None):
    """
    Return the constructed dataset
    Parameters:
        dataset_name: the dataset name to load.
        formulation: the problem formulation.
        dataset_config: the path to the config file, using the default path if not specified.
        infer_method: inference method, influencing the data format and instructions, should be "generation" or "likelihood".
        in_context_sample: whether to include an in-context sample, defalut to True.
        rando_instruct: use different instruction for the same sample, default to True.
        data_duplication: the number of multiple tests for the same sample, default to 5.
        shuffle_options: shuffle the options for the same sample, default to True.
        load_from_hf: whether to load from huggingface, load from local if set to False.
        preprocessor: Optional, the model processor to process.
    Return:
        dataset: the constructed dataset
    Usage:
        >>> from datasets import build_dataset
        >>> dataset = build_dataset("VisDial", "SingleChoice")
    """

    # prepare the argument namespace for input
    from argparse import Namespace
    args = Namespace()

    # set the important arguments to the namespace
    args.hf = load_from_hf

    assert inference_method in ['generation', 'likelihood'], "the inference method should be 'generation' or 'likelihood'"
    args.infer_method = inference_method

    args.in_context_sample = in_context_sample

    # randomness-related parameters
    args.dataset_duplication = data_duplication
    args.random_instruct = random_instruct
    args.shuffle_options = shuffle_options

    # set the default arguments
    args.capitalize = True
    args.dataset_subsample = None
    args.options_in_history = True
    args.online_multi_round = True

    
    if dataset_name == 'VQA':
        return get_vqa(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VQA_Random':
        return get_vqa_random(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'KVQA':
        return get_kvqa(args, dataset_config, formulation, preprocessor) 
    elif dataset_name == 'VisDial':
        return get_visdial(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MSCOCO':
        return get_mscoco(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'COCO_text':
        return get_cocotext(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'WikiHow':
        return get_wikihow(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'SNLI-VE':
        return get_snli_ve(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'NLVR':
        return get_nlvr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'TextCaps':
        return get_textcaps(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'NoCaps':
        return get_nocaps(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'Flickr30K':
        return get_flickr30k(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MEDIC':
        return get_medic(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'RefCOCO':
        return get_refcoco(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CUTE80':
        return get_cute80(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'IC15':
        return get_ic15(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'IIIT5K':
        return get_iiit5k(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'WordArt':
        return get_wordart(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'TDIUC':
        return get_tdiuc(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'TextOCR':
        return get_textocr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'FUNSD':
        return get_funsd(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'POIE':
        return get_poie(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'SROIE':
        return get_sroie(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CLEVR':
        return get_clevr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VSR':
        return get_vsr(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MCV':
        return get_mcv(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VizWiz':
        return get_vizwiz(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'ImageNet-1K':
        return get_imagenet1k(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CIFAR10':
        return get_cifar10(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'Pets37':
        return get_pets37(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VQRA':
        return get_vqra(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'VQAR':
        return get_vqar(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'CaptionSelection':
        return get_caption_selection(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'Matching':
        return get_matching(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'MP3D':
        return get_mp3d(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'Flowers102':
        return get_flowers102(args, dataset_config, formulation, preprocessor)
    elif dataset_name == 'OCR':
        return get_ocr(args, dataset_config, formulation, preprocessor)
    else:
        raise NotImplementedError('current dataset {} is not supported yet.'.format(dataset_name))