def get_model(model_name, model_config=None):
    """
    Return the wrapped model interface for evaluation
    
    Parameters:
        model_name: the name of the model family
        model_config: {
                    "model_name": the sub-name of the model family
                    "model_type": the type setup for 
                    }
    Return:
        model: the constructed model interface
        processor: the conversation processor
    Usage:
        >>> from models import get_model
        >>> model = get_model("blip2", {"model_name": "blip2_t5", "model_type": "pretrain_flant5xl", "half": False, "device": "cuda:1"})
    """
    if model_name == 'blip2':
        from .interfaces.blip2 import get_blip2
        return get_blip2(model_config)
    elif model_name == 'llava':
        from .interfaces.llava_interface import get_llava
        return get_llava(model_config)
    elif model_name == 'minigpt4':
        from .interfaces.minigpt4_interface import get_minigpt4
        return get_minigpt4(model_config)
    elif model_name == 'mmgpt':
        from .interfaces.mmgpt_interface import get_mmgpt
        return get_mmgpt(model_config)
    elif model_name == 'mplugowl':
        from .interfaces.mplug_owl_interface import get_mPLUG_Owl
        return get_mPLUG_Owl(model_config)
    elif model_name == 'otter':
        from .interfaces.otter_interface import get_otter
        return get_otter(model_config)
    elif model_name == 'shikra':
        from .interfaces.shikra_interface import get_shikra
        return get_shikra(model_config)
    elif model_name == 'lynx':
        from .interfaces.lynx_interface import get_lynx
        return get_lynx(model_config)
    elif model_name == 'cheetor':
        from .interfaces.cheetor_interface import get_cheetor
        return get_cheetor(model_config)
    elif model_name == 'bliva':
        from .interfaces.bliva_interface import get_bliva
        return get_bliva(model_config)
    elif model_name == 'imagebindLLM':
        from .interfaces.imagebindLLM_interface import get_imagebindLLM
        return get_imagebindLLM(model_config)
    elif model_name == 'llama_adapterv2':
        from .interfaces.llama_adapterv2_interface import get_llama_adapterv2
        return get_llama_adapterv2(model_config)
    elif model_name == 'pandagpt':        
        from .interfaces.pandagpt_interface import get_pandagpt
        return get_pandagpt(model_config)
    elif model_name == 'visualglm':
        from .interfaces.visualglm_interface import get_visualglm
        return get_visualglm(model_config)
    elif model_name == 'lavin':
        from .interfaces.lavin_interface import get_lavin
        return get_lavin(model_config)
    elif model_name == 'kosmos2':
        from .interfaces.kosmos2_interface import get_kosmos2
        return get_kosmos2(model_config)
    else:
        raise ValueError('the target model is not supported yet!')
    
if __name__=='__main__':
    get_model(model_name='blip2')
