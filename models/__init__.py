def get_model(model_name, model_config=None):
    if model_name == 'blip2':
        from .interfaces.blip2 import get_blip2
        return get_blip2(model_config)
    elif model_name == 'llava':
        from .interfaces.llava import get_llava
        return get_llava(model_config)
        
    
if __name__=='__main__':
    get_model(model_name='blip2')