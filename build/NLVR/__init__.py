import yaml

def get_nlvr(args, config, formulation, preprocessor):
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    if config['task'] == 'nlvrm': # natural language visual reasoning matching
        from .natural_language_visual_reasoning_dataset import NLVRMatching
        if formulation == 'SingleChoice':
            return NLVRMatching(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    elif config['task'] == 'nlvrs': # natural language visual reasoning selection
        from .natural_language_visual_reasoning_dataset import NLVRSelection
        if formulation == 'SingleChoice':
            return NLVRSelection(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    else:
        raise ValueError('current task {} is not supported yet'.format(config['task']))
        
        