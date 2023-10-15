import yaml

def get_snli_ve(args, config, formulation, preprocessor):
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    if config['task'] == 'ves': # visual entailment selection
        from .visual_entailment_dataset import VisualEntailmentSelection
        if formulation == 'SingleChoice':
            return VisualEntailmentSelection(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not support yet'.format(formulation, config['task']))
    elif config['task'] == 'vem': # visual entailment matching
        from .visual_entailment_dataset import VisualEntailmentMatching
        if formulation == 'SingleChoice':
            return VisualEntailmentMatching(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not support yet'.format(formulation, config['task']))
    else:
        raise ValueError('current task {} is not supported yet'.format(config['task']))