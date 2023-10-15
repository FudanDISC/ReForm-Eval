import yaml

def get_wikihow(args, config, formulation, preprocessor):
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    if config['task'] == 'wnss': # wikihow next step selection
        from .temporal_ordering_dataset import WikiHowNextStepSelection
        if formulation == 'SingleChoice':
            return WikiHowNextStepSelection(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    elif config['task'] == 'wtito': # wikihow text image temporal ordering
        from .temporal_ordering_dataset import WikiHowTextImageTemporalOrdering
        if formulation == 'SingleChoice':
            return WikiHowTextImageTemporalOrdering(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    elif config['task'] == 'witto': # wikihow image text temporal ordering
        from .temporal_ordering_dataset import WikiHowImageTextTemporalOrdering
        if formulation == 'SingleChoice':
            return WikiHowImageTextTemporalOrdering(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    elif config['task'] == 'wits': # wikihow image text selection
        from .temporal_ordering_dataset import WikiHowImageTextSelection
        if formulation == 'SingleChoice':
            return WikiHowImageTextSelection(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    else:
        raise ValueError('current task {} is not supported yet'.format(config['task']))
        