import yaml

def get_nocaps(args, config, formulation, preprocessor):
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    if config['task'] == 'caption':
        from .caption_dataset import Caption
        if formulation == 'Generation':
            return Caption(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} is not supported yet for caption task'.format(formulation))
    else:
        raise ValueError('current task {} is not supported yet for NoCaps dataset'.format(config['task']))