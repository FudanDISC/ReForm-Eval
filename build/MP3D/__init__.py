import yaml

def get_mp3d(args, config, formulation, preprocessor):
    if config is None:
        from .spatial_dataset import Spatial_SingleChoice
        if formulation == 'SingleChoice': # but use true or false
            return Spatial_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'spatial':
            from .spatial_dataset import Spatial_SingleChoice
            if formulation == 'SingleChoice': # but use true or false
                return Spatial_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
        else:
            raise ValueError('current task {} is not supported yet'.format(config['task']))
        