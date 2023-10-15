import yaml

def get_mp3d(args, config, formulation, preprocessor):
    if config is None:
        from .spatial_dataset import Spatial_TrueOrFalse, Spatial_SingleChoice
        if formulation == 'SingleChoice': # but use true or false
            return Spatial_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        elif formulation == 'TrueOrFalse':
            return Spatial_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'spatial':
            from .spatial_dataset import Spatial_TrueOrFalse, Spatial_SingleChoice
            if formulation == 'SingleChoice': # but use true or false
                return Spatial_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return Spatial_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
        elif config['task'] == 'oo_relation':
            from .spatial_dataset import Spatial_OO_TrueOrFalse, Spatial_OO_SingleChoice
            if formulation == 'SingleChoice': # but use true or false
                return Spatial_OO_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
                return Spatial_OO_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return Spatial_OO_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
        elif config['task'] == 'oa_depth':
            from .spatial_dataset import Spatial_OA_Depth_TrueOrFalse, Spatial_OA_Depth_SingleChoice
            if formulation == 'SingleChoice': # but use true or false
                return Spatial_OA_Depth_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
                return Spatial_OA_Depth_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return Spatial_OA_Depth_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
            else:
                raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
        # multiple images
        elif config['task'] == 'oa_count':
            from .spatial_dataset import Spatial_OA_Count_SingleChoice
            if formulation == 'SingleChoice': # but use true or false
                return Spatial_OA_Count_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
        elif config['task'] == 'oa_relation':
            from .spatial_dataset import Spatial_OA_TrueOrFalse, Spatial_OA_SingleChoice
            if formulation == 'SingleChoice': # but use true or false
                return Spatial_OA_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
                return Spatial_OA_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return Spatial_OA_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
            else:
                raise ValueError('current formulation {} for current task {} is not supported yet'.format(formulation, config['task']))
        else:
            raise ValueError('current task {} is not supported yet'.format(config['task']))
        