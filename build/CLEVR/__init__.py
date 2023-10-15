import yaml

def get_clevr(args, config, formulation, preprocessor):
    if config is None:
        from .spatial_dataset import Spatial_SingleChoice, Spatial_TrueOrFalse, Spatial_OpenEnded
        if formulation == 'SingleChoice': # but use true or false
            return Spatial_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        elif formulation == 'TrueOrFalse':
            return Spatial_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        elif formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
            return Spatial_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'spatial':
            from .spatial_dataset import Spatial_SingleChoice, Spatial_TrueOrFalse, Spatial_OpenEnded
            if formulation == 'SingleChoice': # but use true or false
                return Spatial_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return Spatial_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
                return Spatial_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)