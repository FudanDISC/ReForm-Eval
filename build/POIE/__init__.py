import yaml

def get_poie(args, config, formulation, preprocessor):
    if config is None:
        from .kie_dataset import KIE_OpenEnded
        if formulation == 'OCROpenEnded' or formulation == 'KIEOpenEnded':
            return KIE_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'kie':
            from .kie_dataset import KIE_OpenEnded
            if formulation == 'OCROpenEnded' or formulation == 'KIEOpenEnded':
                return KIE_OpenEnded(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))