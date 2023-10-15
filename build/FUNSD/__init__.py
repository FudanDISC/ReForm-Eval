import yaml

def get_funsd(args, config, formulation, preprocessor):
    if config is None:
        from .kie_dataset import KIE_SingleChoice, KIE_OpenEnded
        if formulation == 'SingleChoice': # but use true or false
            return KIE_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        elif formulation == 'OCROpenEnded' or formulation == 'KIEOpenEnded':
            return KIE_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'kie':
            from .kie_dataset import KIE_SingleChoice, KIE_OpenEnded
            if formulation == 'SingleChoice': # but use true or false
                return KIE_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'OCROpenEnded' or formulation == 'KIEOpenEnded':
                return KIE_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))