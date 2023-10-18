import yaml

def get_cute80(args, config, formulation, preprocessor):
    if config is None:
        from .ocr_dataset import OCR_OpenEnded
        if formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
            return OCR_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'ocr':
            from .ocr_dataset import OCR_OpenEnded
            if formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
                return OCR_OpenEnded(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))