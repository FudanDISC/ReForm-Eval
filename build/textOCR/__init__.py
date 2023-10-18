import yaml

def get_textocr(args, config, formulation, preprocessor):
    if config is None:
        from .ocr_dataset import OCR_OpenEnded
        if formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
            return OCR_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} is not supported yet'.format(formulation))
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'ocr':
            from .ocr_dataset import OCR_OpenEnded
            if formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
                return OCR_OpenEnded(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        if config['task'] == 'gocr':
            from .groundocr_dataset import GroundOCR_OpenEnded
            if formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
                return GroundOCR_OpenEnded(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))