import yaml

def get_cocotext(args, config, formulation, preprocessor):
    if config is None:
        from .text_legibility_dataset import TextLegibility_TrueOrFlase
        if formulation == 'SingleChoice': # but use true or false
            return TextLegibility_TrueOrFlase(args=args, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'tl':
            from .text_legibility_dataset import TextLegibility_TrueOrFlase
            if formulation == 'SingleChoice':
                return TextLegibility_TrueOrFlase(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'ttc':
            from .text_type_classification_dataset import TextTypeClassification_SingleChoice
            if formulation == 'SingleChoice':
                return TextTypeClassification_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'ocr':
            from .ocr_dataset import OCR_SingleChoice, OCR_TrueOrFalse, OCR_OpenEnded
            if formulation == 'SingleChoice':
                return OCR_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return OCR_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
                return OCR_OpenEnded(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'gocr':
            from .groundocr_dataset import GroundOCR_SingleChoice, GroundOCR_TrueOrFalse, GroundOCR_OpenEnded
            if formulation == 'SingleChoice':
                return GroundOCR_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return GroundOCR_TrueOrFalse(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'OCROpenEnded' or formulation == 'OpenEnded':
                return GroundOCR_OpenEnded(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))