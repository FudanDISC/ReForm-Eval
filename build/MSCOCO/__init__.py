import yaml

def get_mscoco(args, config, formulation, preprocessor):
    if config is None:
        from .multiclass_identification_dataset import MultiClassIden_SingleChoice
        if formulation == 'SingleChoice':
            return MultiClassIden_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'mci':
            from .multiclass_identification_dataset import MultiClassIden_SingleChoice
            if formulation == 'SingleChoice':
                return MultiClassIden_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'goi':
            from .grounded_object_identification_dataset import GroundedObjIden_SingleChoice
            if formulation == 'SingleChoice':
                return GroundedObjIden_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'mos':
            from .missing_object_selection_dataset import MissingObjectSelection_SingleChoice
            if formulation == 'SingleChoice':
                return MissingObjectSelection_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'itm':
            from .image_text_matching_dataset import ImageTextMatching
            if formulation == 'SingleChoice':
                return ImageTextMatching(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported by current itm task yet'.format(formulation))
        elif config['task'] == 'its':
            from .image_text_selection_dataset import ImageTextSelection
            if formulation == 'SingleChoice':
                return ImageTextSelection(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported by current its task yet'.format(formulation))
        elif config['task'] == 'caption':
            from .caption_dataset import Caption
            if formulation == 'Generation':
                return Caption(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            else:
                raise ValueError('current formulation {} is not supported by current caption task yet'.format(formulation))
        elif config['task'] == 'oc':
            from .object_counting_dataset import ObjectCounting_SingleChoice
            if formulation == 'SingleChoice':
                return ObjectCounting_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))


