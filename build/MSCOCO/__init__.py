import yaml

def get_mscoco(args, config, formulation, preprocessor):
    if config is None:
        from .multiclass_identification_dataset import MultiClassIden_SingleChoice, MultiClassIden_TrueOrFlase
        if formulation == 'SingleChoice':
            return MultiClassIden_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        elif formulation == 'TrueOrFalse':
            return MultiClassIden_TrueOrFlase(args=args, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        if config['task'] == 'mci':
            from .multiclass_identification_dataset import MultiClassIden_SingleChoice, MultiClassIden_TrueOrFlase
            if formulation == 'SingleChoice':
                return MultiClassIden_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return MultiClassIden_TrueOrFlase(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'goi':
            from .grounded_object_identification_dataset import GroundedObjIden_SingleChoice, GroundedObjIden_TrueOrFlase
            if formulation == 'SingleChoice':
                return GroundedObjIden_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return GroundedObjIden_TrueOrFlase(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'og':
            from .object_grounding_dataset import ObjectGrounding_SingleChoice, ObjectGrounding_TrueOrFlase
            if formulation == 'SingleChoice':
                return ObjectGrounding_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
            elif formulation == 'TrueOrFalse':
                return ObjectGrounding_TrueOrFlase(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
            else:
                raise ValueError('current formulation {} is not supported yet'.format(formulation))
        elif config['task'] == 'om':
            from .object_matching_dataset import ObjectMatching_TrueOrFlase
            if formulation == 'SingleChoice':
                return ObjectMatching_TrueOrFlase(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication) 
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


