from .visual_dialog_dataset import VisualDialog_SingleChoice

def get_visdial(args, config, formulation, preprocessor):
    if formulation == 'SingleChoice':
        if config is None:
            return VisualDialog_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return VisualDialog_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))
