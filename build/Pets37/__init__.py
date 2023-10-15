import yaml

def get_pets37(args , config , formulation , preprocessor):
    if type(config) == str:
        config = yaml.load(open(config , 'r') , Loader=yaml.Loader)
    else:
        config = config
    
    from .pets37_datset import Pets37_Dataset
    if formulation == 'SingleChoice':
        if config is None:
            return Pets37_Dataset(args=args , proc=preprocessor , duplication=args.dataset_duplication)
        else:
            return Pets37_Dataset(args=args , config=config , proc=preprocessor , duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))   
