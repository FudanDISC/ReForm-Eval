import yaml

def get_flowers102(args , config , formulation , preprocessor):
    if type(config) == str:
        config = yaml.load(open(config , 'r') , Loader=yaml.Loader)
    else:
        config = config
    
    from .flowers102_dataset import Flowers102_Dataset
    if formulation == 'SingleChoice':
        if config is None:
            return Flowers102_Dataset(args=args , proc=preprocessor , duplication=args.dataset_duplication)
        else:
            return Flowers102_Dataset(args=args , config=config , proc=preprocessor , duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))   