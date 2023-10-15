import yaml

def get_cifar10(args , config , formulation , preprocessor):
    if type(config) == str:
        config = yaml.load(open(config , 'r') , Loader=yaml.Loader)
    else:
        config = config
    
    from .cifar10_dataset import CIFAR10_Dataset
    if formulation == 'SingleChoice':
        if config is None:
            return CIFAR10_Dataset(args=args , proc=preprocessor , duplication=args.dataset_duplication)
        else:
            return CIFAR10_Dataset(args=args , config=config , proc=preprocessor , duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))   