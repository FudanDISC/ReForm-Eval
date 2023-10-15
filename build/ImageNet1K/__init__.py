import yaml

def get_imagenet1k(args , config , formulation , preprocessor):
    if type(config) == str:
        config = yaml.load(open(config , 'r') , Loader=yaml.Loader)
    else:
        config = config

    from .imagenet1k_dataset import ImageNet1K_Dataset
    if formulation == 'SingleChoice':
        if config is None:
            return ImageNet1K_Dataset(args=args , proc=preprocessor , duplication=args.dataset_duplication)
        else:
            return ImageNet1K_Dataset(args=args , config=config , proc=preprocessor , duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))   