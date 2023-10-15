import yaml

def get_vizwiz(args, config, formulation, preprocessor):
    if type(config) == str:
        config = yaml.load(open(config , 'r') , Loader=yaml.Loader)
    else:
        config = config

    from .vizwiz_dataset import VizWiz_Dataset
    if formulation == 'SingleChoice':
        if config is None:
            return VizWiz_Dataset(args=args , proc=preprocessor , duplication=args.dataset_duplication)
        else:
            return VizWiz_Dataset(args=args , config=config , proc=preprocessor , duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation)) 
