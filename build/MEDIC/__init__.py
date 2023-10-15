import yaml

def get_medic(args, config, formulation, preprocessor):
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    if config['task'] == 'dts': # disaster type selection
        from .disaster_type_dataset import DisasterTypeSelection
        if formulation == 'SingleChoice':
            return DisasterTypeSelection(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not support yet'.format(formulation, config['task']))
    else:
        raise ValueError('current task {} is not supported yet'.format(config['task']))