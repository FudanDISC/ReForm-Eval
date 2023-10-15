import yaml

def get_refcoco(args, config, formulation, preprocessor):
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    if config['task'] == 'res': # referring expression selection
        from .referring_expression_selection import ReferringExpressionSelection
        if formulation == 'SingleChoice':
            return ReferringExpressionSelection(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            raise ValueError('current formulation {} for current task {} is not support yet'.format(formulation, config['task']))
    else:
        raise ValueError('current task {} is not supported yet'.format(config['task']))