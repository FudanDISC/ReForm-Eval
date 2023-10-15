import yaml
# from .dataset import TDIUC_Dataset
from .tdiuc_dataset import TDIUC_Dataset

# all 12 tasks in TDIUC
all_task = ['color' , 'object_presence' , 'object_recognition' , 'scene_recognition' , \
    'counting' , 'sentiment_understanding' , 'positional_reasoning' , 'utility_affordance' , \
    'sport_recognition' , 'attribute' , 'activity_recognition' , 'absurd']

def get_tdiuc(args , config , formulation , preprocessor):
    if formulation == 'SingleChoice':
        if config is None:
            print('The task is not set in config, the color recognition task is evaluated by default')
            return TDIUC_Dataset(
                args = args,
                proc = preprocessor,
                config = config,
                duplication = args.dataset_duplication
            )
        else:
            config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
            if config['task'] in all_task:
                return TDIUC_Dataset(
                    args = args,
                    proc = preprocessor,
                    config = config,
                    duplication = args.dataset_duplication,
                    task_kind = all_task.index(config['task'])+1
                )
            else:
                raise ValueError('No such task for TDIUC !!')
    else:
        raise ValueError('Haven\'t finished the other parts yet')


