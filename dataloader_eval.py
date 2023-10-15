import os, json
from utils.logger import setup_logger
from metrics import get_metric
import tqdm
from utils.run_utils import *

from argparse import Namespace

def loader_eval(formulation, multi_round_eval, dataset_duplication, eval_stability, json_path):
    args = Namespace()
    
    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = -1

    args.formulation = formulation
    args.multi_round_eval = multi_round_eval
    args.dataset_duplication = dataset_duplication
    args.eval_stability = eval_stability


    global logger
    logger = setup_logger('LLMV-Bench Evaluation', json_path, args.local_rank)
    # logger.info('Evaluating with {} GPUs'.format(args.n_gpus))
    if os.path.exists(json_path):
        logger.info('found the existing prediction in {}'.format(json_path))
        full_res = json.load(open(json_path, 'r'))
        # ori_args = torch.load(get_output_name(args, mid_output=False)[:-4]+'args.bin')
        # logger.info('And the original arguments are: %s', ori_args)
        metric_eval(args, full_res=full_res)
        return

def metric_eval(args, full_res):
    from collections import defaultdict
    import numpy as np
    # loading the evluating metric
    logger.info('evaluating the predictions with the {} metric'.format(args.formulation))
    metric = get_metric(args.formulation)
    
    sum_of_metric = 0
    # for accuracy metric
    question2metric = defaultdict(list)
    # for stability measurement
    question2pred = defaultdict(list)
    # for multi-round measurement
    if args.multi_round_eval:
        round2metric = defaultdict(list)
    
    ### for format hit rate
    hit_num  = 0
    for item in tqdm.tqdm(full_res, desc='judging with the selected metric'):
        m, pred = metric(item['prediction'], item['answer'])
        sum_of_metric += m
        if args.multi_round_eval:
            round2metric[item['round_id']].append(m)
        question2metric[item['sample_id']].append(m)
        # map the predicted index back to the option
        if pred is not None:
            hit_num += 1
            try:
                question2pred[item['sample_id']].append(item['answer_options'][pred])
            except:
                print('found out of range prediction: {}'.format(pred))
                question2pred[item['sample_id']].append(item['prediction'])
        else:
            question2pred[item['sample_id']].append(item['prediction'])
    
    metric_matrix = np.array(list(question2metric.values()))
    mean_metric = np.mean(metric_matrix)
    logger.info('the evalueted {} result: {}'.format(args.formulation, mean_metric))
    logger.info('the format hit rate is {}'.format(hit_num/len(full_res)))
    if args.dataset_duplication > 1 or args.eval_stability:
        # perform stability measurement
        mean_entropy = entropy_calculation(question2pred)
        logger.info('the measured stability (entropy on predictions) across prompts: {}'.format(mean_entropy))
    
    if args.multi_round_eval:
        multi_round_res = multi_round_eval(round2metric)
        logger.info('corr(round, performance):{}, slope of linear_model(round, performance):{}'.format(multi_round_res[0], multi_round_res[1]))

# if __name__=='__main__':
#     loader_eval(formulation, multi_round_eval, dataset_duplication, eval_stability, json_path)
