import os, json
from utils.logger import setup_logger
from metrics import get_metric
import tqdm
from utils.run_utils import *

from argparse import Namespace
from argparse import ArgumentParser

def loader_eval(formulation, multi_round_eval, eval_stability, prediction_file=None):
    args = Namespace()

    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = -1

    args.formulation = formulation
    args.multi_round_eval = multi_round_eval
    args.eval_stability = eval_stability
    # check the output_dir
    args.prediction_file = prediction_file
    args.output_dir = os.path.dirname(args.prediction_file)
    # args.full_path = json_file # os.path.join(args.output_dir, args.json_file)

    global logger
    logger = setup_logger('ReForm-Eval Evaluation', args.output_dir, args.local_rank)
    # logger.info('Evaluating with {} GPUs'.format(args.n_gpus))
    if os.path.exists(args.prediction_file):
        logger.info('found the existing prediction in {}'.format(args.prediction_file))
        full_res = json.load(open(args.prediction_file, 'r'))
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
    
    if args.formulation == 'Generation':
        cider_metric, cider_metrics = metric(full_res)
        logger.info('the evalueted {} result: {}'.format(args.formulation, cider_metric))
    else:
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
    if args.eval_stability:
        # perform stability measurement
        assert args.formulation == 'SingleChoice', 'only single-choice problems support instability evaluation!'
        mean_entropy = entropy_calculation(question2pred)
        logger.info('the measured stability (entropy on predictions) across prompts: {}'.format(mean_entropy))
    
    if args.multi_round_eval:
        multi_round_res = multi_round_eval(round2metric)
        logger.info('corr(round, performance):{}, slope of linear_model(round, performance):{}'.format(multi_round_res[0], multi_round_res[1]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--formulation', type=str, default=None, help='the problem formulation to perform, must be in ("Generation", "SingleChoice")')
    parser.add_argument('--eval_stability', action='store_true', help='whether to evaluate the stability')
    parser.add_argument('--multi_round_eval', action='store_true', help='whether to evaluate multi-round performance')
    # output setup
    parser.add_argument('--prediction_file', type=str, default=None, required=True, help='the prediction json file')
    # parser.add_argument('--output_dir', type=str, default=None, help='the path to save the log, default to be in the directory of the prediction file')
    args = parser.parse_args()
    
    # set the output dir to the prediction directory
    # if args.output_dir is None:
    #     args.output_dir = os.path.dirname(args.prediction_file)
    
    loader_eval(args.formulation, args.multi_round_eval, args.dataset_duplication,
                args.eval_stability, args.prediction_file)

if __name__=='__main__':
    main()