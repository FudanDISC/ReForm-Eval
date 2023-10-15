import argparse
from models import get_model
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import os, json
from utils.logger import setup_logger
from build import build_dataset
from metrics import get_metric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
from utils.run_utils import *
from PIL import Image    

def load_reform_dataset(dataset_name:str, 
                 formulation:str, 
                 dataset_config:Optional[dict]=None,
                 inference_method:Optional[str]='generation',
                 in_context_sample:Optional[bool]=True,
                 random_instruct:Optional[bool]=True,
                 data_duplication:Optional[int]=5,
                 shuffle_options:Optional[bool]=True,
                 load_from_hf:Optional[bool]=True, 
                 preprocessor=None):
    """
    Return the constructed dataset
    Parameters:
        dataset_name: the dataset name to load.
        formulation: the problem formulation.
        dataset_config: the path to the config file, using the default path if not specified.
        infer_method: inference method, influencing the data format and instructions, should be "generation" or "likelihood".
        in_context_sample: whether to include an in-context sample, defalut to True.
        rando_instruct: use different instruction for the same sample, default to True.
        data_duplication: the number of multiple tests for the same sample, default to 5.
        shuffle_options: shuffle the options for the same sample, default to True.
        load_from_hf: whether to load from huggingface, load from local if set to False.
        preprocessor: Optional, the model processor to process.
    Return:
        dataset: the constructed dataset
    Usage:
        >>> from datasets import build_dataset
        >>> dataset = build_dataset("VisDial", "SingleChoice")
    """

    # prepare the argument namespace for input
    from argparse import Namespace
    args = Namespace()

    # set the important arguments to the namespace
    args.hf = load_from_hf

    assert inference_method in ['generation', 'likelihood'], "the inference method should be 'generation' or 'likelihood'"
    args.infer_method = inference_method

    args.in_context_sample = in_context_sample

    # randomness-related parameters
    args.dataset_duplication = data_duplication
    args.random_instruct = random_instruct
    args.shuffle_options = shuffle_options

    # set the default arguments
    args.capitalize = True
    args.dataset_subsample = None
    args.options_in_history = True
    args.online_multi_round = True

from argparse import Namespace

def loader_eval(formulation,multi_round_eval, json_path):
        
    args = Namespace()
    args.formulation = formulation
    args.multi_round_eval = multi_round_eval
    args.dataset_duplication
    args.eval_stability
    args.per_gpu_eval_batch_size
    args.n_gpus

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
        

def run_eval_multi_round(args, dataset, model):
    # setup the dataloader
    logger.info('Running with the online multi round format')
    model.eval()
    args.eval_batch_size = args.per_gpu_eval_batch_size * args.n_gpus
    sampler = SequentialSampler(dataset) if not args.distributed else DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, sampler=sampler, batch_size=args.per_gpu_eval_batch_size, collate_fn=naive_list_collate_fn)
    
    # setup the generation parameters
    generation_kwargs = {}
    if args.temperature is not None:
        generation_kwargs['temperature'] = args.temperature
    if args.max_new_tokens is not None:
        generation_kwargs['max_new_tokens'] = args.max_new_tokens

    # setup to the likelihood parameters
    likelihood_kwargs = {}
    if args.likelihood_reduction is not None:
        likelihood_kwargs['likelihood_reduction'] = args.likelihood_reduction

    current_res = []
    metric = get_metric(args.formulation)
    logger.info("***** Runing Evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    for batch in tqdm.tqdm(dataloader, desc='evaluating'):
        # print(batch)
        if args.infer_method == 'generation':
            res = model(batch['image'], batch['text'], **generation_kwargs)
        else:
            res = model(batch['image'], batch['text'], batch['answer_options'], **likelihood_kwargs)
        
        if type(batch['text'][0]) == dict:
            # for Shikra, the processed object is not only a string
            batch['text'] = [raw_item['raw_text'] for raw_item in batch['text']]
        # make the update to the dialog history in the dataset
        generated_history_infos = get_pred_result(batch, res, metric)
        # print(args.local_rank, generated_history_infos)
        batch['prediction'] = res
        gathered_history = [i for i in range(args.n_gpus)]
        dist.all_gather_object(gathered_history, generated_history_infos)
        dataset.update_history(gathered_history)
        current_res.append(batch)
    
    # post_processing the results
    final_res = []
    for item in current_res:
        for i in range(len(item['prediction'])):
            sample = {k: v[i] for k,v in item.items() if not isinstance(v, torch.Tensor)}
            final_res.append(sample)
    
    # remove duplication if necessary in Distributed version
    if args.distributed and len(dataset) % args.n_gpus != 0:
        residual_samples = len(dataset) % args.n_gpus
        if not args.local_rank < residual_samples:
            final_res = final_res[:-1]

    with open(get_output_name(args, mid_output=True), 'w') as wf:
        json.dump(final_res, wf)

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