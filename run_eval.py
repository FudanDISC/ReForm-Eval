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

def get_output_name(args, mid_output=True):
    tmp_model_name = get_model_name(args.model_name)
    if mid_output:
        return os.path.join(args.output_dir, 
                            '{}_{}_{}_{}_{}_rank{}.json'.format(args.dataset_name, args.formulation, args.infer_method,
                                                            args.model, tmp_model_name, args.local_rank))
    else:
        return os.path.join(args.output_dir, 
                            '{}_{}_{}_{}_{}.json'.format(args.dataset_name, args.formulation, args.infer_method,
                                                            args.model, tmp_model_name, args.local_rank))

def get_all_output_names(args):
    tmp_model_name = get_model_name(args.model_name)
    return [os.path.join(args.output_dir, 
                            '{}_{}_{}_{}_{}_rank{}.json'.format(args.dataset_name, args.formulation, args.infer_method,
                                                            args.model, tmp_model_name, r)) for r in range(args.n_gpus)]

def run_eval(args, dataset, model):
    # setup the dataloader
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
    logger.info("***** Runing Evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    for batch in tqdm.tqdm(dataloader, desc='evaluating'):
        if args.infer_method == 'generation':
            res = model(batch['image'], batch['text'], **generation_kwargs)
        else:
            res = model(batch['image'], batch['text'], batch['answer_options'], **likelihood_kwargs)
        batch['prediction'] = res
        if type(batch['text'][0]) == dict:
            # for Shikra, the processed object is not only a string
            batch['text'] = [raw_item['raw_text'] for raw_item in batch['text']]
        current_res.append(batch)
    
    # post_processing the results
    final_res = []
    for item in current_res:
        for i in range(len(item['prediction'])):
            # to avoid output torch.Tensor and Images into the output file
            sample = {k: v[i] for k,v in item.items() if (not isinstance(v, torch.Tensor) and not isinstance(v[i], Image.Image))}
            final_res.append(sample)
    
    # remove duplication if necessary in Distributed version
    if args.distributed and len(dataset) % args.n_gpus != 0:
        residual_samples = len(dataset) % args.n_gpus
        if not args.local_rank < residual_samples:
            final_res = final_res[:-1]

    with open(get_output_name(args, mid_output=True), 'w') as wf:
        json.dump(final_res, wf)

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
    if args.dataset_duplication > 1 or args.eval_stability:
        # perform stability measurement
        mean_entropy = entropy_calculation(question2pred)
        logger.info('the measured stability (entropy on predictions) across prompts: {}'.format(mean_entropy))
    
    if args.multi_round_eval:
        multi_round_res = multi_round_eval(round2metric)
        logger.info('corr(round, performance):{}, slope of linear_model(round, performance):{}'.format(multi_round_res[0], multi_round_res[1]))

def main():
    parser = argparse.ArgumentParser()
    # model-related parameters
    parser.add_argument('--model', type=str, default=None, help='the model family name')
    parser.add_argument('--model_name', type=str, default=None, help='the model name to load')
    parser.add_argument('--model_type', type=str, default=None, help='the model type to set')
    # dataset-related parameters
    parser.add_argument('--dataset_name', type=str, default=None, help='the dataset name to evaluate on')
    parser.add_argument('--formulation', type=str, default=None, help='the problem formulation to perform, must be in ("Generation", "SingleChoice")')
    parser.add_argument('--dataset_config', type=str, default=None, help='the config file path, using the default path without explicit ')
    parser.add_argument('--dataset_duplication', type=int, default=1, help='duplicate the sample for evaluating the stability')
    parser.add_argument('--in_context_sample', action='store_true', help='whether to provide in-context-learning samples')
    parser.add_argument('--capitalize', action='store_true', help='whether to capitalize the qa')
    # 0805 add
    parser.add_argument('--yesno_instruct', action='store_true', help='whether add "please answer yes or no" to the full instruct')
    parser.add_argument('--answer_space_instruct', action='store_true', help='whether add answer space to the full instruct')
    # running parameters
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=1, help='the batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='workers in dataloader')
    parser.add_argument('--half_evaluation', action='store_true', help='whether to use half precision for evluation')
    # general evaluation setup
    parser.add_argument('--do_eval', action='store_true', help='whether to evluate the output.')
    parser.add_argument('--eval_stability', action='store_true', help='whether to evaluate the stability')
    # parameters for model generation
    parser.add_argument('--temperature', type=float, default=None, help='the temperature for generation')
    parser.add_argument('--max_new_tokens', type=int, default=None, help='max new tokens to generate')
    # parameters for likelihood measurement
    parser.add_argument('--likelihood_reduction', type=str, default=None, help='the reduction method for likelihood measurement')
    # parameters for SingleChoice problem
    parser.add_argument('--infer_method', type=str, default='generation', help='the inference method to use, must be in ["generation", "likelihood"]')
    parser.add_argument('--option_mark', type=str, default=None, help='the index mark for options in single-shoice questions, \
                        "number" for (1,2,3,4), "lower" for (a,b,c,d) while "upper" for (A,B,C,D)')
    # parameters for randomness control
    parser.add_argument('--random_instruct', action='store_true', help='whether to use random instructions')
    parser.add_argument('--shuffle_options', action='store_true', help='whether to shuffle options')
    # parameters for multi-round problem
    parser.add_argument('--options_in_history', action='store_true', help='whether to put options in history.')
    parser.add_argument('--online_multi_round', action='store_true', help='make online update to the history during dialog')
    parser.add_argument('--multi_round_eval', action='store_true', help='whether to evaluate multi-round performance')
    # output setup
    parser.add_argument('--output_dir', type=str, default='./output/', help='the path to save the output')
    # debug mode
    parser.add_argument('--dataset_debug', action='store_true', help='debug on the dataset setup')
    parser.add_argument('--dataset_subsample', type=int, default=None, help='only n sub-samples of the dataset')
    # core
    parser.add_argument('--core_eval', action='store_true', help='only eval on the core datasets')
    # hugging face
    parser.add_argument('--hf', action='store_true', help='whether to load the dataset directly from Hugging Face')
    parser.add_argument('--offline_hf', action='store_true', help='whether to load the Hugging Face data from the local path')
    args = parser.parse_args()

    if 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.n_gpus = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.distributed = False
        args.local_rank = -1
        args.n_gpus = -1

    # debug setting
    if args.dataset_debug:
        from utils.debug_utils import datasetdebug
        datasetdebug(args)
        return None
    
    # setup the logger
    if not os.path.isdir(args.output_dir) and args.local_rank < 1:
        os.makedirs(args.output_dir)
    global logger
    logger = setup_logger('ReForm-Eval Evaluation', args.output_dir, args.local_rank)
    logger.info('Evaluating with {} GPUs'.format(args.n_gpus))

    # if the output prediction already exists
    if args.do_eval:
        if os.path.exists(get_output_name(args, mid_output=False)):
            logger.info('found the existing prediction in {}'.format(get_output_name(args, mid_output=False)))
            full_res = json.load(open(get_output_name(args, mid_output=False), 'r'))
            # ori_args = torch.load(get_output_name(args, mid_output=False)[:-4]+'args.bin')
            # logger.info('And the original arguments are: %s', ori_args)
            metric_eval(args, full_res=full_res)
            return

    # loading the model
    model_config = {'device': device, 'half': args.half_evaluation, 'inference_method': args.infer_method}
    if args.model_name is not None:
        model_config['model_name'] = args.model_name 
    if args.model_type is not None:
        model_config['model_type'] = args.model_type
    logger.info('Loading model: {} with configure: {}'.format(args.model, json.dumps(model_config)))
    model, preprocessor = get_model(args.model, model_config=model_config)
    logger.info('Each GPU consumes memory of {}'.format(gpu_info(0)[1]))

    # setup the single-choice
    if args.formulation == 'SingleChoice' and args.option_mark is not None:
        logger.info('Using {} option mark for the single-choice questions'.format(args.option_mark))
        preprocessor.set_mark(args.option_mark)

    if args.half_evaluation:
        model = model.half()

    # loading the dataset
    logger.info('Evaluating model: {} with configure: {}'.format(args.model, json.dumps(model_config)))
    eval_dataset = build_dataset(args, args.dataset_name, args.formulation, args.dataset_config, 
                                 preprocessor)
    
    # if args.dataset_subsample is not None:
    #     eval_dataset = eval_dataset[:args.dataset_subsample]
    
    # run the evaluation
    if args.online_multi_round:
        assert args.num_workers == 0, 'current multi-round evaluation requires the num_workers to be 0 (no pre-fetch)'
        run_eval_multi_round(args, eval_dataset, model)
    else:
        run_eval(args, eval_dataset, model)
    torch.distributed.barrier()
    if args.local_rank == 0 or args.local_rank == -1:
        full_res = []
        for fn in get_all_output_names(args):
            full_res.extend(json.load(open(fn, 'r')))
            os.remove(fn)
        with open(get_output_name(args, mid_output=False), 'w') as wf:
            json.dump(full_res, wf)
        # saving the arguments
        torch.save(args, get_output_name(args, mid_output=False)[:-4]+'args.bin')
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

    if args.do_eval and args.local_rank <= 0:
        metric_eval(args, full_res=full_res)
    return 

if __name__=='__main__':
    main()
