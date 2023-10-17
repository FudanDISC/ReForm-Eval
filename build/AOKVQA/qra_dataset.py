import torch
import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import logging
from collections import defaultdict, Counter
from utils.data_utils import base64_to_image, get_image
from datasets import load_dataset

def random_options(options, answer):
    neg_options = [opt for opt in options if opt != answer]
    random.shuffle(neg_options)
    # valid_options = neg_options[:n-1] if n < len(neg_options) + 1 else neg_options
    valid_options = neg_options
    valid_options.append(answer)
    random.shuffle(valid_options)
    return valid_options, valid_options.index(answer)
def get_options(options, answer):
    return options, options.index(answer)

class VQRA_SingleChoice(Dataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/VQRA_aokvqa_qra_val.yaml', proc=None, duplication=1):
        logging.info('Loading the VQA from {}'.format(config))
        self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        logging.info('The data config is: {}'.format(json.dumps(self.config)))
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        
        if args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
            data = data[0]
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['data_config']['offline_huggingface_data']}, split='test')
            data = data[0]
        else: 
            data = json.load(open(self.config['data_config']['data_path'], 'r'))
        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        data = data['data']
        if args.infer_method == 'generation' and args.formulation == 'SingleChoice':
            self.instruction_list = [
                "Kindly examine the picture, reasoning, and query, and subsequently choose the accurate response from the provided choices.",
                "Take a moment to analyze the image, along with the underlying logic and the posed question, before picking the right answer from the given alternatives.",
                "Begin by dissecting the image, understanding the reasoning, and considering the question; then, indicate the correct answer from the provided options.",
                "Your task is to assess the image, the rationale behind it, and the question being asked. Afterward, select the appropriate answer from the given options.",
                "Your assignment involves a careful analysis of the image, rationale, and question. Once done, proceed to select the correct answer from the provided options.",
            ]
        elif args.infer_method == 'likelihood' or args.formulation == 'Generation':
            self.instruction_list = [
                "Kindly examine the image and its underlying reasoning, and proceed to respond to the posed question.",
                "Take a moment to analyze both the image and the reasoning behind it, and then provide an answer to the question.",
                "Begin by carefully assessing the image and its rationale, and subsequently offer your response to the presented question.",
                "Your task involves analyzing the image and the accompanying reasoning, followed by addressing the question with your answer.",
                "Your assignment requires you to thoroughly review the image and its rationale, and then provide a response to the question.",
            ]
        
        self.in_context_history = [
            # {'from': 'human', 'value': 'can you see the image? Options: (A) yes; (B) no; (C) not sure; (D) maybe.'},
            {'from': 'human', 'value': 'can you see the image? Rationale: There is an image as input which is successfully loaded. Options: (A) yes; (B) no; (C) not sure; (D) maybe.'}, 
            {'from': 'assistant', 'value': '(A) yes'}
        ]        
        answer_prefix = getattr(proc, 'response_prefix', None)
        if answer_prefix is not None:
            # need prefix in the in-context sample
            self.in_context_history[1]['value'] = "{} {}".format(answer_prefix, self.in_context_history[1]['value'])

         
        if duplication > 1:
            # the data sample is repeated with different prompts
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number of different prompts"
        
        self.samples = []
        self.proc = proc
        self.duplication = duplication
        
        for i, item in tqdm.tqdm(enumerate(data), desc='preprocessing the data file'):
            if self.args.hf or self.args.offline_hf:
                image_path = item['image_id']
            else:
                image_name = '{:012d}.jpg'.format(item['image_id'])
                image_path = os.path.join(self.image_path, image_name)
                assert os.path.exists(image_path), 'the image {} does not exist, please check'.format(image_path)


            current_sample = {
                'sample_id': item['question_id'],
                'image': image_path,
                'question': item['question'],
                'answer': item['answer'],
                'rationale': item['rationale'],
                'answer_options': item['answer_options']
            }
            self.samples.append(current_sample)

    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        rationale = new_sample['rationale']
        
        if self.args.hf == True or self.args.offline_hf:
            image = base64_to_image(new_sample['image'])
            new_sample['image'] = image 
        else:
            image = get_image(new_sample['image'])
            new_sample['image'] = image  
            
        if self.args.shuffle_options:
            valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        else:
            valid_options, answer = get_options(new_sample['answer_options'], new_sample['answer'])
        if self.args.formulation == 'SingleChoice':
            new_sample['answer_options'] = valid_options
        elif self.args.formulation == 'Generation':
            new_sample.pop('answer_options')
        new_sample['answer'] = str(answer)
        
        # if self.duplication > 1:
        #     # iterate through all possible prompt
        #     inner_sample_index = index % self.duplication
        #     new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        # elif self.args.random_instruct:
        #     # randomly choose one prompt
        #     new_sample['instruct'] = random.choice(self.instruction_list)
        # else:
        #     new_sample['instruct'] = self.instruction_list[0]
        
        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        else:
            sample_index = index // self.duplication
            new_sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]
        
            
        new_sample['question'] = new_sample['question'] + f' Rationale: {rationale}' 
                
        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            new_sample['history'] = [msg for msg in self.in_context_history]
        
        
        if self.proc is not None:
            # print(new_sample)
            new_sample['text'] = self.proc(new_sample)
            # print(new_sample['text'])
        
        
        
        return new_sample
    
    def rawitem(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        new_sample['answer_options'] = valid_options
        new_sample['answer'] = str(answer)
        new_sample['instruct'] = random.choice(self.instruction_list)
        return new_sample

    def __len__(self):
        return len(self.samples) * self.duplication



def get_vqra(args, config, formulation, preprocessor):
    if formulation in ['SingleChoice', 'Generation']:
        if config is None:
            return VQRA_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return VQRA_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))


if __name__=='__main__':
    ds = VQRA_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
