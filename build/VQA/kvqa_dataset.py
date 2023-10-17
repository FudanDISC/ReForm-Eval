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

class KVQA_SingleChoice(Dataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/KVQA_viquae_val_v2.0.yaml', proc=None, duplication=1):
        logging.info('Loading the KVQA from {}'.format(config))
        self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        logging.info('The data config is: {}'.format(json.dumps(self.config)))
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        
        if self.args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['data_config']['offline_huggingface_data']}, split='test')
        else: 
            data = json.load(open(self.config['data_config']['data_path'], 'r'))
        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        data = data['data']
        if args.infer_method == 'generation' and args.formulation == 'SingleChoice':
            self.instruction_list = [
                "Take a moment to examine both the picture and its context, then pick the accurate choice from the provided options.",
                # "After reviewing the image and its surrounding circumstances, choose the right option from the following selections.",
                # "Analyze the picture and its context carefully, then indicate the correct answer from the choices given.",
                "Spend some time evaluating the image and its context, and then make your selection from the available options.",
                "Please consider both the image and its context before making a selection from the provided choices.",
                "Assess the image and its context, then indicate the appropriate choice from the following options.",
                # "Examine the image and its context thoroughly, then identify the correct option from the list provided.",
                # "Prior to choosing your answer, analyze the image and its relevant context."
                "Make sure to understand the image and its context before picking the correct option from the choices."
                # "After evaluating the image and considering its context, please proceed to select the accurate answer from the provided options."
            ]
        elif args.infer_method == 'likelihood' or args.formulation == 'Generation':
            self.instruction_list = [
                "Kindly assess both the image and the surrounding context before responding to the question.",
                "Take a moment to examine the picture and its context, and then provide an answer to the question.",
                "After reviewing the image and its context, please proceed to address the question at hand.",
                # "Evaluate the image and its surrounding circumstances, and then offer a response to the question asked.",
                "Carefully consider the image and its context before formulating a response to the posed question.",
                # "Spend some time analyzing both the image and the context, and then proceed to answer the question.",
                # "Examine the image and its context thoroughly, and then provide a response to the question posed.",
                "Prior to answering the question, take the time to analyze the image and its relevant context.",
                # "Make sure to understand the image and its context before providing a response to the question.",
                # "After evaluating the image and considering its context, please share your answer to the question."
            ]
        
        self.in_context_history = [
            {'from': 'human', 'value': 'Can you see the image? Context: There is an image. Options: (A) yes; (B) no; (C) not sure; (D) maybe.'},
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
                image_name = '{}'.format(item['image_id'])
                image_path = os.path.join(self.image_path, image_name)
                assert os.path.exists(image_path), 'the image {} does not exist, please check'.format(image_path)


            current_sample = {
                'sample_id': item['question_id'],
                'image': image_path,
                'question': item['question'],
                'answer': item['answer'],
                'answer_options': item['answer_options'],
                'context': item['context'],
            }
            self.samples.append(current_sample)

    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
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
        else:
            return 
        new_sample['answer'] = str(answer)
        context = new_sample['context'].strip()
        
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
         
            
        new_sample['question'] = new_sample['question'] + f' Context: {context}' 
        
        if self.args.yesno_instruct:
            new_sample['instruct'] = new_sample['instruct'] + ' Please answer yes or no.'
        
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



def get_kvqa(args, config, formulation, preprocessor):
    if formulation in ['SingleChoice', 'Generation']:
        if config is None:
            return KVQA_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return KVQA_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))


if __name__=='__main__':
    ds = KVQA_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
