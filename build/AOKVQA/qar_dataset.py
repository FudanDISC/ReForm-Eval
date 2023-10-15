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

class VQAR_SingleChoice(Dataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/VQAR_aokvqa_qar_val.yaml', proc=None, duplication=1):
        logging.info('Loading the VQA from {}'.format(config))
        self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        logging.info('The data config is: {}'.format(json.dumps(self.config)))
        self.image_path = self.config['data_config']['image_path']
        self.args = args
    
        if args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
        else: 
            data = json.load(open(self.config['data_config']['data_path'], 'r'))
            
        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        data = data['data']
        if args.infer_method == 'generation' and args.formulation == 'SingleChoice':
            self.instruction_list = [
                "Kindly assess the picture, question, and response provided. Then, select the accurate explanation from the given choices.",
                "Take a moment to analyze the image, question, and answer. Your task is to identify the correct reasoning from the options provided.",
                "Begin by examining the image, question, and provided answer. Your objective is to pick the right justification from the available options.",
                "Your assignment involves analyzing the image, question, and answer. Subsequently, select the appropriate rationale from the choices below.",
                "Carefully review the image, question, and answer. Your goal is to choose the correct explanation from the options provided.",
            ]
        elif args.infer_method == 'likelihood' or args.formulation == 'Generation':
            self.instruction_list = [
                "Take a look at the image, question, and answer, and afterwards, offer an explanation for your chosen response.",
                "After examining the image, question, and answer, explain the reasoning behind your selected response.",
                "Analyze the image, question, and answer, and then provide the rationale that supports your chosen response.",
                "Evaluate the image, question, and answer, and subsequently give the reasoning behind your provided response.",
                "Carefully assess the image, question, and answer, and then provide the justification for your selected response."
            ]
        
        self.in_context_history = [
            # {'from': 'human', 'value': 'can you see the image? Options: (A) yes; (B) no; (C) not sure; (D) maybe.'},
            {'from': 'human', 'value': 'can you see the image? Answer: yes Options: (A) There is an image as input which is successfully loaded; (B) There is no image; (C) The image is masked; (D) The image failed to load.'},
            {'from': 'assistant', 'value': '(A) There is an image as input which is successfully loaded'}
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
            if self.args.hf:
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
                'rationale_options': item['rationale_options']
            }
            self.samples.append(current_sample)

    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        answer = new_sample['answer']
        if self.args.hf == True:
            image = base64_to_image(new_sample['image'])
            new_sample['image'] = image 
        else:
            image = get_image(new_sample['image'])
            new_sample['image'] = image   
            
        if self.args.shuffle_options:
            valid_options, rationale = random_options(new_sample['rationale_options'], new_sample['rationale'])
        else:
            valid_options, rationale = get_options(new_sample['rationale_options'], new_sample['rationale']) 
        if self.args.formulation == 'SingleChoice':
            new_sample['answer_options'] = valid_options
        elif self.args.formulation == 'Generation':
            new_sample.pop('answer_options')
        new_sample['answer'] = str(rationale)
        
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
        
        new_sample['question'] = new_sample['question'] + f' Answer: {answer}' 
        
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
        valid_options, rationale = random_options(new_sample['rationale_options'], new_sample['rationale'])
        new_sample['answer_options'] = valid_options
        new_sample['answer'] = str(rationale)
        new_sample['instruct'] = random.choice(self.instruction_list)
        return new_sample

    def __len__(self):
        return len(self.samples) * self.duplication



def get_vqar(args, config, formulation, preprocessor):
    if formulation in ['SingleChoice', 'Generation']:
        if config is None:
            return VQAR_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return VQAR_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))


if __name__=='__main__':
    ds = VQAR_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
