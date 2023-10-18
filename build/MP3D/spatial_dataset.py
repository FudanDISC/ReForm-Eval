import torch
import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import numpy as np
import cv2
from PIL import Image
from utils.data_utils import get_image, base64_to_image, question_with_options
from datasets import load_dataset
refined_answers = {
    'supported': 'yes',
    'refuted': 'no',
    'not enough information': 'not sure'
}

def get_options(options, answer):
    ## answer is option index
    return options, answer

def random_options(options, answer_idx):
    answer = options[answer_idx]
    valid_options = [opt for opt in options if opt != answer]
    random.shuffle(valid_options)
    answer_idx = random.randint(0,len(valid_options))
    valid_options.insert(answer_idx, answer)
    return valid_options, answer_idx

class Spatial_SingleChoice(Dataset):
    # the TrueOrFlase version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/Spatial_mp3d_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        else:
            self.config = config
        self.image_path = self.config['data_config']['image_path']
        self.args = args

        if args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':config['data_config']['huggingface_data']}, split='test')
            data = data[0]
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['data_config']['offline_huggingface_data']}, split='test')
            data = data[0]
        else:
            data = json.load(open(self.config['data_config']['data_path'], 'r'))
 
        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        # load instruct
        if args.infer_method == 'generation':
            self.instruction_list = [
                'Answer the following questions based on the image and the conversation history.',
                'Select the correct option for the questions by referring to the provided image and dialogue history.',
                'Utilize the content of the image and conversation to infer the answers to the questions.',
                'Based on the image and previous conversation, answer the questions with the provided options.',
                'Respond to the following questions according to the image and the dialogue history.'
            ]
        elif args.infer_method == 'likelihood':
            self.instruction_list = [
                'Answer the following questions based on the image and the conversation history.',
                'Provide answers to the questions by referring to the provided image and dialogue history.',
                'Utilize the content of the image and conversation to infer the answers to the questions.',
                'Based on the image and previous conversation, answer the questions.',
                'Respond to the following questions according to the image and the dialogue history.'
            ]
        # load data
        data = data['data']

        self.in_context_history = [
            {'from': 'human', 'value': 'What is the shape of this image? Options: (A) rectangle; (B) circle; (C) triangle; (D) hexagon; (E) pentagon; (F) heptagon; (G) octagon.'},
            {'from': 'assistant', 'value': '(A) rectangle;'}
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
            current_sample = {'sample_id': item['question_id'],
                              'image': item['image'],
                              'question': item['questions'],
                              'answer': item['answer'],
                              'answer_options': item['answer_options']
            }
            self.samples.append(current_sample)
    
    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        if self.args.shuffle_options:
            valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        else:
            valid_options, answer = get_options(new_sample['answer_options'], new_sample['answer']) 
        new_sample['answer_options'] = valid_options
        new_sample['answer'] = str(answer)
        
        if self.args.hf == True or self.args.offline_hf:
            image = base64_to_image(new_sample['image'])
            new_sample['image'] = image 
        else:
            image = get_image(new_sample['image'])
            new_sample['image'] = image 
        
        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
            assert (self.duplication < len(self.samples[sample_index]['question'])) or (self.duplication % len(self.samples[sample_index]['question'])==0)
            question_index = index % self.duplication
            new_sample['question'] = self.samples[sample_index]['question'][question_index % len(self.samples[sample_index]['question'])]
        else:
            sample_index = index // self.duplication
            new_sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]  
            new_sample['question'] = self.samples[sample_index]['question'][sample_index % len(self.samples[sample_index]['question'])]
            
        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            new_sample['history'] = [msg for msg in self.in_context_history]    
            
        if self.proc is not None:
            # print(new_sample)
            new_sample['text'] = self.proc(new_sample)
            # print(new_sample['text'])
        new_sample['question_with_option'] = question_with_options(new_sample, option_mark=self.args.option_mark)
        return new_sample

    def __len__(self):
        return len(self.samples) * self.duplication



if __name__=='__main__':
    ds = Spatial_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
