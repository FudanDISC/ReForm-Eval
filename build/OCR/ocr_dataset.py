import torch
import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import numpy as np
from PIL import Image
from utils.data_utils import base64_to_image, get_image
from datasets import load_dataset

class OCR_OpenEnded(Dataset):
    def __init__(self, args, config='datasets/configs/OCR_textvqa_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        else:
            self.config = config
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        
        
        if args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['offline_huggingface_data']}, split='test')
        else: 
            data = json.load(open(self.config['data_config']['data_path'], 'r'))
        
        assert str(data['version']) == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        
        data = data['data']
        # load instruct
        self.instruction_list = [
            "Assess this image and provide an answer to the question.",
            "Take a look at this image and give your thoughts on the question.",
            "Please investigate this image and share your response to the question.",
            "Analyze this image and answer the question.",
            "Your task is to analyze this picture and respond to the question.",
        ]
        
        
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
                if self.config['dataset']  == 'Text-VQA':
                    image_name = '{}.jpg'.format(item['image_id'])
                elif self.config['dataset'] == 'OCRVQA':
                    if os.path.exists(os.path.join(self.image_path, '{}.jpg'.format(item['image_id']))):
                        image_name = '{}.jpg'.format(item['image_id'])
                    else:
                        image_name = '{}.gif'.format(item['image_id'])
                elif self.config['dataset'] == 'DocVQA':
                    image_name = '{}'.format(item['image_id'])
                else:
                    image_name = '{}'.format(item['image_id']) 
                image_path = os.path.join(self.image_path, image_name)
                assert os.path.exists(image_path), 'the image {} does not exist, please check'.format(image_path)

            current_sample = {'sample_id': item['question_id'],
                              'image': image_path,
                              'question': item['question'],
                              'answer': item['answer'],
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
        if self.duplication > 1:
            # iterate through all possible prompt
            inner_sample_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        elif self.args.random_instruct:
            # randomly choose one prompt
            new_sample['instruct'] = random.choice(self.instruction_list)
        else:
            new_sample['instruct'] = self.instruction_list[0]
            
        # if self.args.random_instruct:
        #     assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
        #     instruct_index = index % self.duplication
        #     new_sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        # else:
        #     sample_index = index // self.duplication
        #     new_sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]
            
        if self.proc is not None:
            # print(new_sample)
            new_sample['text'] = self.proc(new_sample)
            # print(new_sample['text'])
        return new_sample
    
    def rawitem(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        new_sample['answer'] = self.samples[sample_index]['answer']
        if self.duplication > 1:
            # iterate through all possible prompt
            inner_sample_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        else:
            # randomly choose one prompt
            new_sample['instruct'] = random.choice(self.instruction_list)
        return new_sample

    def __len__(self):
        return len(self.samples) * self.duplication
    

def get_ocr(args, config, formulation, preprocessor):
    if formulation in ['OCROpenEnded']:
        if config is None:
            return OCR_OpenEnded(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return OCR_OpenEnded(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))