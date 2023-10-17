import json
import os
from torch.utils.data import Dataset , DataLoader
from utils.data_utils import base64_to_image
from datasets import load_dataset
from PIL import Image
import pickle
import random
import yaml
import argparse
import random

class Flowers102_Dataset(Dataset):
    def __init__(self , args , config='build/configs/ImageClassification_flowers102_val.yaml' , proc=None , duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config , 'r') , Loader=yaml.Loader)
        else:
            self.config = config
        
        if self.config['instruct_kind'] == 'single_choice' and args.infer_method == 'generation':
            # Single-choice oriented instruction , but we will need different "next_word" in different infer_methods
            self.instruction_list = [
                'Below I will give a question, a picture and options, which are ABCD or 1234, I need you to output only the correct option(such as "(A)" or "(1)")',
                'To answer the question, you can evaluate the picture and choose the accurate answer among the given options.',
                'From the options provided, choose the answer option that fits the question and image formatted as "(A)" or "(1)".',
                'Select the best answer choice for the given question and image, output the correct option(such as "(A)" or "(1)")',
                'Please select the appropriate response option by assessing the question, image, and provided choices. Output the correct option in the format of "(A)" or "(1)".'
            ]
        elif self.config['instruct_kind'] == 'vqa' or (self.config['instruct_kind'] == 'single_choice' and args.infer_method == 'likelihood'):
            # VQA or likelihood mode oriented instruction
            self.instruction_list = [
                'Please evaluate this image and offer a response to the question.',
                'Take a look at this image and give your thoughts on the question.', 
                'Assess this image and provide an answer to the question.',
                'Please investigate this image and share your response to the question.',
                'Your task is to analyze this picture and respond to the question.'
            ]
        else:
            raise ValueError('Wrong instruction kind , you need to check the .yaml file !!!')

        self.in_context_history = [
            {'from': 'human', 'value': 'can you see the image? Options: (A) yes; (B) no; (C) maybe; (D) not sure.'},
            {'from': 'assistant', 'value': '(A) yes'}
        ]
        answer_prefix = getattr(proc, 'response_prefix', None)
        if answer_prefix is not None:
            # need prefix in the in-context sample
            self.in_context_history[1]['value'] = "{} {}".format(answer_prefix, self.in_context_history[1]['value'])

        if duplication > 1:
            # the data sample is repeated with different prompts
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number(5) of different prompts"
        
        self.args = args
        self.proc = proc
        self.duplication = duplication
        self.image_dir = self.config['data_config']['image_dir']
        if self.args.hf:
            self.anns_path = self.config['data_config']['hf_anns_path']
            anns = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.anns_path}, split='test')
        elif self.args.offline_hf:
            self.anns_path = self.config['data_config']['offline_huggingface_anns']
            anns = load_dataset("json",data_files={'test':self.anns_path}, split='test')
        else:
            self.anns_path = self.config['data_config']['anns_path']
            anns = json.load(open(self.anns_path , 'r'))

        # start to generate the data
        self.data = []
        for ann in anns:
            item = {
                'sample_id' : ann['id'],
                'image_name' : ann['image'],
                'question' : ann['question'],
                'answer' : ann['answer'],
                'options' : ann['options']
            }

            self.data.append(item)  
        
    def __getitem__(self , idx):
        sample_index = idx // self.duplication
        data_item = self.data[sample_index]
            
        if self.args.shuffle_options:
            origin_ans = data_item['options'][data_item['answer']]
            random.shuffle(data_item['options'])
            data_item['answer'] = data_item['options'].index(origin_ans)
        
        if self.args.hf or self.args.offline_hf:
            image = base64_to_image(data_item['image_name'])
        else:
            image = os.path.join(self.image_dir , data_item['image_name'])
        
        sample = {
            'sample_id' : data_item['sample_id'],
            'image' : image,
            'question' : data_item['question'],
            'answer' : data_item['answer'],
            'answer_options': data_item['options'],
        }

        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = idx % self.duplication
            sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        else:
            sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]

        # if self.duplication > 1:
        #     # iterate through all possible prompt
        #     inner_sample_index = idx % self.duplication
        #     sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        # else:
        #     if self.args.random_instruct:
        #         # randomly choose one prompt
        #         sample['instruct'] = random.choice(self.instruction_list)
        #     else:
        #         sample['instruct'] = self.instruction_list[0]


        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            sample['history'] = [msg for msg in self.in_context_history]

        if self.proc is not None:
            sample['text'] = self.proc(sample)
        
        return sample

    def __len__(self):
        return len(self.data) * self.duplication


# def get_flowers102(args , config , formulation , preprocessor):
#     if formulation == 'SingleChoice':
#         if config is None:
#             return Flowers102_Dataset(args=args , proc=preprocessor , duplication=args.dataset_duplication)
#         else:
#             return Flowers102_Dataset(args=args , config=config , proc=preprocessor , duplication=args.dataset_duplication)
#     else:
#         raise ValueError('current formulation {} is not supported yet'.format(formulation))   


if __name__ == '__main__':
    ds = Flowers102_Dataset(args='' , duplication=1)
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])