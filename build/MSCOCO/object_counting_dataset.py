import json
import os
from torch.utils.data import Dataset , DataLoader
from utils.data_utils import base64_to_image , get_image , question_with_options
from datasets import load_dataset
from PIL import Image
import pickle
import random
import yaml
import argparse


def make_choices(answer , choice_path , hf , offline_hf):
    if hf:
        choice_list = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':choice_path}, split='test')['choice'][0]
    elif offline_hf:
        choice_list = load_dataset("json",data_files={'test':choice_path}, split='test')['choice'][0]
    else:
        with open(choice_path , 'rb') as f:
            choice_list = pickle.load(f)
    # In addition to the correct answer, 3 other choices are drawn
    try:
        choice_list.remove(int(answer))
    except ValueError:
        print(f"element {answer} is not in the list")

    generated_choices = random.sample(choice_list , 3)
    generated_choices.append(answer)
    # random.shuffle(generated_choices)
    str_choice = [str(item) for item in generated_choices]

    return str_choice


class ObjectCounting_SingleChoice(Dataset):
    def __init__(self , args , config='build/configs/ObjectCounting_mscoco_val.yaml' , proc=None , duplication=1):
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
            self.choice_path = self.config['data_config']['hf_choice_path']
            self.anns_path = self.config['data_config']['hf_anns_path']
            anns = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.anns_path}, split='test')
        elif self.args.offline_hf:
            self.choice_path = self.config['data_config']['offline_huggingface_choice']
            self.anns_path = self.config['data_config']['offline_huggingface_anns']
            anns = load_dataset("json",data_files={'test':self.anns_path}, split='test')
        else:
            self.choice_path = self.config['data_config']['choice_path']
            self.anns_path = self.config['data_config']['anns_path']
            anns = json.load(open(self.anns_path , 'r'))

        self.data = []
        for idx , ann in enumerate(anns):
            item = {
                'sample_id' : str(idx+1),
                'image' : ann['Image_ID'],
                'question' : ann['Question'],
                'answer' : ann['Answer'],
                'COCO_category' : ann['Resolved_COCO_category']
            }

            self.data.append(item)
    
    def __getitem__(self , idx):
        sample_index = idx // self.duplication
        data_item = self.data[sample_index]

        question = data_item['question']
        if self.args.hf or self.args.offline_hf:
            image = base64_to_image(data_item['image'])
        else:
            image = os.path.join(self.image_dir , data_item['image'])
        answer = str(data_item['answer'])
        choice_list = make_choices(answer , self.choice_path , self.args.hf , self.args.offline_hf)

        if self.args.shuffle_options:
            random.shuffle(choice_list)

        sample = {
            'sample_id' : data_item['sample_id'],
            'image' : image,
            'question' : question,
            'answer' : choice_list.index(answer),
            'answer_options': choice_list
        }

        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = idx % self.duplication
            sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        else:
            sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]

        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            sample['history'] = [msg for msg in self.in_context_history]

        if self.proc is not None:
            sample['text'] = self.proc(sample)
        
        sample['question_with_option'] = question_with_options(sample, option_mark=self.args.option_mark)

        return sample
    
    def __len__(self):
        return len(self.data) * self.duplication
    
   
if __name__ == '__main__':
    ds = ObjectCounting_SingleChoice(args='' , duplication=5)
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
