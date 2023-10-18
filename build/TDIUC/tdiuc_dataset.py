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

"""
The original question and answer will be transformed according to the question type, 
the question will be transformed into a single choice, and the answer will be replaced with the 
corresponding option
"""
def make_choices(question_type , answer , choice_path , hf , offline_hf):
    yes_or_no = ['yes' , 'no']
    if answer in yes_or_no:
        return yes_or_no
    
    if hf:
        choice_list = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':os.path.join(choice_path , f'{question_type}.json')}, split='test')['choice'][0]
    elif offline_hf:
        choice_list = load_dataset("json",data_files={'test':os.path.join(choice_path , f'{question_type}.json')}, split='test')['choice'][0]
    else:
        with open(os.path.join(choice_path , f'{question_type}.pkl') , 'rb') as f:
            choice_list = pickle.load(f)
    
    # In addition to the correct answer, 3 other choices are drawn
    try:
        choice_list.remove(answer)
    except ValueError:
        print(f"element {answer} is not in the list")

    generated_choices = random.sample(choice_list , 3)
    generated_choices.append(answer)

    return generated_choices


class TDIUC_Dataset(Dataset):
    def __init__(self , args , config='build/configs/TDIUC.yaml' , proc=None , duplication=1 , task_kind=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
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

        self.instruction_list4absurd = [
            'Is the question answerable based on the image\'s content?',
            'Can the question be answered using the content of the image?',
            'Is it possible to provide an answer using the information in the image?',
            'Does the image\'s content allow for a feasible answer to the question?',
            'Is the question capable of being addressed through the image\'s content?',
        ]

        self.in_context_history = [
            {'from': 'human', 'value': 'can you see the image? Options: (A) yes; (B) no; (C) maybe; (D) not sure.'},
            {'from': 'assistant', 'value': '(A) yes'}
        ]

        if duplication > 1:
            # the data sample is repeated with different prompts
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number(5) of different prompts"
         
        self.args = args
        self.proc = proc
        self.duplication = duplication
        self.image_path = self.config['data_config']['image_path']
        self.task_kind = task_kind
        if self.args.hf:
            self.choice_path = self.config['data_config']['hf_choice_path']
            question_path = self.config['data_config']['hf_question_path']
            anns_path = self.config['data_config']['hf_anns_path']
            questions = sorted(
                # load_dataset("Aweminus/ReForm-Eval",data_files={'test':question_path}, split='test')[0],
                load_dataset("Aweminus/ReForm-Eval",data_files={'test':question_path}, split='test'),
                key=lambda x : x['question_id']
            )
            annotations = sorted(
                # load_dataset("Aweminus/ReForm-Eval",data_files={'test':anns_path}, split='test')[0],
                load_dataset("Aweminus/ReForm-Eval",data_files={'test':anns_path}, split='test'),
                key=lambda x : x['question_id']
            )
        elif self.args.offline_hf:
            self.choice_path = self.config['data_config']['offline_huggingface_choice']
            question_path = self.config['data_config']['offline_huggingface_question']
            anns_path = self.config['data_config']['offline_huggingface_anns']
            questions = sorted(
                # load_dataset("json",data_files={'test':question_path}, split='test')[0],
                load_dataset("json",data_files={'test':question_path}, split='test'),
                key=lambda x : x['question_id']
            )
            annotations = sorted(
                # load_dataset("json",data_files={'test':anns_path}, split='test')[0],
                load_dataset("json",data_files={'test':anns_path}, split='test'),
                key=lambda x : x['question_id']
            )  
        else:
            self.choice_path = self.config['data_config']['choice_path']
            question_path = self.config['data_config']['question_path']
            anns_path = self.config['data_config']['anns_path']
            questions = sorted(
                json.load(open(question_path))['questions'],
                key=lambda x : x['question_id']
            )
            annotations = sorted(
                json.load(open(anns_path))['annotations'],
                key=lambda x : x['question_id']
            )

        all_data = []
        for question , annotation in zip(questions , annotations):
            data = {
                'question_id' : question['question_id'],
                'image_id' : question['image_id'],
                'question' : question['question'],
                'question_type' : annotation['question_type'],
                'answer' : annotation['answers'],
                'source' : annotation['ans_source']
            }
            all_data.append(data)

        if self.task_kind == 1:
            self.data = [item for item in all_data if item['question_type'] == 'color']
        elif self.task_kind == 2:
            self.data = [item for item in all_data if item['question_type'] == 'object_presence']
        elif self.task_kind == 3:
            self.data = [item for item in all_data if item['question_type'] == 'object_recognition']
        elif self.task_kind == 4:
            self.data = [item for item in all_data if item['question_type'] == 'scene_recognition']
        elif self.task_kind == 5:
            self.data = [item for item in all_data if item['question_type'] == 'counting']
        elif self.task_kind == 6:
            self.data = [item for item in all_data if item['question_type'] == 'sentiment_understanding']
        elif self.task_kind == 7:
            self.data = [item for item in all_data if item['question_type'] == 'positional_reasoning']
        elif self.task_kind == 8:
            self.data = [item for item in all_data if item['question_type'] == 'utility_affordance']
        elif self.task_kind == 9:
            self.data = [item for item in all_data if item['question_type'] == 'sport_recognition']
        elif self.task_kind == 10:
            self.data = [item for item in all_data if item['question_type'] == 'attribute']
        elif self.task_kind == 11:
            self.data = [item for item in all_data if item['question_type'] == 'activity_recognition']
        elif self.task_kind == 12:
            self.data = [item for item in all_data if item['question_type'] == 'absurd']
        elif self.task_kind == 0:
            raise ValueError('Can not use all data in multi-choice eval !!!!') 
        else:
            raise ValueError('Wrong task type !!!') 
        
        if self.args.hf != True and self.args.offline_hf != True:
            # get 1/100 data for scene color detection and counting
            # get 1/40 for sport and position
            if self.task_kind in [7 , 9]:
                self.sample_data = self.data[::40]
            elif self.task_kind == 8:
                self.sample_data = self.data
            else:
                self.sample_data = self.data[::100]
            
    def __getitem__(self, idx):
        sample_index = idx // self.duplication
        data_item = self.sample_data[sample_index]
        question = data_item['question']

        if self.args.hf or self.args.offline_hf:
            image = base64_to_image(data_item['image_id'])
        else:
            image_filename = str(data_item['image_id'])
            image_filename = 'COCO_val2014_%s.jpg' %(image_filename.rjust(12,'0'))
            image = os.path.join(self.image_path , image_filename)

        # Treated formulation as multi-choice
        answer = data_item['answer'][0]['answer']
        if self.task_kind == 1:
            choice_list = make_choices('color' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 2:
            choice_list = make_choices('object_presence' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 3:
            choice_list = make_choices('object_recognition' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 4:
            choice_list = make_choices('scene_recognition' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 5:
            choice_list = make_choices('counting' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 6:
            choice_list = make_choices('sentiment_understanding' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 7:
            choice_list = make_choices('positional_reasoning' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 8:
            choice_list = make_choices('utility_affordance' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 9:
            choice_list = make_choices('sport_recognition' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 10:
            choice_list = make_choices('attribute' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 11:
            choice_list = make_choices('activity_recognition' , answer , self.choice_path , self.args.hf , self.args.offline_hf)
        elif self.task_kind == 12:
            # The answer set has only one 'doesnotapply' to give special treatment here
            choice_list = ['yes' , 'no']
            answer = 'yes'
        elif self.task_kind == 0:
            raise ValueError('Can not use all data in multi-choice eval !!!!') 
        else:
            raise ValueError('Wrong question type !!!!') 
        
        if self.args.shuffle_options:
            random.shuffle(choice_list)

        sample = {
            'sample_id' : str(data_item['question_id']),
            'image' : image,
            'question' : question,
            'answer' : choice_list.index(answer),
            'answer_options': choice_list
        }

        if self.task_kind != 12:
            if self.args.random_instruct:
                assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
                instruct_index = idx % self.duplication
                sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
            else:
                sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]
        else: # not these task data in our paper and val_data
            # for absurd task
            if self.args.random_instruct:
                assert (self.duplication < len(self.instruction_list4absurd)) or (self.duplication % len(self.instruction_list4absurd)==0)
                instruct_index = idx % self.duplication
                sample['instruct'] = self.instruction_list4absurd[instruct_index % len(self.instruction_list4absurd)]
            else:
                sample['instruct'] = self.instruction_list4absurd[sample_index % len(self.instruction_list4absurd)]

        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            sample['history'] = [msg for msg in self.in_context_history]

        if self.proc is not None:
            sample['text'] = self.proc(sample)
    
        sample['question_with_option'] = question_with_options(sample, option_mark=self.args.option_mark)

        return sample

    
    def __len__(self):
        return len(self.sample_data) * self.duplication

        
if __name__ == '__main__':
    ds = TDIUC_Dataset(args='' , duplication=5)
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
