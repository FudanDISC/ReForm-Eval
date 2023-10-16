import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import numpy as np
import cv2
from PIL import Image
from utils.data_utils import base64_to_image,get_image
from datasets import load_dataset
refined_answers = {
    'supported': 'yes',
    'refuted': 'no',
    'not enough information': 'not sure'
}

def get_options(options, answer):
    return options, options.index(answer)

def draw_bbox(img, bboxs, is_label=False):
    img = np.array(img)
    thickness = 3 # Line thickness of 2 px
    colors =[(255,0,0),(0,128,0), (0,0,255),(255,215,0)] # red
    back_color = (255, 255, 255) # white
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    for i in range(len(bboxs)):
        text = f'<region{i+1}>'
        text_size, baseLine = cv2.getTextSize(text, font, font_scale, thickness)
        x, y, w, h = bboxs[i]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        start_point = (x1, y1)
        end_point = (x2, y2)
        img = cv2.rectangle(img, start_point, end_point, colors[i], thickness)
        if is_label:
            img = cv2.rectangle(img, start_point, (x1 + text_size[0], y1 + text_size[1]+baseLine), back_color, -1)
            img = cv2.putText(img, text, (int(x1), int(y1) + text_size[1]), font, font_scale, colors[i], 1)
    return img

def random_options(options, answer_idx):
    answer = options[answer_idx]
    valid_options = [opt for opt in options if opt != answer]
    random.shuffle(valid_options)
    answer_idx = random.randint(0,len(valid_options))
    valid_options.insert(answer_idx, answer)
    return valid_options, answer_idx

class MissingObjectSelection_SingleChoice(Dataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/MissingObjectSelection_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        else:
            self.config = config
        self.image_path = self.config['data_config']['image_path']
        self.args = args

        if args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':config['huggingface_data']}, split='test')
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
                              'question': item['question'],
                              'answer': item['answer'],
                              'answer_options': item['answer_options'],
                              'bbox': item['bbox']
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
                
        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        else:
            sample_index = index // self.duplication
            new_sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]

        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            new_sample['history'] = [msg for msg in self.in_context_history]

        if self.args.hf == True:
            raw_image = base64_to_image(new_sample['image'])
        else:
            raw_image = get_image(new_sample['image'])
        ## draw bbox
        image = draw_bbox(raw_image, new_sample['bbox'],is_label=False)
        new_sample['image'] = Image.fromarray(image).convert('RGB')
        if self.proc is not None:
            # print(new_sample)
            new_sample['text'] = self.proc(new_sample)
            # print(new_sample['text'])
        return new_sample
    
    def __len__(self):
        return len(self.samples) * self.duplication


if __name__=='__main__':
    ds = MissingObjectSelection_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
