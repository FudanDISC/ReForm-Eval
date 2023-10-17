import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import numpy as np
import cv2
from PIL import Image
from utils.data_utils import get_image, base64_to_image
from datasets import load_dataset

def random_options(options, answer):
    ori_answer = options[answer]
    random.shuffle(options)
    return options, options.index(ori_answer)

def draw_bbox(img, bboxs, is_label=False):
    # for IC-15 the annotation is two points
    img = np.array(img)
    thickness = 2 # Line thickness of 2 px
    colors =[(255,0,0),(0,128,0), (0,0,255),(255,215,0)] # red
    back_color = (255, 255, 255) # white
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    for i in range(len(bboxs)):
        text = f'<{i+1}>'
        text_size, baseLine = cv2.getTextSize(text, font, font_scale, thickness)
        x, y, w, h = bboxs[i]
        x1, y1, x2, y2 = int(x), int(y), int(w), int(h)
        
        start_point = (x1, y1)
        end_point = (x2, y2)
        img = cv2.rectangle(img, start_point, end_point, colors[i], thickness)
        if is_label:
            img = cv2.rectangle(img, start_point, (x1 + text_size[0], y1 + text_size[1]+baseLine), back_color, -1)
            img = cv2.putText(img, text, (int(x1), int(y1) + text_size[1]), font, font_scale, colors[i], 1)
    return img

class ReferringExpressionSelection(Dataset):
    def __init__(self, args, config='datasets/configs/ReferringExpression_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), loader=yaml.loader)
        else:
            self.config = config
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        if args.hf:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['huggingface_data']}, split='test')
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['offline_huggingface_data']}, split='test')
        else:
            data = json.load(open(self.config['data_config']['res_path'], 'r'))
        assert data['dataset_name'] == 'RefCOCO'
        data = data['data']

        self.in_context_history = [
            {'from': 'human', 'value': 'What is the shape of this image? Options: (A) rectangle; (B) circle; (C) triangle; (D) hexagon.'},
            {'from': 'assistant', 'value': '(A) rectangle.'}
        ]
        
        answer_prefix = getattr(proc, 'response_prefix', None)
        if answer_prefix is not None:
            # need prefix in the in-context sample
            self.in_context_history[1]['value'] = "{} {}".format(answer_prefix, self.in_context_history[1]['value'])
        
        if args.infer_method == 'generation':
            self.instruction_list = [
                "Select one text to describe the region in the red bounding box in the image. Select your answer from the options.",
                "Which text option most closely corresponds to the red bounding box region in the image? Select your answer from the given selections.",
                "Based on the red bounding box region in the image, select one most matched text from following options.",
                "From the available options, please indicate the text that corresponds most closely with the content contained within the red bounding box in the image.",
                "Which of the provided text choices best matches the area enclosed by the red bounding box in the image? Choose your response from the choices provided.",
            ]
        elif args.infer_method == 'likelihood':
            self.instruction_list = [
                "Give one text answer to describe the region in the red bounding box in the image.",
                "What text closely corresponds to the red bounding box region in the image?",
                "Based on the red bounding box region in the image, give one matched text.",
                "Please give the text answer that corresponds closely with the content contained within the red bounding box in the image.",
                "What text matches the area enclosed by the red bounding box in the image?",
            ]
        else:
            raise Exception("Invalid infer method!")
        if duplication > 1:
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number of different prompts"
        
        self.samples = []
        self.proc = proc
        self.duplication = duplication
        if not self.config['data_config']['load_from_bootstrap']:
            raise Exception("current dataset didn't support temporal bootstrap!")
        else:
            for i, item in tqdm.tqdm(enumerate(data), desc='preprocessing the data file'):
                """
                item is {'img':ab/path,'anno':[str],'gt':int}
                """
                current_sample = {'sample_id': item['id'],
                                'image': item['img'],
                                'bbox': item['region'],
                                'answer': str(item['gt']), # set to index now
                                'answer_options': item['options'],
                }
                self.samples.append(current_sample)
    
    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        
        if self.args.hf or self.args.offline_hf:
            raw_image = base64_to_image(new_sample['image'])
        else:
            raw_image = get_image(new_sample['image'])
        image = draw_bbox(raw_image, new_sample['bbox'])
        new_sample['image'] = Image.fromarray(image).convert('RGB')
        
        if new_sample['answer_options'] is None:
            raise Exception("current dataset didn't support temporal bootstrap!")
        
        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = index % self.duplication
            new_sample['question'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        else:
            new_sample['question'] = self.instruction_list[sample_index % len(self.instruction_list)]
        
        if self.args.shuffle_options:
            shuffled_options, shuffled_answer = random_options(new_sample['answer_options'],new_sample['answer'])
            new_sample['answer_options'], new_sample['answer'] = shuffled_options, shuffled_answer
        
        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            new_sample['history'] = [msg for msg in self.in_context_history]
        
        if self.proc is not None:
            new_sample['text'] = self.proc(new_sample)
    
        return new_sample
    
    def rawitem(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        if self.args.hf or self.args.offline_hf:
            image = base64_to_image(new_sample['image'])
        else:
            image = get_image(new_sample['image'])
        new_sample['image'] = image
        if new_sample['answer_options'] is None:
            raise Exception("current dataset didn't support temporal bootstrap!")
        new_sample['question'] = random.choice(self.instruction_list)
        # draw bbox
        raw_image = Image.open(new_sample['image']).convert("RGB")
        image = draw_bbox(raw_image, [new_sample['bbox']])
        new_sample['image'] = Image.fromarray(image).convert('RGB')
        
        return new_sample
    
    def __len__(self):
        return len(self.samples) * self.duplication
        

if __name__ == "__main__":
    ds = ReferringExpressionSelection(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])