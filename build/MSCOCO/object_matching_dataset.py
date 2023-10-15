import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import numpy as np
import cv2
from PIL import Image

def draw_bbox(img, bboxs, is_label=False):
    img = np.array(img)
    thickness = 2 # Line thickness of 2 px
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

class ObjectMatching_TrueOrFlase(Dataset):
    # the TrueOrFlase version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/ObjectMatching_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        else:
            self.config = config
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        data = json.load(open(self.config['data_config']['true_or_false_path'], 'r'))

        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        # load instruct
        self.instruction_list = data['instructs']
        # load data
        data = data['data']
        
        self.samples = []
        self.proc = proc
        self.duplication = duplication

        for i, item in tqdm.tqdm(enumerate(data), desc='preprocessing the data file'):
            image_path = os.path.join(self.image_path, item['image'])
            assert os.path.exists(image_path), 'the image {} does not exist, please check'.format(image_path)

            current_sample = {'sample_id': item['question_id'],
                              'image': image_path,
                              'question': item['question'],
                              'answer': item['answer'],
                              'answer_options': item['answer_options'],
                              'bbox': item['bbox']
            }
            self.samples.append(current_sample)
    
    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        new_sample['answer_options'] = valid_options
        new_sample['answer'] = str(answer)
        if self.duplication > 1:
            # iterate through all possible prompt
            inner_sample_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        else:
            # randomly choose one prompt
            new_sample['instruct'] = random.choice(self.instruction_list)
        ## draw bbox
        raw_image = Image.open(new_sample['image']).convert("RGB")
        image = draw_bbox(raw_image, new_sample['bbox'],is_label=True)
        new_sample['image'] = Image.fromarray(image).convert('RGB')
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
        if self.duplication > 1:
            # iterate through all possible prompt
            inner_sample_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        else:
            # randomly choose one prompt
            new_sample['instruct'] = random.choice(self.instruction_list)
        ## draw bbox
        raw_image = Image.open(new_sample['image']).convert("RGB")
        image = draw_bbox(raw_image, [new_sample['bbox']])
        new_sample['image'] = Image.fromarray(image).convert('RGB')
        return new_sample

    def __len__(self):
        return len(self.samples) * self.duplication


if __name__=='__main__':
    ds = ObjectMatching_TrueOrFlase(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
