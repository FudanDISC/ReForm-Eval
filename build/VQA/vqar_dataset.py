import torch
import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import logging
from collections import defaultdict, Counter
from build.base_dataset import BaseDataset

def random_options(options, answer):
    neg_options = [opt for opt in options if opt != answer]
    random.shuffle(neg_options)
    # valid_options = neg_options[:n-1] if n < len(neg_options) + 1 else neg_options
    valid_options = neg_options
    valid_options.append(answer)
    random.shuffle(valid_options)
    return valid_options, valid_options.index(answer)

class VQARandom_SingleChoice(BaseDataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/VQA_vqav2_val_v2.0.yaml', proc=None, duplication=1):
        logging.info('Loading the VQA from {}'.format(config))
        self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        logging.info('The data config is: {}'.format(json.dumps(self.config)))
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        
        # if 'option_type' in self.config:
        #     option_type = self.config['option_type']
        #     data = json.load(open(self.config['data_config']['data_path'].replace('.json', f'_I{option_type}.json'), 'r'))
        # else:
        data = json.load(open(self.config['data_config']['data_path'], 'r'))
        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        data = data['data']
        if args.infer_method == 'generation' and args.formulation == 'SingleChoice':
            self.instruction_list = [
                'Please analyze the image and the question, then select the correct option.',
                'Take a close look at the image and question, and then choose the correct option.',
                'Examine both the image and the question before selecting the right option.',
                'Carefully analyze the image and question and then pick the correct option.',
                'Evaluate the image and question thoroughly before making your selection.',
                # 'Make sure to analyze both the image and the question, and then choose the correct option.',
                # 'Before answering, carefully consider both the image and the question and select the right option.',
                # 'After analyzing the image and question, select the appropriate option.',
            ]
        elif args.infer_method == 'likelihood' or args.formulation == 'Generation':
            self.instruction_list = [
                "Assess this image and provide an answer to the question.",
                "Take a look at this image and give your thoughts on the question.",
                "Please investigate this image and share your response to the question.",
                "Analyze this image and answer the question.",
                "Your task is to analyze this picture and respond to the question.",
            ]
        
        self.in_context_history = [
            {'from': 'human', 'value': 'can you see the image? Options: (A) yes; (B) no; (C) not sure; (D) maybe.'},
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
            if self.config['dataset'] in ['VQA', 'OK-VQA']:
                image_name =  '{}_{:012d}.jpg'.format(self.config['split'], item['image_id'])
            elif self.config['dataset'] == 'GQA':
                image_name = '{}.jpg'.format(item['image_id'])
            elif self.config['dataset'] == 'A-OKVQA':
                image_name = '{:012d}.jpg'.format(item['image_id'])
            elif self.config['dataset'] == 'Whoops':
                image_name = '{}.png'.format(item['image_id']) 
            elif self.config['dataset'] == 'ScienceQA':
                image_name = '{}/image.png'.format(item['image_id'])
            else:
                image_name = '{}'.format(item['image_id'])
            image_path = os.path.join(self.image_path, image_name)
            assert os.path.exists(image_path), 'the image {} does not exist, please check'.format(image_path)


            current_sample = {
                'sample_id': item['question_id'],
                'image': image_path,
                'question': item['question'],
                'answer': item['answer'],
                'answer_options': item['answer_options']
            }
            self.samples.append(current_sample)

    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}

        new_sample['answer'] = new_sample['answer_options'].index(new_sample['answer'])
        self.randomness_control(new_sample, index)
        
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



def get_vqa_random(args, config, formulation, preprocessor):
    if formulation in ['SingleChoice', 'Generation']:
        if config is None:
            return VQARandom_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return VQARandom_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))


if __name__=='__main__':
    ds = VQARandom_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
