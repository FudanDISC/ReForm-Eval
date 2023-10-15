import torch
import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import logging
from collections import defaultdict, Counter

def random_options(options, answer):
    neg_options = [opt for opt in options if opt != answer]
    random.shuffle(neg_options)
    # valid_options = neg_options[:n-1] if n < len(neg_options) + 1 else neg_options
    valid_options = neg_options
    valid_options.append(answer)
    random.shuffle(valid_options)
    return valid_options, valid_options.index(answer)

class Matching_SingleChoice(Dataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/Matching_winoground_val.yaml', proc=None, duplication=1):
        logging.info('Loading the Winoground from {}'.format(config))
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
                "Take a moment to examine this image and choose the correct option for the question.",
                "Your task is to analyze this image and then pick the right choice for the question.",
                "Kindly assess the image and make the appropriate selection in response to the question.",
                "Begin by evaluating this image and then indicate the correct option for the question.",
                "Please review this image and select the appropriate answer for the question."
            ]
        elif args.infer_method == 'likelihood' or args.formulation == 'Generation':
            self.instruction_list = [
                "Kindly examine the image and provide a response to the question.",
                "Take a moment to analyze the picture and then respond to the question.",
                "Your task is to assess the image and offer an answer to the question.",
                "Begin by analyzing the image and then proceed to answer the question.",
                "Please review the image and then provide your response to the question."
            ]
        
        self.in_context_history = [
            {'from': 'human', 'value': 'can you see the image? Options: (A) yes; (B) no.'},
            {'from': 'assistant', 'value': '(A) yes'}
        ]
         
        if duplication > 1:
            # the data sample is repeated with different prompts
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number of different prompts"
        
        self.samples = []
        self.proc = proc
        self.duplication = duplication
        
        for i, item in tqdm.tqdm(enumerate(data), desc='preprocessing the data file'):
            image_name = '{}.png'.format(item['image_id'])
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

        valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        if self.args.formulation == 'SingleChoice':
            new_sample['answer_options'] = valid_options
        elif self.args.formulation == 'Generation':
            new_sample.pop('answer_options')
        new_sample['answer'] = str(answer)
        
        if self.duplication > 1:
            # iterate through all possible prompt
            inner_sample_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        else:
            # randomly choose one prompt
            new_sample['instruct'] = random.choice(self.instruction_list)
        

        new_sample['question'] = 'Does this text accurately describe this image?'
        
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



def get_matching(args, config, formulation, preprocessor):
    if formulation in ['SingleChoice', 'Generation']:
        if config is None:
            return Matching_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return Matching_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))


if __name__=='__main__':
    ds = Matching_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
