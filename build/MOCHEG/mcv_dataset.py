import torch
import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
import logging
from collections import defaultdict, Counter
from utils.data_utils import base64_to_image, get_image, question_with_options
from datasets import load_dataset

refined_answers = {
    'supported': 'yes',
    'refuted': 'no',
    'not enough information': 'not sure'
}

def random_options(options, answer):
    neg_options = [opt for opt in options if opt != answer]
    random.shuffle(neg_options)
    # valid_options = neg_options[:n-1] if n < len(neg_options) + 1 else neg_options
    valid_options = neg_options
    valid_options.append(answer)
    random.shuffle(valid_options)
    return valid_options, valid_options.index(answer)

def get_options(options, answer):
    return options, options.index(answer)

class MCV_SingleChoice(Dataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/MCV_mocheg_val.yaml', proc=None, duplication=1):
        logging.info('Loading the MCV from {}'.format(config))
        self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        logging.info('The data config is: {}'.format(json.dumps(self.config)))
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        if args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
            data = data[0]
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['data_config']['offline_huggingface_data']}, split='test')
            data = data[0]
        else: 
            data = json.load(open(self.config['data_config']['data_path'], 'r'))
        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        data = data['data']
        if args.infer_method == 'generation' and args.formulation == 'SingleChoice':
            self.instruction_list = [
                "Utilizing the textual and visual proof, confirm the assertion by choosing the accurate response.",
                "After examining the text and image evidence, validate the claim by selecting the appropriate choice.",
                "Evaluate the claim's authenticity through the provided text and image evidence, then choose the correct option.",
                "Using the information from both the text and image, determine the accuracy of the claim by selecting the right answer.",
                "Assess the claim's validity based on the text and image evidence, and then pick the correct option.",
                # "Verify the claim's accuracy by considering both the text and image evidence, and then choose the appropriate response.",
                # "After analyzing the text and image evidence, confirm the claim's truth by selecting the right option.",
                # "Based on the evidence from both the text and image, establish the claim's veracity by choosing the correct answer.",
                # "By reviewing the text and image evidence, ascertain the claim's authenticity through the correct option.",
                # "Examine the text and image evidence to confirm the claim's accuracy, then select the proper response.",
            ]
        elif args.infer_method == 'likelihood' or args.formulation == 'Generation':
            self.instruction_list = [
                "Based on the text and image evidence, predict the truthfulness of the claim.",
                "Using textual and visual evidence, make an educated guess about the accuracy of the statement.",
                "Assess the credibility of the assertion by analyzing both the text and image proof.",
                "Evaluate the validity of the claim by considering both textual and visual evidence.",
                "Determine the likelihood of the statement being true based on the provided text and image.",
                # "Examine the claim's veracity using the information presented in both text and image formats.",
                # "Make a judgment about the claim's truthfulness by analyzing the supporting text and image.",
                # "Based on the available textual and visual information, estimate the accuracy of the assertion.",
                # "Using the text and image data, speculate on the honesty of the presented claim.",
                # "Consider both the textual and visual evidence to make a prediction about the claim's truth.",
                # "By reviewing both the text and image evidence, form a projection of the claim's accuracy.",
                
            ]

        self.in_context_history = [
            {'from': 'human', 'value': 'What is the truthfulness of this claim? Claim: You can see the image. Text evidence: There is an image. Options: (A) supported; (B) refuted (C) not enough information.'},
            {'from': 'assistant', 'value': '(A) supported'}
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
            if self.args.hf or self.args.offline_hf:
                image_path = item['image_id']
            else:
                image_name = '{}'.format(item['image_id'])
                image_path = os.path.join(self.image_path, image_name)
                assert os.path.exists(image_path), 'the image {} does not exist, please check'.format(image_path)

            claim = item['claim'].strip()
            text_evidence = item['text_evidence'].strip()
            # question = f'claim is "{claim}", text evidence is "{text_evidence}", what is the truthfulness of this claim?'
            question = f'What is the truthfulness of this claim? Claim: {claim} Text Evidence: {text_evidence}'
            current_sample = {
                'sample_id': item['instance_id'],
                'image': image_path,
                'question': question,
                'answer': item['answer'],
                'answer_options': item['answer_options'],
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
        if self.args.shuffle_options:
            valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        else:
            valid_options, answer = get_options(new_sample['answer_options'], new_sample['answer']) 
        if self.args.formulation == 'SingleChoice':
            new_sample['answer_options'] = valid_options
        elif self.args.formulation == 'Generation':
            new_sample.pop('answer_options')
        else:
            return 
        new_sample['answer'] = str(answer)
        
        # if self.duplication > 1:
        #     # iterate through all possible prompt
        #     inner_sample_index = index % self.duplication
        #     new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        # elif self.args.random_instruct:
        #     # randomly choose one prompt
        #     new_sample['instruct'] = random.choice(self.instruction_list)
        # else:
        #     new_sample['instruct'] = self.instruction_list[0]
        
        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = index % self.duplication
            new_sample['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        else:
            sample_index = index // self.duplication
            new_sample['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]
        
        if self.args.infer_method == 'likelihood':
            new_sample['instruct'] = new_sample['instruct'] + ' Please answer supported, refuted or not enough information.'
        
        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
            new_sample['history'] = [msg for msg in self.in_context_history]
        
        new_sample['question_with_option'] = question_with_options(new_sample, option_mark=self.args.option_mark)
        
        
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



def get_mcv(args, config, formulation, preprocessor):
    if formulation in ['SingleChoice', 'Generation']:
        if config is None:
            return MCV_SingleChoice(args=args, proc=preprocessor, duplication=args.dataset_duplication)
        else:
            return MCV_SingleChoice(args=args, config=config, proc=preprocessor, duplication=args.dataset_duplication)
    else:
        raise ValueError('current formulation {} is not supported yet'.format(formulation))


if __name__=='__main__':
    ds = MCV_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
