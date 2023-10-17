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
from utils.data_utils import get_image, base64_to_image
from datasets import load_dataset

def random_options(options, answer, n=4):
    neg_options = [opt for opt in options if opt != answer]
    random.shuffle(neg_options)
    valid_options = neg_options[:n-1]
    valid_options.append(answer)
    random.shuffle(valid_options)
    return valid_options, valid_options.index(answer)

class VisualDialog_SingleChoice(BaseDataset):
    # the single-choice version of the visual Dialog dataset
    def __init__(self, args, config='datasets/configs/VisDial_val.yaml', proc=None, duplication=1):
        logging.info('Loading the Visual Dialog from {}'.format(config))
        self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        logging.info('The data config is: {}'.format(json.dumps(self.config)))
        self.image_path = self.config['data_config']['image_path']
        if args.hf == True:
            data = load_dataset("Aweminus/ReForm-Eval-Data", data_files={'test': self.config['data_config']['huggingface_data']}, split='test')
        elif args.offline_hf:
            data = load_dataset("json", data_files={'test': self.config['data_config']['offline_huggingface_data']}, split='test')
        else:
            data = json.load(open(self.config['data_config']['data_path'], 'r'))

        assert data['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
        assert data['split'] == self.config['split'], 'the data split ({}) and the config split ({}) does not match, please check'.format(data['split'], self.config['split'])
        data = data['data']
        if args.capitalize:
            questions_list = [ques.capitalize() if ques.endswith('?') else ques.capitalize()+'?' for ques in data['questions']]
            self.answer_list = [ans.capitalize() for ans in data['answers']]
        else:
            questions_list = [ques if ques.endswith('?') else ques+'?' for ques in data['questions']]
            self.answer_list = [ans for ans in data['answers']]
        if args.infer_method == 'generation':
            self.instruction_list = [
                'Answer the following questions based on the image and the conversation history.',
                'Select the correct option for the questions by referring to the provided image and dialogue history.',
                'Utilize the content of the image and conversation to infer the answers to the questions.',
                'Based on the image and previous conversation, answer the questions with the provided options.',
                'Respond to the following questions according to the image and the dialogue history.'
            ]
            args.options_in_history = True
        elif args.infer_method == 'likelihood':
            self.instruction_list = [
                'Answer the following questions based on the image and the conversation history.',
                'Provide answers to the questions by referring to the provided image and dialogue history.',
                'Utilize the content of the image and conversation to infer the answers to the questions.',
                'Based on the image and previous conversation, answer the questions.',
                'Respond to the following questions according to the image and the dialogue history.'
            ]
            args.options_in_history = False

        if args.capitalize:
            self.in_context_history = [
                {'from': 'human', 'value': 'Can you see the image? Options: (A) Yes; (B) No; (C) Maybe; (D) Not sure.'},
                {'from': 'assistant', 'value': '(A) Yes'}
            ]
        else:
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
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number of different prompts"
        
        self.samples = []
        self.proc = proc
        self.duplication = duplication
        self.sample2history = defaultdict(list)
        self.sample2update = defaultdict(list)
        self.already_updated = defaultdict(bool)
        self.sample_id2index = dict()
        self.index_info = []
        self.args = args
        
        if args.dataset_subsample is not None:
            num_samples = args.dataset_subsample
        else:
            num_samples = len(data['dialogs'])
        for i, item in tqdm.tqdm(enumerate(data['dialogs'][:num_samples]), desc='preprocessing the data file'):
            if 'COCO' in self.config['split']:
                image_path = os.path.join(self.image_path, '{}_{:012d}.jpg'.format(self.config['split'], item['image_id']))
            else:
                image_path = os.path.join(self.image_path, 'VisualDialog_{}_{:012d}.jpg'.format(self.config['split'], item['image_id']))
            assert os.path.exists(image_path), 'the image {} does not exist, please check'.format(image_path)
            # current_image = {'image': image_path, 'image_caption': item['caption']}
            history = []
            for j, round in enumerate(item['dialog']):
                # if j > 2:
                #     break
                question_id = '{}_{}'.format(item['image_id'], j)
                # tmp_history = [hi for hi in history]
                question = questions_list[round['question']]
                if args.options_in_history:
                    # need to put the options in the history
                    assert proc is not None, 'to put options in the history, a preprocessor is required'
                    full_question, full_answer = proc.process_qa(question=question, options=[self.answer_list[ans] for ans in round['answer_options']],\
                                                                  answer=self.answer_list[round['answer']])
                    history.append({'from': 'human', 'value': full_question})
                    history.append({'from': 'assistant', 'value': full_answer})
                else:
                    # put questions and raw answers in the dialog history
                    history.append({'from': 'human', 'value':question})
                    history.append({'from': 'assistant', 'value': self.answer_list[round['answer']]})
                current_sample = {'sample_id': question_id,
                                  'round_id': j,
                                  # 'history': tmp_history,
                                  'image': image_path,
                                  # 'image_caption': item['caption'],
                                  'question': question,
                                  'answer': round['answer'],
                                  'answer_options': round['answer_options']}
                self.sample_id2index[question_id] = len(self.samples)
                self.index_info.append(['{}_{}'.format(j, item['image_id']), len(self.samples)])
                self.samples.append(current_sample)
                if j == 0:
                    self.already_updated[question_id] = True
            self.sample2history[str(item['image_id'])] = history
        # the order of index_info to ensure the prior rounds are processed before
        self.index_info = sorted(self.index_info, key=lambda x: x[0])
    
    def __getitem__(self, index):
        sample_index = index // self.duplication
        if self.args.online_multi_round:
            sample_index = self.index_info[sample_index][1]
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        if self.args.online_multi_round:
            if not self.already_updated[new_sample['sample_id']] and self.args.local_rank == 0:
                print(self.sample2update[new_sample['sample_id']], new_sample['sample_id'], index, sample_index)
            assert self.already_updated[new_sample['sample_id']], 'the history of the current sample {} is not updated yet'.format(new_sample['sample_id'])

        image_id, round_id = new_sample['sample_id'].split('_')
        if self.args.in_context_sample:
            new_sample['history'] = [msg for msg in self.in_context_history] + self.sample2history[image_id][:2*int(round_id)]
        else:
            new_sample['history'] = self.sample2history[image_id][:2*int(round_id)]
        # valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        new_sample['answer'] = new_sample['answer_options'].index(new_sample['answer'])
        new_sample['answer_options'] = [self.answer_list[i] for i in new_sample['answer_options']]
        
        self.randomness_control(new_sample, index)

        if self.args.hf == True or self.args.offline_hf:
            image = base64_to_image(new_sample['image'])
            new_sample['image'] = image 
        else:#统一读取图片
            image = get_image(new_sample['image'])
            new_sample['image'] = image 
        # if self.duplication > 1:
        #     # iterate through all possible prompt
        #     inner_sample_index = index % self.duplication
        #     new_sample['instruct'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        # else:
        #     # randomly choose one prompt
        #     new_sample['instruct'] = random.choice(self.instruction_list)
        # del new_sample['instruct']
        if self.proc is not None:
            # print(new_sample)
            new_sample['text'] = self.proc(new_sample)
            # print(new_sample['text'])
        # del new_sample['history']
        return new_sample
    
    def rawitem(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        valid_options, answer = random_options(new_sample['answer_options'], new_sample['answer'])
        new_sample['answer_options'] = [self.answer_list[i] for i in valid_options]
        new_sample['answer'] = str(answer)
        new_sample['instruct'] = random.choice(self.instruction_list)
        return new_sample

    def __len__(self):
        return len(self.samples) * self.duplication

    def index2round(self, index):
        # return the number of rounds for a specific sample
        return self.samples[index // self.duplication]['round_id']
    
    def update_history(self, history_infos):
        # update the dialogue history
        for batch in history_infos:
            for item in batch:
                sample_id, pred = item
                self.sample2update[sample_id].append(pred)
                assert(len(self.sample2update[sample_id])<= self.duplication)
                if len(self.sample2update[sample_id]) == self.duplication:
                    id, round_id = sample_id.split('_')
                    round_id = int(round_id)
                    most_common_response = Counter(self.sample2update[sample_id]).most_common(1)[0][0]
                    if self.args.options_in_history:
                        target_sample = self.samples[self.sample_id2index[sample_id]]
                        full_question, full_answer = self.proc.process_qa(target_sample['question'], [self.answer_list[ans] for ans in target_sample['answer_options']]\
                                                                          , most_common_response)
                        self.sample2history[id][2*round_id+1]['value'] = full_answer
                        self.sample2history[id][2*round_id]['value'] = full_question
                    else:
                        self.sample2history[id][2*round_id+1]['value'] = most_common_response
                    del(self.sample2update[sample_id])
                    self.already_updated['{}_{}'.format(id, round_id+1)] = True

        return True


if __name__=='__main__':
    ds = VisualDialog_SingleChoice(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])
