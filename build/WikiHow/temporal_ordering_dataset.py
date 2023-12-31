import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
from utils.preprocessors import BaseProcessor, SingleChoiceProcessor
from utils.data_utils import get_image, base64_to_image, question_with_options
from datasets import load_dataset

def random_options(options, answer):
    ori_answer = options[int(answer)]
    random.shuffle(options)
    return options, options.index(ori_answer)

class WikiHowImageTextSelection(Dataset):
    def __init__(self, args, config='datasets/configs/TemporalOrdering_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config == yaml.load(open(config, 'r'), Loader=yaml.Loader)
        else:
            self.config = config
        self.image_dir = self.config['data_config']['image_path']
        self.args = args
        if args.hf:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
            data = data[0]
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['data_config']['offline_huggingface_data']}, split='test')
            data = data[0]
        else:
            data = json.load(open(self.config['data_config']['wits_path'], 'r'))
        assert data['dataset_name'] == 'WikiHow'
        data = data['data']

        self.in_context_history = [
            {'from': 'human', 'value': 'What is the shape of this image? Options: (A) rectangle; (B) circle; (C) triangle; (D) hexagon;'},
            {'from': 'assistant', 'value': '(A) rectangle;'}
        ]
        answer_prefix = getattr(proc, 'response_prefix', None)
        if answer_prefix is not None:
            # need prefix in the in-context sample
            self.in_context_history[1]['value'] = "{} {}".format(answer_prefix, self.in_context_history[1]['value'])
        
        if self.args.infer_method == 'generation':
            self.instruction_list = [
                "For the task '{}', the image represents one step of the task. Select the text that best matches the image.",
                "For the task '{}', the image showcases a single stage within the task. Choose the text that most accurately corresponds to the image.",
                "The image depicts a specific step of the task '{}'. Indicate the text that correlates most effectively with the image.",
                "Concerning the task '{}', the image symbolizes one particular step of the task. Point out the text that demonstrates the closest resemblance to the image.",
                "Regarding the assigned task '{}', the image exhibits a distinct phase in the task. Select the text that mirrors the image with the highest fidelity.",
            ]
        elif args.infer_method == 'likelihood':
            self.instruction_list = [
                "For the task '{}', the image represents one step of the task. Give the answer that closely describes the image.",
                "For the task '{}', the image showcases a single stage within the task. Give the answer that most accurately corresponds to the image.",
                "The image depicts a specific step of the task '{}'. Answer the text that correlates most effectively with the image.",
                "Concerning the task '{}', the image symbolizes one particular step of the task. Answer the text that demonstrates the closest resemblance to the image.",
                "Regarding the assigned task '{}', the image exhibits a distinct phase in the task. Answer the text that mirrors the image with the highest fidelity.",
            ]
        else:
            raise Exception("Invalid infer method!")
        
        if duplication > 1:
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number of different prompts"
        
        self.samples= []
        self.proc = proc
        self.duplication = duplication
        if not self.config['data_config']['load_from_bootstrap']:
            raise Exception("Don't support temporary bootstrapping now!")
        else:
            for i, item in tqdm.tqdm(enumerate(data), desc='preprocessing the data file'):
                """
                item is {'img':ab/path,'method':str,'title':str,'options':[str],'gt':int}
                """
                current_sample = {
                    'sample_id': i,
                    'image': item['img'],
                    'task': item['method'],
                    'title': item['title'],
                    'answer': str(item['gt']), # set to index now
                    'answer_options': item['options'],
                }
                self.samples.append(current_sample)
        
    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        
        if self.args.hf or self.args.offline_hf:
            image = base64_to_image(new_sample['image'])
        else:
            image = get_image(new_sample['image'])
        new_sample['image'] = image
        
        if new_sample['answer_options'] is None:
            raise Exception("Don't support temporary bootstrpping now!")
                
        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = index % self.duplication
            new_sample['question'] = self.instruction_list[instruct_index % len(self.instruction_list)].format(new_sample['task'])
        else:
            new_sample['question'] = self.instruction_list[sample_index % len(self.instruction_list)].format(new_sample['task'])
        
        if self.args.shuffle_options:
            shuffled_options, shuffled_answer = random_options(new_sample['answer_options'],new_sample['answer'])
            new_sample['answer_options'], new_sample['answer'] = shuffled_options, shuffled_answer
        
        if self.args.in_context_sample and self.args.formulation == 'SingleChoice':
                new_sample['history'] = [msg for msg in self.in_context_history]
        
        if self.proc is not None:
            new_sample['text'] = self.proc(new_sample)
        
        new_sample['question_with_option'] = question_with_options(new_sample, option_mark=self.args.option_mark)
        
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
            raise Exception("current answer option doesn't support improvisation!")
        new_sample['question'] = random.choice(self.instruction_list).format(new_sample['task'])
        new_sample['question_with_option'] = question_with_options(new_sample, option_mark=self.args.option_mark)
        return new_sample
    
    def __len__(self):
        return len(self.samples) * self.duplication
        
        
if __name__ == "__main__":
    proc=SingleChoiceProcessor(' ', '\n', roles=['Question', 'Answer'], infer_method='generation')
    ds = WikiHowImageTextSelection(args='',proc=proc)
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])