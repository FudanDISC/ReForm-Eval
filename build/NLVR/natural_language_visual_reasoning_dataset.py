from typing import Any
import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
from utils.preprocessors import BaseProcessor, SingleChoiceProcessor

class NLVRMatching(Dataset):
    def __init__(self, args, config='datasets/configs/NaturalLanguageVisualReasoning_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config == yaml.load(open(config, 'r'), Loader=yaml.Loader)
        else:
            self.config = config
        self.image_dir = self.config['data_config']['image_path']
        self.args = args
        data = json.load(open(self.config['data_config']['data_path'], 'r'))
        assert data['gt_info'] == {'0':'no','1':'yes'}
        data = data['data'] # retain data
        
        self.in_context_history = [
            {'from': 'human', 'value': 'can you see the image? Options: (A) yes; (B) no.'},
            {'from': 'assistant', 'value': '(A) yes'}
        ]
        self.instruction_list = [
            "Decide if the sentence '{}' correctly describes the geometric relationships of objects in a synthesized image.",
            "Determine whether the statement '{}' accurately explains the spatial arrangement of elements in the generated image.",
            "Analyze and decide if the sentence '{}' provides an accurate depiction of how objects are geometrically related in the artificial image.",
            "Your task is to determine if the sentence '{}' appropriately outlines the geometric relationships among objects depicted in the generated image.",
            "Does the sentence '{}' correctly describe the geometric relationships of objects in the synthesized image?",
        ]
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
                current_sample = {'sample_id': i,
                                'image': item['img'],
                                'text': item['anno'],
                                'answer': str(item['gt']), # set to index now
                                'answer_options': ['no','yes'],
                }
                self.samples.append(current_sample)

    def __getitem__(self, index):
        sample_index = index // self.duplication
        new_sample = {k:v for k,v in self.samples[sample_index].items()}
        
        if new_sample['answer_options'] is None:
            raise Exception("current dataset didn't support temporal bootstrap!")
        
        if self.args.in_context_sample:
            new_sample['history'] = [msg for msg in self.in_context_history]
        
        if self.duplication > 1:
            # iterate through all possible prompt
            inner_sample_index = index % self.duplication
            new_sample['question'] = self.instruction_list[inner_sample_index % len(self.instruction_list)].format(new_sample['text'])
        else:
            # randomly choose one prompt
            new_sample['question'] = random.choice(self.instruction_list).format(new_sample['text'])
        if self.proc is not None:
            new_sample['text'] = self.proc(new_sample)
        
        return new_sample
    
    def rawitem(self, index):
        pass
    
    def __len__(self):
        return len(self.samples) * self.duplication

class NLVRSelection(Dataset):
    pass



if __name__ == "__main__":
    pass