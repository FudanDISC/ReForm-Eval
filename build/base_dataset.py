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
    ori_answer = options[answer]
    random.shuffle(options)
    return options, options.index(ori_answer)

class BaseDataset(Dataset):
    def randomness_control(self, item, index):
        # a randomness control function in the dataset, should be used before the preprocessor
        if self.args.random_instruct:
            assert (self.duplication < len(self.instruction_list)) or (self.duplication % len(self.instruction_list)==0)
            instruct_index = index % self.duplication
            item['instruct'] = self.instruction_list[instruct_index % len(self.instruction_list)]
        else:
            sample_index = index // self.duplication
            item['instruct'] = self.instruction_list[sample_index % len(self.instruction_list)]
        if self.args.shuffle_options:
            shuffled_options, shuffled_answer = random_options(item['answer_options'], item['answer'])
            item['answer_options'] = shuffled_options
            item['answer'] = shuffled_answer
        return item

