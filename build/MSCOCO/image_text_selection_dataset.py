import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
import os
from utils.data_utils import get_image, base64_to_image
from datasets import load_dataset

def random_options(options, answer):
    ori_answer = options[answer]
    random.shuffle(options)
    return options, options.index(ori_answer)

class ImageTextSelection(Dataset):
    def __init__(self, args, config='datasets/configs/ImageTextSelection_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
        else:
            self.config = config
        self.image_path = self.config['data_config']['image_path']
        self.args = args
        if args.hf:
            data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['huggingface_data']}, split='test')
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['offline_huggingface_data']}, split='test')
        else:
            data = json.load(open(self.config['data_config']['its_path'], 'r'))
        if not self.config['data_config']['load_from_bootstrap']:
            assert data['info']['version'] == self.config['version'], 'the data version ({}) and the config version ({}) does not match, please check'.format(data['version'], self.config['version'])
            data = data['annotations']
        else:
            assert data['dataset_name'] == 'MSCOCO'
            data = data['data']

        self.in_context_history = [
            {'from': 'human', 'value': 'What is the shape of this image? Options: (A) rectangle; (B) circle; (C) triangle; (D) hexagon.'},
            {'from': 'assistant', 'value': '(A) rectangle;'}
        ]
        answer_prefix = getattr(proc, 'response_prefix', None)
        if answer_prefix is not None:
            # need prefix in the in-context sample
            self.in_context_history[1]['value'] = "{} {}".format(answer_prefix, self.in_context_history[1]['value'])
        
        if args.infer_method == 'generation':
            self.instruction_list = [
                "Which text option most closely corresponds to the image?",
                "Which text option best matches the image?",
                "Based on the image, select one text option to describe it.",
                "Which text correctly describes the content of image?",
                "Please select the correct text description based on the image provided."
            ]
        elif args.infer_method == 'likelihood':
            self.instruction_list = [
                "Which text most closely corresponds to the image?",
                "Which text best matches the image?",
                "Based on the image, select one text to describe it.",
                "Which text correctly describes the content of image?",
                "Please select the correct text description based on the image provided."
            ]
        else:
            raise Exception("Invalid infer method!")
        if duplication > 1:
            # the data sample is repeated with different prompts
            assert duplication % len(self.instruction_list) == 0, "the duplication times should be multiplication of the number of different prompts"

        self.samples = []
        self.proc = proc
        self.duplication = duplication
        if not self.config['data_config']['load_from_bootstrap']:
            raise Exception("current answer option doesn't support improvisation!")
        else:
            for i, item in tqdm.tqdm(enumerate(data), desc='preprocessing the data file'):
                """
                item is {'img':ab/path,'anno':[str],'gt':int}
                """
                current_sample = {'sample_id': i,
                                'image': item['img'],
                                'answer': str(item['gt']), # set to index now
                                'answer_options': item['anno'],
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
            raise Exception("current answer option doesn't support improvisation!")
        
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
            raise Exception("current answer option doesn't support improvisation!")
        new_sample['question'] = random.choice(self.instruction_list)
        return new_sample

    def __len__(self):
        return len(self.samples) * self.duplication



if __name__ == "__main__":
    ds = ImageTextSelection(args='')
    print('the dataset has {} samples'.format(len(ds)))
    random_index = random.randint(0, len(ds))
    print('examples in the dataset:')
    print('{}-th sample:'.format(random_index+1), ds[random_index+1])
    print('{}-th sample:'.format(random_index), ds[random_index])
    print('{}-th sample:'.format(random_index-1), ds[random_index-1])