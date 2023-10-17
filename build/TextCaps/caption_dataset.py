import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
from utils.data_utils import get_image, base64_to_image
from datasets import load_dataset

# Caption for TextCaps
class Caption(Dataset):
    def __init__(self, args, config='datasets/configs/Caption_TextCaps_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), loader=yaml.loader)
        else:
            self.config = config
        self.args = args
        if args.hf:
            data = load_dataset("Aweminus/ReForm-Eval-Data", data_files={'test':self.config['huggingface_data']}, split='test')
        elif args.offline_hf:
            data = load_dataset("json",data_files={'test':self.config['offline_huggingface_data']}, split='test')
        else:
            data = json.load(open(self.config['data_config']['textcaps_path'], 'r'))
        assert data['dataset_name'] == 'textcaps'
        data = data['data']

        # proc.response_prefix="The image features"
        proc.response_prefix=None
        
        self.instruction_list = [
            "Write a one-sentence description of the image, which would require reading the text in the image.",
            "Write a brief caption that encompasses the key elements of the image, paying special attention to the text in the image and its connection to other objects or entities.",
            "Write a short description of the image, emphasizing visible text and its relationship to the surrounding visual context.",
            "In a single sentence, describe the image and its key elements, with particular attention to the visible text.",
            "Describe the image in a single sentence, making sure to mention the text present in the image."
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
                item is {'img':ab/path,'annos':[str]}
                """
                current_sample = {'sample_id': i,
                                'image': item['img'],
                                'references': item['annos'],
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
        
        if self.duplication > 1:
            inner_sample_index = index % self.duplication
            new_sample['question'] = self.instruction_list[inner_sample_index % len(self.instruction_list)]
        else:
            new_sample['question'] = random.choice(self.instruction_list)
        
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

if __name__ == '__main__':
    pass