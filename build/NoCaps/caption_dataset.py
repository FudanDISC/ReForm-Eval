import tqdm
from torch.utils.data import Dataset
import yaml
import random
import json
from utils.data_utils import get_image, base64_to_image
from datasets import load_dataset

# Caption for NoCaps
class Caption(Dataset):
    def __init__(self, args, config='datasets/configs/Caption_NoCaps_val.yaml', proc=None, duplication=1):
        if type(config) == str:
            self.config = yaml.load(open(config, 'r'), loader=yaml.loader)
        else:
            self.config = config
        self.args = args
        if args.hf:
            data = load_dataset("Aweminus/ReForm-Eval-Data", data_files={'test':self.config['huggingface_data']}, split='test')
        else:
            data = json.load(open(self.config['data_config']['nocaps_path'], 'r'))
        assert data['dataset_name'] == 'nocaps'
        data = data['data']

        # proc.response_prefix="The image features"
        proc.response_prefix=None
        
        self.instruction_list = [
            "Generate one sentence to describe the content of the image.",
            "Produce a single-sentence caption of the image's contents.",
            "Generate a brief descriptive statement for the image's content.",
            "Compose a short sentence that outlines what is depicted in the figure.",
            "Develop a one-sentence summary of the visual elements present in the picture.",
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
        
        if self.args.hf:
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
        if self.args.hf:
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