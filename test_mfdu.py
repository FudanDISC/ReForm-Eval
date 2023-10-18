from datasets import load_dataset
test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/COCO_Text/cocotext_core_open_ended.json"}, split='test')
print(test['data'][0])#test['data']和[0]之间可能有其他项，视不同数据集而定
import json
with open('/remote-home/share/multimodal-datasets/huggingface_data/COCO_Text/cocotext_core_open_ended.json','r') as f:
    data = json.load(f)

import json
with open('/remote-home/share/multimodal-datasets/huggingface_data/COCO_Text/cocotext_core_open_ended1.json','w') as f:
    json.dump(data,f)
a = 1