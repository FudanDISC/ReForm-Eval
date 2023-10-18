from datasets import load_dataset
test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/CLEVR/clevr_core_multiple_choice.json"}, split='test')
print(test[0]['data'][0]) #test和['data']之间的[0]不可以省略

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/VSR/vsr_core_multiple_choice.json"}, split='test')
print(test[0]['data'][0]) #test和['data']之间的[0]不可以省略

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/MP3D/mp3d_core_multiple_choice.json"}, split='test')
print(test[0]['data'][0]) #test和['data']之间的[0]不可以省略

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/MSCOCO/goi_core_multiple_choice.json"}, split='test')
print(test[0]['data'][0]) #test和['data']之间的[0]不可以省略

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/MSCOCO/mci_core_multiple_choice.json"}, split='test')
print(test[0]['data'][0]) #test和['data']之间的[0]不可以省略

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/MSCOCO/mos_core_multiple_choice.json"}, split='test')
print(test[0]['data'][0])


test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/COCO_Text/cocotext_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/COCO_Text/cocotext_gocr_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/IC15/ic15_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/IC15/ic15_gocr_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/TextOCR/textocr_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/TextOCR/textocr_gocr_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/IIIT5K/iiit5k_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/CUTE80/cute80_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/WordArt/wordart_core_open_ended.json"}, split='test')
print(test[0]['data'][0])


test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/SROIE/sroie_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/POIE/poie_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

test = load_dataset("json", data_files={'test':"/remote-home/share/multimodal-datasets/huggingface_data/FUNSD/funsd_core_open_ended.json"}, split='test')
print(test[0]['data'][0])

import json
with open('/remote-home/share/multimodal-datasets/huggingface_data/FUNSD/funsd_core_open_ended.json','r') as f:
    data = json.load(f)
    
with open('/remote-home/share/multimodal-datasets/huggingface_data/FUNSD/funsd_core_open_ended1.json','w') as f:
    json.dump(data,f)

    
    
