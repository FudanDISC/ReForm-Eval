## 📥 Prepare Dataset
To address the wide range of questions posed by users, LVLMs need to possess diverse capabilities. For a comprehensive evaluation, we curate 61 benchmark datasets from existing resources, summarizing the assessed capabilities into 2 major categories and 8 sub-categories.

<p align="center"><img src="../base_dimensions.png" /></p>
Assessed capability dimensions and tasks in ReForm-Eval. “Desc” and “Classif.” are respectively short for description and classification.


### Load Dataset from Hugging Face (Recommended)
You can load our dataset directly from our Hugging Face repository, avoiding the hassle of manual downloading, so this is also the most recommended method. If you want to read the raw json data directly, use the following code:
```python
from datasets import load_dataset
# You can add (field="data") in parameters for extracting "data" keys.
# Load from the Hugging Face page
dataset = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':'huggingface_data/MEDIC/disaster-type-selection-sampled.json'}, split='test') 
# Load from the local path
dataset = load_dataset("json",data_files={'test':'/path/to/disaster-type-selection.json'}, split='test')
```
In our repository, `huggingface_data` is already set in `./build/configs/DisasterType_val.yaml` as shown below:
```YAML
dataset: 'MEDIC'
task: 'dts' # disaster type selection
data_config:
  load_from_bootstrap: True
  image_path: "/remote-home/share/multimodal-datasets/raw_datasets/MEDIC/data"
  medic_path: "/remote-home/share/multimodal-datasets/Gen_Eval/Disaster-Type-Selection/disaster-type-selection-sampled.json"
  huggingface_data: "huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the path of hugging face data
  offline_huggingface_data: "ReForm-Eval-Data/huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the relative local path of hugging face data
```
  
And in `./build/MEDIC/disaster_type_dataset.py`, the hugging data path is read directly from the config file, so no changes are needed.
```python
    if args.hf:
        data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['huggingface_data']}, split='test')
```

If you cannot access hugging face, you can use the following command to download the dataset, and then load the dataset locally still with a single line of code.
```bash
git lfs install
git clone https://huggingface.co/datasets/Aweminus/ReForm-Eval-Data
```

When you git clone the dataset from the root directory of this repository, `offline_huggingface_data` do not need to be modified, otherwise it need to be modified:
```YAML
dataset: 'MEDIC'
task: 'dts' # disaster type selection
data_config:
  load_from_bootstrap: True
  image_path: "/remote-home/share/multimodal-datasets/raw_datasets/MEDIC/data"
  medic_path: "/remote-home/share/multimodal-datasets/Gen_Eval/Disaster-Type-Selection/disaster-type-selection-sampled.json" 
  huggingface_data: "/path/to/disaster-type-selection.json" # the path of hugging face data
  offline_huggingface_data: "ReForm-Eval-Data/huggingface_data/MEDIC/disaster-type-selection-sampled.json" # The place you may need to modify (the relative local path of hugging face data)
```

### Manually Download
Alternatively, all datasets are also provided with URLs and you can manually download them.

| Dataset          | URL                                                                          |
|------------------|-------------------------------------------------------------------------------|
| AOKVQA           | [https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz](https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz) |
| CIFAR10          | [http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)                              |
| CLEVR            | [https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip)                   |
| COCO_text        | [https://s3.amazonaws.com/cocotext/COCO_Text.zip](https://s3.amazonaws.com/cocotext/COCO_Text.zip)                        |
| CUTE80           | [http://cs-chan.com/downloads_CUTE80_dataset.html](http://cs-chan.com/downloads_CUTE80_dataset.html)                        |
| Flickr30K        | [http://shannon.cs.illinois.edu/DenotationGraph/data/index.html](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)        |
| Flowers102       | [https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset) |
| FUNSD            | [https://guillaumejaume.github.io/FUNSD/download/](https://guillaumejaume.github.io/FUNSD/download/)                        |
| IC15             | [http://rrc.cvc.uab.es/?ch=4&com=downloads](http://rrc.cvc.uab.es/?ch=4&com=downloads)                                |
| IIIT5K           | [https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) |
| ImageNet1K       | [https://www.image-net.org/update-mar-11-2021.php](https://www.image-net.org/update-mar-11-2021.php)                          |
| MEDIC            | [https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz](https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz)                       |
| MOCHEG           | [https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform?usp=sf_link](https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform?usp=sf_link) |
| MP3D             | [https://niessner.github.io/Matterport/](https://niessner.github.io/Matterport/)                                   |
| MSCOCO           | [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)                           |
| NoCaps           | [https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json](https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json)           |
| Pets37           | [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)                              |
| POIE             | [https://drive.google.com/file/d/1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs/view](https://drive.google.com/file/d/1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs/view)  |
| RefCOCO          | [https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)          |
| RefCOCO+         | [https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)         |
| RefCOCOg         | [https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)         |
| SNLI_VE          | [https://github.com/necla-ml/SNLI-VE](https://github.com/necla-ml/SNLI-VE)                                      |
| SROIE            | [https://rrc.cvc.uab.es/?ch=13&com=downloads](https://rrc.cvc.uab.es/?ch=13&com=downloads)                              |
| TDIUC            | [https://kushalkafle.com/projects/tdiuc.html](https://kushalkafle.com/projects/tdiuc.html)                              |
| TextCaps         | [https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)      |
| TextCaps JSON    | [https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json) |
| TextOCR          | [https://textvqa.org/textocr/dataset/](https://textvqa.org/textocr/dataset/)                                    |
| VisDIal          | [https://visualdialog.org/data](https://visualdialog.org/data)                                           |
| VizWiz           | [https://vizwiz.org/tasks-and-datasets/vqa/](https://vizwiz.org/tasks-and-datasets/vqa/)                              |
| VQA              | [https://visualqa.org](https://visualqa.org)                                                    |
| VSR              | [https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data) |
| WikiHow          | [https://drive.google.com/u/0/uc?id=1vnDduJmuFpeT8yTgtBR9Bj8bXlI4zIJR&export=download](https://drive.google.com/u/0/uc?id=1vnDduJmuFpeT8yTgtBR9Bj8bXlI4zIJR&export=download) |
| Winoground       | [https://huggingface.co/datasets/facebook/winoground](https://huggingface.co/datasets/facebook/winoground)                     |
| WordArt          | [https://drive.google.com/file/d/1SanxRwTxd2q](https://drive.google.com/file/d/1SanxRwTxd2q)|


After downloading the all dataset, you need to modify all paths of config files in `PATH_TO_REFORM-EVAL/datasets/configs/`.
```YAML
dataset: 'MEDIC'
task: 'dts' # disaster type selection
data_config:
  load_from_bootstrap: True
  image_path: "/remote-home/share/multimodal-datasets/raw_datasets/MEDIC/data" #The place you need to modify
  medic_path: "/remote-home/share/multimodal-datasets/Gen_Eval/Disaster-Type-Selection/disaster-type-selection-sampled.json" #The place you need to modify
  huggingface_data: "huggingface_data/MEDIC/disaster-type-selection-sampled.json"
  offline_huggingface_data: "ReForm-Eval-Data/huggingface_data/MEDIC/disaster-type-selection-sampled.json" 
```
