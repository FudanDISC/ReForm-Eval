## 📥 Prepare Dataset
To address the wide range of questions posed by users, LVLMs need to possess diverse capabilities. For a comprehensive evaluation, we curate 61 benchmark datasets from existing resources, summarizing the assessed capabilities into 2 major categories and 8 sub-categories.

<p align="center"><img src="../base_dimensions.png" width="600" height="600"/></p>
Assessed capability dimensions and tasks in ReForm-Eval. “Desc” and “Classif.” are respectively short for description and classification.


### Load ReForm-Eval-Data (Recommended)
In order to make it easier for users to use our benchmark, we further convert the formulated dataset and store image in the form of `base64`. We called this converted dataset **ReForm-Eval-Data**, which is uploaded to Hugging Face Hub and Google Drive. You can load our dataset directly from our Hugging Face repository or from the local path, avoiding the hassle of manual downloading, so this is also the most recommended method. 

**Please set `--hf` or `--offline_hf` if you would like to load ReForm-Eval-Data when using our framework (`run_eval.py` and `run_loader_eval.py`). `--hf` is loading from Hugging Face Hub, and `--offline_hf` is loading ReForm-Eval-Data from the local path. If set at the same time, data will be loaded from Hugging Face Hub.**

**Please set `load_from_hf=True` or `offline_from_hf=True` if you would like to load ReForm-Eval-Data when using Data Loader (`from build import load_reform_dataset`). `load_from_hf=True` is loading from Hugging Face Hub, and `offline_from_hf=True` is loading ReForm-Eval-Data from the local path. If `True` is set at the same time, data will be loaded from Hugging Face Hub.**

#### Load from Hugging Face Hub
In our repository, `huggingface_data` is the relative path in Hugging Face Hub and is configured in `/path/to/ReForm-Eval/build/configs/DisasterType_val.yaml` as shown below:
```YAML
dataset: 'MEDIC'
task: 'dts' # disaster type selection
data_config:
  load_from_bootstrap: True
  image_path: "/remote-home/share/multimodal-datasets/raw_datasets/MEDIC/data"
  medic_path: "/remote-home/share/multimodal-datasets/Gen_Eval/Disaster-Type-Selection/disaster-type-selection-sampled.json"
  huggingface_data: "huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the relative path in Hugging Face Hub
  offline_huggingface_data: "ReForm-Eval-Data/huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the relative local path of Hugging Face data
```
  
In `/path/to/ReForm-Eval/build/MEDIC/disaster_type_dataset.py`, the specific data in Hugging Face Hub is loaded directly from the path in the config file using `load_dataset` function, so no changes are needed.
```python
    if args.hf:
        data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
```

#### Load from the Local Path
If you cannot access Hugging Face, you can use the following command to download the dataset, and then load the dataset locally.

**git clone**
```bash
git lfs install
git clone https://huggingface.co/datasets/Aweminus/ReForm-Eval-Data
```

**download URL**

[https://drive.google.com/file/d/1YUq6pacbusNUPviQeilMQhZdP-o7Dd_b/view?usp=sharing](https://drive.google.com/file/d/1YUq6pacbusNUPviQeilMQhZdP-o7Dd_b/view?usp=sharing)

**wget**
```
wget https://drive.google.com/uc?export=download&id=1YUq6pacbusNUPviQeilMQhZdP-o7Dd_b
```

When you git clone the dataset or place the `ReForm-Eval-Data` folder on the root directory of this repository , `offline_huggingface_data` does not need to be modified. 

```
|-- ReForm-Eval
    |-- ReForm-Eval-Data
        |-- huggingface_data
            |-- A-OKVQA
            |-- A-OKVQAR
            ...
    |-- build
    |-- commands
    |-- metrics
    |-- models
    ...
```

Otherwise it needs to be modified:
```YAML
dataset: 'MEDIC'
task: 'dts' # disaster type selection
data_config:
  load_from_bootstrap: True
  image_path: "/remote-home/share/multimodal-datasets/raw_datasets/MEDIC/data"
  medic_path: "/remote-home/share/multimodal-datasets/Gen_Eval/Disaster-Type-Selection/disaster-type-selection-sampled.json" 
  huggingface_data: "huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the relative path in Hugging Face Hub
  offline_huggingface_data: "ReForm-Eval-Data/huggingface_data/MEDIC/disaster-type-selection-sampled.json" # The place you may need to modify (the relative local path of Hugging Face data)
```

#### Load the Raw Hugging Face Json Data
If you are really interested in exactly how we formulate the data and desire to check out the raw Hugging Face json data, use the following code:
```python
from datasets import load_dataset
# You can add (field="data") in parameters for extracting "data" keys.
# Load from the Hugging Face Hub
dataset = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':'huggingface_data/MEDIC/disaster-type-selection-sampled.json'}, split='test') 
# Load from the local path
dataset = load_dataset("json",data_files={'test':'/path/to/disaster-type-selection.json'}, split='test')
```

If you intend to check out one sample of our formulated data, mostly, you should add `[0]` between `dataset` and `['data']`, which is different from `json.load`.
```python
dataset = dataset[0]['data'][n] #n: The `n` th sample you want to check out
```

However, a few json files for some dataset such as TDIUC, you do not need to specify a `['data']` field or add `[0]` between `dataset` and `[0]` to get a sample.
```python
dataset = dataset[n] #n: The `n` th sample you want to check out
```

We saved the image in the form of `base64` in all json files. These processed texts are restored to complete images by PIL when the dataset is built.

<!-- You can load our dataset directly from our Hugging Face repository, avoiding the hassle of manual downloading, so this is also the most recommended method. 

**Please set `--hf` or `--offline_hf` if you would like to load data from Hugging Face when using our framework (`run_eval.py` and `run_loader_eval.py`). `--hf` is loading from Hugging Face Hub, and `--offline_hf` is loading Hugging Face data from the local path. If set at the same time, data will be loaded from Hugging Face Hub.**

**Please set `load_from_hf=True` or `offline_from_hf=True` if you would like to load Hugging Face data when using Data Loader (`from build import load_reform_dataset`). `load_from_hf=True` is loading from Hugging Face Hub, and `offline_from_hf=True` is loading Hugging Face data from the local path. If `True` is set at the same time, data will be loaded from Hugging Face Hub.**

If you want to read the raw json data directly, use the following code:
```python
from datasets import load_dataset
# You can add (field="data") in parameters for extracting "data" keys.
# Load from the Hugging Face Hub
dataset = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':'huggingface_data/MEDIC/disaster-type-selection-sampled.json'}, split='test') 
# Load from the local path
dataset = load_dataset("json",data_files={'test':'/path/to/disaster-type-selection.json'}, split='test')
```
In our repository, `huggingface_data` is already set in `/path/to/ReForm-Eval/build/configs/DisasterType_val.yaml` as shown below:
```YAML
dataset: 'MEDIC'
task: 'dts' # disaster type selection
data_config:
  load_from_bootstrap: True
  image_path: "/remote-home/share/multimodal-datasets/raw_datasets/MEDIC/data"
  medic_path: "/remote-home/share/multimodal-datasets/Gen_Eval/Disaster-Type-Selection/disaster-type-selection-sampled.json"
  huggingface_data: "huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the relative path in Hugging Face Hub
  offline_huggingface_data: "ReForm-Eval-Data/huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the relative local path of Hugging Face data
```
  
And in `/path/to/ReForm-Eval/build/MEDIC/disaster_type_dataset.py`, the specific path of Hugging Face Hub is read directly from the config file, so no changes are needed.
```python
    if args.hf:
        data = load_dataset("Aweminus/ReForm-Eval-Data",data_files={'test':self.config['data_config']['huggingface_data']}, split='test')
```

**If you cannot access Hugging Face, you can use the following command to download the dataset, and then load the dataset locally.**

**git clone**
```bash
git lfs install
git clone https://huggingface.co/datasets/Aweminus/ReForm-Eval-Data
```

**download URL**

[https://drive.google.com/drive/folders/1RMi-Mbl6VqJ4oFZL5eb8K6iHGmJCdEGl?usp=drive_link](https://drive.google.com/drive/folders/1RMi-Mbl6VqJ4oFZL5eb8K6iHGmJCdEGl?usp=drive_link)

**wget**
```

```

When you git clone the dataset or place the unzipped data folder on the root directory of this repository , `offline_huggingface_data` does not need to be modified. 

```
|-- ReForm-Eval
    |-- ReForm-Eval-Data
        |-- huggingface_data
            |-- A-OKVQA
            |-- A-OKVQAR
            ...
    |-- build
    |-- commands
    |-- metrics
    |-- models
    ...
```

Otherwise it needs to be modified:
```YAML
dataset: 'MEDIC'
task: 'dts' # disaster type selection
data_config:
  load_from_bootstrap: True
  image_path: "/remote-home/share/multimodal-datasets/raw_datasets/MEDIC/data"
  medic_path: "/remote-home/share/multimodal-datasets/Gen_Eval/Disaster-Type-Selection/disaster-type-selection-sampled.json" 
  huggingface_data: "huggingface_data/MEDIC/disaster-type-selection-sampled.json" # the relative path in Hugging Face Hub
  offline_huggingface_data: "ReForm-Eval-Data/huggingface_data/MEDIC/disaster-type-selection-sampled.json" # The place you may need to modify (the relative local path of Hugging Face data)
``` -->

### Manual Download (Not Recommended)
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


After downloading all dataset, you need to modify following paths of config files in `PATH_TO_REFORM-EVAL/datasets/configs/`.

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

We also provide the raw json file, like the one pointing to "medic_path". 

**download URL**

[https://drive.google.com/file/d/1D4CH9_RJKoCGFqDy5eIhG7h-ZRllgSfc/view](https://drive.google.com/file/d/1D4CH9_RJKoCGFqDy5eIhG7h-ZRllgSfc/view)

**wget**
```
wget https://drive.google.com/uc?export=download&id=1D4CH9_RJKoCGFqDy5eIhG7h-ZRllgSfc
```

### Online Multi-round dialogue
For multi-round VQA tasks, different from VisDial to perform offline multi-round dialogue (use GT in the dialogue history), we consider online multi-round dialogue (use previous output in the dialogue history).

In our framework, we use the "--online_multi_round" parameter to indicate the setting.

If you perform online multi-round dialogue without out framework, you need to be careful to update the history in the dataset during the evaluation, here is the example of this procedure in our framework (in run_eval.py):
```python
def get_pred_result(samples, prediction, metric):
    history_result = []
    # iterate through the prediction batch
    for i in range(len(prediction)):
        # detect whether the prediction matches a opton
        correct, final_pred = metric(prediction[i], samples['answer'][i], samples['answer_options'][i])
        if final_pred is None:
            # if the prediction does not match the option, then keep it
            final_pred = prediction[i]
        else:
            # then map the prediction to the original option
            try:
                final_pred = samples['answer_options'][i][final_pred]
            except:
                print('found invalid prediction: {}'.format(prediction[i]))
                final_pred = prediction[i]
                # raise ValueError
        history_result.append([samples['sample_id'][i], final_pred])
    return history_result

def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return power, memory  
for batch in tqdm.tqdm(dataloader, desc='evaluating'):
    # the inference process
    if args.infer_method == 'generation':
        res = model(batch['image'], batch['text'], **generation_kwargs)
    else:
        res = model(batch['image'], batch['text'], batch['answer_options'], **likelihood_kwargs)
    
    # get the prediction from the output
    generated_history_infos = get_pred_result(batch, res, metric)
    # gather all predictions from all gpus
    gathered_history = [i for i in range(args.n_gpus)]
    dist.all_gather_object(gathered_history, generated_history_infos)
    # make the update to the dialog history in the dataset
    """
    gathered_history: List[List[str, str]], each element is a list of the sample_id and prediction result.
    Here is an example of dataset_duplication=5, our model finishes the prediction for the first round in the VisDial_00 sample.
    >>> gathered_history
    [['VisDial_00_round0', 'yes'], ['VisDial_00_round0', 'no'], ['VisDial_00_round0', 'yes'], ['VisDial_00_round0']]
    """
    dataset.update_history(gathered_history)
``` 
Then in the next round, the history will be updated with the prediction "yes". 

Also notice that "round_id" should be included in the output json, during evaluation, use "--multi_round_eval" to evaluate the relationship between the model performance and the number of dialogue rounds.