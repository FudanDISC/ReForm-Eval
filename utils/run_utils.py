from collections import Counter
import math
import numpy as np
from sklearn.linear_model import LinearRegression as LR
import os

def naive_list_collate_fn(item_lits):
    full_items = {k: [] for k in item_lits[0].keys()}
    for item in item_lits:
        for k,v in item.items():
            full_items[k].append(v)
    return full_items

def entropy_calculation(question2pred):
    ent_list = []
    for k,v in question2pred.items():
        dist = [count/len(v) for pred,count in Counter(v).items()]
        ce = sum([-1*prob*math.log(prob) for prob in dist])
        ent_list.append(ce)
    return sum(ent_list) / len(ent_list)

import os

def multi_round_eval(round2metric):
    tmp_matrix = []
    for k,v in round2metric.items():
        tmp_matrix.append([k, sum(v)/len(v)])
    tmp_matrix = np.array(tmp_matrix)
    corr_matrix = np.corrcoef(tmp_matrix.transpose())
    lr_model = LR().fit(tmp_matrix[:,0:1], tmp_matrix[:, 1])
    return corr_matrix[0, 1], lr_model.coef_[0]

def get_pred_result(samples, prediction, metric):
    history_result = []
    for i in range(len(prediction)):
        correct, final_pred = metric(prediction[i], samples['answer'][i], samples['answer_options'][i])
        if final_pred is None:
            final_pred = prediction[i]
        else:
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

def get_model_name(model_name):
    if os.path.isdir(model_name):
        tmp_name = model_name.strip('/')
        return os.path.split(tmp_name)[1]
    elif '/' in model_name:
        tmp_name = model_name.strip('/')
        tmp_name = os.path.split(tmp_name)[1]
        tmp_name = tmp_name.split('.')[0]
        return tmp_name
    else:
        return model_name