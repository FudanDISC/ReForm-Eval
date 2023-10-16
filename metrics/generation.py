from typing import Any
from pycocoevalcap.cider.cider import Cider

class GenerationMetric(object):
    def __init__(self):
        self.scorer = Cider()
    
    def __call__(self, results):
        """
        each dict of results has keys: 'prediction', 'references'
        """
        # bootstrap prediction and gt_answers dict
        pred_dict , gt_ans_dict = {}, {}
        for i, result in enumerate(results):
            pred_dict[f'{i}']=[result['prediction']]
            gt_ans_dict[f'{i}']=result['references']
        score, scores = self.scorer.compute_score(gt_ans_dict, pred_dict)
        
        return score, scores
