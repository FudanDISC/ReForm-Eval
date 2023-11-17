from typing import Any
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

class GenerationMetric(object):
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            (Meteor(), "meteor"),
            (Rouge(), "rouge_l"),
            (Cider(), "cider")
        ]
    
    def __call__(self, results):
        """
        each dict of results has keys: 'prediction', 'references'
        """
        # bootstrap prediction and gt_answers dict
        pred_dict , gt_ans_dict = {}, {}
        for i, result in enumerate(results):
            pred_dict[f'{i}']=[result['prediction']]
            gt_ans_dict[f'{i}']=result['references']
        
        scores = {}
        for scorer, method in self.scorers:
            if isinstance(method, list):
                # BLEU scores
                score, _ = scorer.compute_score(gt_ans_dict, pred_dict)
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                # Meteor, Rouge-L, CIDEr
                score, _ = scorer.compute_score(gt_ans_dict, pred_dict)
                scores[method] = score
        
        return scores
