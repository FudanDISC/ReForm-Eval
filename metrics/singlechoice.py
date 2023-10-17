import re
class SingleChoiceMetric(object):
    def __init__(self, alphabet='all', infer_method='generation'):
        if alphabet == 'all':
            re_format = '\([1-9A-Za-z]\)'
        elif alphabet == 'upper':
            re_format = '\([A-Z]\)'
        elif alphabet == 'lower':
            re_format = '\([a-z]\)'
        elif alphabet == 'number':
            re_format = '\([1-9]\)'
        self.re_format = re_format
        ab = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ',
              'abcdefghijklmnopqrstuvwxyz',
              '123456789']
        self.ab_map = {}
        self.infer_method = infer_method
        for ab_item in ab:
            self.ab_map.update({k:i for i,k in enumerate(ab_item)})
    
    def __call__(self, prediction, answer, options=None):
        # check the prediction type
        if self.infer_method == 'generation':
            assert isinstance(prediction, str), 'the prediction for gneration-based evaluation should be str'
        elif self.infer_method == 'likelihood':
            assert isinstance(prediction, int), 'the prediction for likelihood-based evaluation should be int'
        else:
            raise ValueError
        if type(prediction) == int:
            # the prediction is already the index
            return int(prediction==int(answer)), int(prediction)
        patterns = re.findall(self.re_format, prediction)
        if len(patterns) == 0:
            if options is not None:
                flag = False
                pred_index = None
                for i,opt in enumerate(options):
                    if opt == prediction:
                        # no answer mark but exact match
                        flag = True
                        pred_index = i
                        break
                if flag:
                    return int(pred_index==int(answer)), pred_index
                else:
                    return 0, None
            else:
                # format error
                return 0, None
        else:
            pred = self.ab_map[patterns[0][1]] # the first (*)
            if pred == int(answer):
                return 1, pred
            else:
                return 0, pred


