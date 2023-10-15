from .singlechoice import SingleChoiceMetric
from .ocropenended import OCROpenEndedMetric, KIEOpenEndedMetric

def get_metric(formulation, param=None):
    if formulation == 'SingleChoice':
        if param is None:
            return SingleChoiceMetric()
        else:
            return SingleChoiceMetric(**param)
    elif formulation == 'OCROpenEnded':
        if param is None:
            return OCROpenEndedMetric()
        else:
            return OCROpenEndedMetric(**param)
    elif formulation == 'KIEOpenEnded' :
        if param is None:
            return KIEOpenEndedMetric()
        else:
            return KIEOpenEndedMetric(**param)     
    elif formulation == 'TrueOrFalse':
        if param is None:
            return SingleChoiceMetric()
        else:
            return SingleChoiceMetric(**param)
    else:
        raise ValueError('current {} formulation is not supported!'.format(formulation))