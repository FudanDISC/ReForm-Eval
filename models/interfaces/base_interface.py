import torch
import torch.nn as nn

class Base_Interface(nn.Module):
    def __init__(self, model_config, device=None, half=False, inference_method='generation'):
        # setup the common attribute
        self.inference_method = inference_method

        # model device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # if half precision
        self.prec_half = half

        '''
        **TODO** Model Loading Part Here:
        1. load your model (consider if half precision is used)
        2. locate your model to self.device
        '''

    def raw_batch_generate(self, images, prompts, temperature=1, max_new_tokens=30):
        r"""
        Black-box Generation-based Inference Method

        Args:
            image (list[PIL.Image]):
                The batch of input images. Each element is loaded as PIL.Image.
            prompt (list[str]):
                The batch of input textual prompts. Prompts should be formulated as a dialoge by the
                model preprocessor (see utils/preprocessors.py)
            temperature (float, **optional**):
                A generation-related parameter: the temperature parameter in the generation process
                of language models.
            max_new_tokens (int, **optional**):
                A generation-related parameter: the maximal number of tokens a model can generate.
                
        Returns:
            outputs (list[str]):
                The generated output response in text.

        Example:

        ```python
        >>> # An example of VQA for LLaVA
        >>> from models.interfaces.llava_interface import LLaVA_Interface
        >>> from PIL import Image

        >>> image = Image.open(PATH_TO_IMAGE).convert('RGB')
        >>> model = LLaVA_Interface(PATH_TO_LLAVA, device='cuda:0')

        >>> prompt = "A chat between a curious human and an artificial intelligence assistant. The\
                      assistant gives helpful detailed, and polite answers to the human's questions.\
                      ###Human: <image>\n Can you see the Image? Options: (A) yes; (B) no.\
                      ###Assistant: The answer is (A) yes.\
                      ###Human: What color is the truck? Options: (A) blue; (B) orange.\
                      ###Assistant: The answer is"

        >>> # Generation-based Inference
        >>> outputs = model.raw_batch_generate([image], [prompt])
        >>> outputs
        "(B) orange."
        ```"""
        raise NotImplementedError

    def raw_batch_predict(self, images, prompts, candidates):
        """
        White-box Likelihood-based Inference Method

        Args:
            image (list[PIL.Image]):
                The batch of input images. Each element is loaded as PIL.Image.
            prompt (list[str]):
                The batch of input textual prompts. Prompts should be formulated as a dialoge by the
                model preprocessor (see utils/preprocessors.py)
            candidates (list[list[str]]):
                The list of candidate lists, each element (candidates[i]) is the candidate list
                of the corresponding question.
                
        Returns:
            outputs (list[int]):
                The generated output prediction index. Each element (outputs[i]) is the selected index
                of the corresponding candidates. The prediction is therefore (candidates[i][outputs[i]])

        Example:

        ```python
        >>> # An example of VQA for LLaVA
        >>> from models.interfaces.llava_interface import LLaVA_Interface
        >>> from PIL import Image

        >>> image = Image.open(PATH_TO_IMAGE).convert('RGB')
        >>> model = LLaVA_Interface(PATH_TO_LLAVA, device='cuda:0')

        >>> prompt = "A chat between a curious human and an artificial intelligence assistant. The\
                      assistant gives helpful detailed, and polite answers to the human's questions.\
                      ###Human: What color is the truck?\
                      ###Assistant:"
        >>> candidates = ['orange', 'blue']

        >>> # Likelihood-based Inference
        >>> outputs = model.raw_batch_predict([image], [prompt], [candidates])
        >>> outputs
        1
        ```"""
        raise NotImplementedError
    
    def forward(self, images, prompts, candidates=None, temperature=1, max_new_tokens=30):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(images, prompts, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(images, prompts, candidates)
        else:
            raise ValueError