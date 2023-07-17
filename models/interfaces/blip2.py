from lavis.models import load_model_and_preprocess
import torch
from PIL import Image

def get_image(image):
    image = Image.open(image)
    return image.convert('RGB')

class BLIP2_Interface:
    def __init__(self, model_name='blip2_t5', model_type='pretrain_flant5xl', device=None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=self.device
        )
        # self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Answer:"
        }, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [f"Question: {question} Answer:" for question in question_list]
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)

        return output
    
def get_blip2(model_config):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_name', 'model_type', 'device']
        for arg in valid_args:
            if arg in model_config:
                model_args[arg] = model_config[arg]
    return BLIP2_Interface(**model_args)

def get_instructblip(model_config):
    pass