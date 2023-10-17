from PIL import Image
import base64
from io import BytesIO
import random

def get_image(img):
    if type(img) == str:
        # img is the image path
        #binary_data = base64.b64decode(base64_data)
        #img = BytesIO(binary_data)
        image = Image.open(img)
        image = image.convert('RGB')
        return image
    else:
        return img
    
def base64_to_image(img):
    binary_data = base64.b64decode(img)
    img = BytesIO(binary_data)
    image = Image.open(img)
    image = image.convert('RGB')
    return image

def question_with_options(item, option_mark='random'):
    alphabet = ['abcdefghijklmnopqrstuvwxyz',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '123456789']
    ret = item['question']
    if option_mark == 'number':
        ab = alphabet[2]
    elif option_mark == 'lower':
        ab = alphabet[0]
    elif option_mark == 'upper':
        ab = alphabet[1]
    else:
        ab = alphabet
    if 'answer_options' in item:
        ret += ' Options: '
        if isinstance(ab, list):
            current_ab = random.choice(ab)
        else:
            current_ab = ab
        for i, opt in enumerate(item['answer_options']):
            ret += '({}) {}'.format(current_ab[i], opt)
            if i == len(item['answer_options']) - 1:
                ret += '.' # + seps[0]
            else:
                ret += '; '
    return ret
