from PIL import Image
import base64
from io import BytesIO

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

def question_with_options(item):
    return 1
