from PIL import Image

def get_image(img):
    if type(img) == str:
        # img is the image path
        image = Image.open(img)
        image = image.convert('RGB')
        return image
    else:
        return img