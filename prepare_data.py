import os
from PIL import Image
import shutil


pixelart_dir = './examples'
image_dir = './datasets/TEST_DATA/Input'
testA_dir = './datasets/TEST_DATA/testA'
testB_dir = './datasets/TEST_DATA/testB'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if not os.path.exists(testA_dir):
    os.makedirs(testA_dir)
if not os.path.exists(testB_dir):
    os.makedirs(testB_dir)
def rescale(image, Rescale=True):
    if not Rescale:
        return image
    if Rescale:
        width, height = image.size
        while width > 1280 or height > 1280:
            image = image.resize((int(width // 2), int(height // 2)), Image.BICUBIC)
            width, height = image.size
        while width < 128 or height < 128:
            image = image.resize((int(width * 2), int(height * 2)), Image.BICUBIC)
            width, height = image.size
        return image

if __name__ == '__main__':
    '''
    Cell_Size: Specify cell size from 2 to 8, default 4.
    Rescale: Whether to scale the image to prevent the image resolution from being too large or too small, default True.
    '''
    Cell_Size = 4
    Rescale = True
    num = 1
    for file in os.listdir(image_dir):
        if file.endswith('png') or file.endswith('jpg'):
            image_path = os.path.join(image_dir, file)
            save_path = os.path.join(testA_dir, '{}_{}.png'.format(Cell_Size, num))
            pixelart_path = os.path.join(pixelart_dir, '{}_1.png'.format(Cell_Size))
            pixelart_save_path = os.path.join(testB_dir, '{}_{}.png'.format(Cell_Size, num))
            image = Image.open(image_path).convert('RGB')
            image = rescale(image, Rescale)
            image.save(save_path)
            shutil.copy(pixelart_path, pixelart_save_path)
            num += 1
        else:
            print('The format of input image should be jpg or png.')

