import os
from PIL import Image

path = '/Users/dim/Desktop/PythonFolder/my_Projects'
path2 = '/Users/dim/Desktop/PythonFolder/new_file'

folder_exists = os.path.exists(path2)

if not folder_exists:
    os.makedirs(path2)

cur_dir = os.listdir(path)
new_dir = os.listdir(path2)


for image in cur_dir:
    img = Image.open(path + '/' + image)
    clean_name = os.path.splitext(image)[0]
    new_image = img.save(f'{path2}/{clean_name}.png', 'png')

