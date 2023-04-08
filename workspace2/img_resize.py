from PIL import Image
import os


image_path = "./data/view_dset_01_splitted/cc_splitted/laterality_fixed"
images = [file for file in os.listdir(image_path) if file.endswith(('jpeg', 'png', 'jpg'))]
for image in images:
    img = Image.open(image_path+"/"+image)
    img.thumbnail((128,128))
    img.save("./data/view_dset_01_splitted/cc_splitted/128x128/"+"resized_"+image)