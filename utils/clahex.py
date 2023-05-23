import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

target_path_0 = "rsna_dset_1/0/"
target_path_1 = "rsna_dset_1/1/"

img_str = str(target_path_1) + "64930_2024346917.png"
print(img_str)

img = cv.imread(img_str, cv.IMREAD_GRAYSCALE)

def write_histEq(img,file_name="histEq"):
    equ = cv.equalizeHist(img)
    res = np.hstack((img,equ)) # stacked images
    cv.imwrite(file_name,res)

def write_clahe(img, cli_limit, file_name="clahe"):
    clahe = cv.createCLAHE(clipLimit=cli_limit, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv.imwrite(file_name+"_"+str(cli_limit)+".jpg",cl1)

for cla_lim in [0.01, 0.1, 1, 2, 4, 8]:
    write_clahe(img, cla_lim)