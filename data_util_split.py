import splitfolders
import os
print("Splitting is started!")

splitfolders.ratio(
    "view_dset_01/mlo", # original image path
    output="view_dset_01_splitted/mlo_splitted",
    seed=42,
    ratio=(.8, .0, .2),
    group_prefix=None,
    move=False
)

print("len train, 0:", len(os.listdir("view_dset_01_splitted/mlo_splitted/train/0")))
print("len train, 1:", len(os.listdir("view_dset_01_splitted/mlo_splitted/train/1")))
print("len test, 0:", len(os.listdir("view_dset_01_splitted/mlo_splitted/test/0")))
print("len test, 1:", len(os.listdir("view_dset_01_splitted/mlo_splitted/test/1")))