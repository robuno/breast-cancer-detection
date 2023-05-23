import splitfolders
import os
print("Splitting is started!")

source_src = "ds/imp0_abc_49/mlo"
target_src = "ds/DARK/aug_49_mlo"

splitfolders.ratio(
    source_src, # original image path
    output=target_src,
    seed=42,
    ratio=(.8, .0, .2),
    group_prefix=None,
    move=False
)

print("len train, 0:", len(os.listdir(target_src+"/train/0")))
print("len train, 1:", len(os.listdir(target_src+"/train/1")))
print("len test, 0:", len(os.listdir(target_src+"/test/0")))
print("len test, 1:", len(os.listdir(target_src+"/test/1")))