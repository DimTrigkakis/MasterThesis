# Convert all jpg files to gameid_framenum_aug.jpg files with corresponding xml files that include all their information, like viewpoint labels

import glob
import os
from shutil import copyfile
image_names = glob.glob("./video_annotations/2014/*.jpg")
print len(image_names)

indexx = 0
for image_name in image_names:
    print indexx
    indexx += 1
    base_name = os.path.basename(image_name).split(".")[0]
    aug = base_name.split("_")[0]
    if aug == "scoreboard":
        aug = 1
    else:
        aug = 0
    
    index = 0
    if aug == 1:
        index = 1
    game_id = base_name.split("_")[index+1]
    frame = base_name.split("_")[index+2]
    label = base_name.split("_")[index+3]

    # write data in xml
    # rename file to something else

    if aug == 0:
        folder = "./training_data/"+game_id+"/"
    else:
        folder = "./training_data/aug/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    image_name_true = folder+game_id+"_"+frame+"_"+label+".jpg"
    
    copyfile(image_name,image_name_true)
    info = base_name.split("_")

print "done"
