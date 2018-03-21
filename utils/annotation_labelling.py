import argparse
import os
from shutil import copyfile


parser = argparse.ArgumentParser(description='Viewpoint annotations')
parser.add_argument("--video", metavar='VIDEO', default=None,help="rename labels of a video given a folder with manual annotations")

args = parser.parse_args()

ma = open("./manual_annotations_2016/"+str(args.video)+".txt","rb")

lines = ma.readlines()
content = [x.strip() for x in lines]

current_frame = 1

directory = "./training_data/"+args.video
if not os.path.exists(directory):
    os.makedirs(directory)
print directory

first_time = True
for line in content:
    #print line 
    frame, label_type = line.split(" ")
    frame = int(frame)
    if first_time:
        first_time = False
        label = int(label_type)
    #label = int(label)
    print "GRAND",frame, label
    
    while current_frame < frame:
        if os.path.isfile("./video_annotations/"+str(args.video)+"/"+str(current_frame).zfill(8)+".png"):
            copyfile("./video_annotations/"+str(args.video)+"/"+str(current_frame).zfill(8)+".png",directory+"/"+args.video+"_"+str(current_frame)+"_"+str(label)+".png")            
        #print "frame is", current_frame, "with label", label    
        current_frame += 1
        #print current_frame, frame

    label = int(label_type) 
