import sys
import os

sys.path.append("/scratch/test/lib/python2.7/site-packages")

import cv2
import numpy as np
import glob

'''video_files = glob.glob("./videos/*.mp4")
for video_file in video_files:
	video_data = cv2.VideoCapture(video_file)
	while (video_data.isOpened()):
		ret, frame = video_data.read()
		image = frame
		print image'''

def scoreboard_deconstruction(video_folder, frame_folder):

    frame_skip = 1
    # glob all videos in video folder
    # for file_name in glob.glob(video_folder+"/*.mp4"):
    #     print file_name

    if not os.path.exists(frame_folder):
        print "FOLDER NOT FOUND"
        exit()
    else:
        for file_name in glob.glob(video_folder+"/*.mp4"):
            print file_name
            
            file_base = file_name.split(".mp")[0].split("/")[-1]
            print file_base
            mydir = frame_folder+"/"+file_base+"/"
            if not os.path.exists(mydir):
                print "creating dir", mydir
                os.makedirs(mydir)
            
            os.system("ffmpeg -i %s -vf \"select=not(mod(n\\,%d))\" -vsync vfr -crf 18 -vframes 56 %s%%08d.jpg" % (file_name, frame_skip, mydir))
            
    # if needed, in another folder copy frames 10 times to augment to appropriate number after checking that they are indeed cut correctly
    

def deconstruct_video(video,frame_folder, overwrite=True, frame_skip = 1):

    '''cap = cv2.VideoCapture(video)
    success, img = cap.read()
    frame = 0
    
    filename = os.path.basename(video).split(".")[0]
    
    directory = frame_folder+"/"+filename
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    while success:
        success,img = cap.read()
        #if frame % 100 == 0:
        print frame
        cv2.imwrite(directory+"/"+filename+"_%d.jpg" % frame, img)
        frame += 1
        #if frame > 100:
        #    break'''
    video = "./data/videos_2016/"+video+".mp4"
    filename = os.path.basename(video).split(".")[0]

    directory = frame_folder+"/"+filename+"/"
    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    print "Will use frame skip", frame_skip
    # added -crf 18 and frame_skip instead of 1
    os.system("ffmpeg -i %s -vf \"select=not(mod(n\\,%d))\" -vsync vfr -crf 18 %s%%08d.jpg" % (video, frame_skip, directory))

def extract_frames(annotations_file, overwrite=True):

    
    with open(annotations_file) as f:
        lines = f.readlines()

        init_game = None
        mp4file = None
        cap = None
        
        for line in lines:
            s = line.split(" ")

            label = s[1].strip("\n")
            code = s[0]
            augmented_file = code.split("/")[0]
            code = code.split("/")[1]

            team_name, game, frame, label_png = code.split("-")

            
            #print "Team:",team_name,"game:",game,"frame:",frame,"label:",label,"aug:",augmented_file
            
            # Every time the previous game was different, open the new .mp4 file and extract frames one by one
            if augmented_file != "images":
                # print "skip"
                # handle this differently#
                #continue
                pass
            if game != init_game:
                init_game = game
                print("Extracting game: %s" % (game))
                # open .mp4 file
                #print "ERROR BELOW?"
                #print team_name, game
                cap = cv2.VideoCapture('/scratch/datasets/NFLvid/videos/'+game+'.mp4')
                prev_frame = -1
                #print cap
                #print "IF NO ERROR ABOVE, SUCCESS"
               
            # extract frame <frame> from current .mp4 file <mp4file> given capture <cap>
            # continue
            
            if (overwrite == False):
                if (os.path.isfile("./frames/"+augmented_file+"_"+game+"_"+frame+"_"+label+".jpg")):
                    continue
            
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,int(frame))
            success = False
            frame_read = int(frame) - 1
            
            success,image = cap.read()
            if success:
                cv2.imwrite("./frames/"+augmented_file+"_"+game+"_"+frame+"_"+label+".jpg", image)
                

            if (not success):
                print(game + " Frame not read correctly! Frame read: %d, Real frame: %s" % (frame_read, frame))
            #print "frame extracted" 

if __name__ == "__main__":
    extract_frames(sys.argv[1])
