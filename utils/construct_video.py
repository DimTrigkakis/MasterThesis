import os, sys
import argparse
import pickle
import glob
sys.path.append("/scratch/test/lib/python2.7/site-packages")

import cv2

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

parser = argparse.ArgumentParser(description='Video constructor')
parser.add_argument("--frame_label_folder",   metavar='FRAME_LABEL_FOLDER', help="Folder from which to extract the frames into a video")
parser.add_argument("--iou_metric",type=bool,metavar="IOU_METRIC",default=False, help="Use IoU metric on the video segmentations")
parser.add_argument("--construct_video",type=bool,metavar="CONSTRUCT_VIDEO",default=False, help="Extract frames into a single video file")
parser.add_argument("--add_annotations",type=bool,metavar="ADD_ANNOTATIONS",default=False,help="Add new frames with drawn annotations")

def construct_annotator(frame_folder,video_target,video_name):

    #folder = frame_label_folder
    #frames = folder+"/frames"
    #labels = folder+"/labels"
    #frame_annotations = folder+"/frame_annotations"

    #if not os.path.exists(frame_annotations):
    #        os.makedirs(frame_annotations)

    #segments = pickle.load(open(labels+"/viewpoint_segments.pkl","rb"))

    # For every segment, find its label and endpoints. Then for every frame, add text to it
    #print args.construct_video
    #print args.add_annotations
    images = glob.glob(frame_folder+"/*.png")
    img_sample = cv2.imread(images[0])
    h,w,c = img_sample.shape
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    
    video = cv2.VideoWriter(video_target+"/video.avi",fourcc,20.0,(w,h))
    # Sort image frames first
    images.sort(key=natural_keys)
    print images
    frame_num = 0
    for image in images:
        frame = cv2.imread(image)
        # "save frame num"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,str(frame_num),(10,40),font,1,(255,255,255),2)
        video.write(frame)
        frame_num += 1
    cv2.destroyAllWindows()
    video.release()
    '''
    if args.construct_video:
        if args.add_annotations:
            for segment in segments:
                label = segment[0][0]
                first = segment[0][1]
                last = segment[1][1]

                # for every frame that you find within the range first-last, label with label
                for i in range(first+1,last+2):
                    print first, last, label
                    frame = cv2.imread(frames+"/"+str(i).zfill(8)+".png")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,str(label),(10,40),font,1,(255,255,255),2)
                    cv2.imwrite(frame_annotations+"/"+str(i).zfill(8)+".png",frame)


        images = glob.glob(frame_annotations+"/*.png")
        img_sample = cv2.imread(images[0])
        height , width , layers =  img_sample.shape
        print folder+"/video.avi"
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        video = cv2.VideoWriter(folder+'/video.avi',fourcc,20.0,(width,height))
        for image in images:
            print "adding image ", image
            frame = cv2.imread(image)
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()
    else:
        print "Not constructing video"
    '''
def main():
    global args
    args = parser.parse_args()

    folder = args.frame_label_folder
    frames = folder+"/frames"
    labels = folder+"/labels"
    frame_annotations = folder+"/frame_annotations"

    if not os.path.exists(frame_annotations):
            os.makedirs(frame_annotations)

    segments = pickle.load(open(labels+"/viewpoint_segments.pkl","rb"))

    # For every segment, find its label and endpoints. Then for every frame, add text to it
    print args.construct_video
    print args.add_annotations
    if args.construct_video:
        if args.add_annotations:
            for segment in segments:
                label = segment[0][0]
                first = segment[0][1]
                last = segment[1][1]

                # for every frame that you find within the range first-last, label with label
                for i in range(first+1,last+2):
                    print first, last, label
                    frame = cv2.imread(frames+"/"+str(i).zfill(8)+".png")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,str(label),(10,40),font,1,(255,255,255),2)
                    cv2.imwrite(frame_annotations+"/"+str(i).zfill(8)+".png",frame)

    
        images = glob.glob(frame_annotations+"/*.png")
        img_sample = cv2.imread(images[0])
        height , width , layers =  img_sample.shape
        print folder+"/video.avi"
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        video = cv2.VideoWriter(folder+'/video.avi',fourcc,20.0,(width,height))
        for image in images:
            print "adding image ", image
            frame = cv2.imread(image)
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()
    else:
        print "Not constructing video"                            

if __name__ == "__main__":

    main()
