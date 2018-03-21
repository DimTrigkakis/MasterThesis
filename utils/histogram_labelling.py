import extract_videos as ev 

import glob
import cv2
from matplotlib import pyplot as plt
import math
import pickle
import numpy as np
import shutil

#videos = ["2016120402","2016122411","2016121108","2016091813"] # "2016091813" requires 2 removals

#videos = ["2016091813","2016091900","2016100300","2016112709","2016121802"]
# We need four final videos
#videos= ["2016121110"]
videos= ["2016110606"]
minimal = False
deconstruct = False
construct = False

def drange(start, stop, step):
     r = start
     while r < stop:
         yield r
         r += step

if minimal:
    for video in videos:
        ev.deconstruct_video(video=video,frame_folder="./data/mini_frame_data_2016", overwrite=True, frame_skip = 7*8)
        exit()
if deconstruct:
    for video in videos:
        ev.deconstruct_video(video=video,frame_folder="./data/full_frame_data_2016", overwrite=True, frame_skip = 7)
elif construct:
    for video in videos:
        frame_folder="./data/full_frame_data_2016"

        directory = frame_folder+"/"+video+"/"
        pics = glob.glob(directory+"*.jpg")

        first_pic = True
        dimensional = 10
        hist_origin = [None for j in range(dimensional*dimensional)]
        pic_d = []

        # do n little box histogram comparisons
        print len(pics)
        for k, pic in enumerate(pics):
            if first_pic:
                img = cv2.imread(pic,1)
                w , h = 954, 540
                wnew, hnew = w/dimensional, h/dimensional
                cropped = [None for i in range(dimensional*dimensional)]
                for i in range(dimensional):
                    for j in range(dimensional):
                        cropped[i+dimensional*j] = img[i*h/dimensional:i*h/dimensional+h/dimensional, j*w/dimensional:j*w/dimensional+w/dimensional]
                first_pic = False
                for d in range(dimensional*dimensional):
                    hist_origin[d] = cv2.calcHist([cropped[d]],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])

                pic_d.append((pic,k,1))
            else:
                img = cv2.imread(pic,1)
                w , h = 954, 540
                wnew, hnew = w/dimensional, h/dimensional
                cropped = [None for i in range(dimensional*dimensional)]
                hist = [None for j in range(dimensional*dimensional)]
                for i in range(dimensional):
                    for j in range(dimensional):
                        cropped[i+dimensional*j] = img[i*h/dimensional:i*h/dimensional+h/dimensional, j*w/dimensional:j*w/dimensional+w/dimensional]
                for d in range(dimensional*dimensional):
                    hist[d] = cv2.calcHist([cropped[d]],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])

                loss = [0 for i in range(dimensional*dimensional)]
                for d in range(dimensional*dimensional):
                    if hist[d] != None:
                        loss[d] = cv2.compareHist(hist_origin[d], hist[d], cv2.HISTCMP_CORREL)
                    else:
                        loss[d] = 2
                    if loss[d] != loss[d]:
                        loss[d] = loss[d-1]
                #print loss
                print k
                #loss_general = np.median(loss)
                #print loss_general, k
                pic_d.append((pic,k,loss))

            '''
            elif first_pic == -1:
                img = cv2.imread(pic,1)
                img2 = img[0:954, 0:540/4]     
                img3 = img[0:954, -75+540/2:75+540/2]
                
                hist = cv2.calcHist([img2],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
                hist = cv2.normalize(hist,hist).flatten()
                hist0 = cv2.calcHist([img3],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
                hist0 = cv2.normalize(hist0,hist0).flatten()
                
                d1 = min(cv2.compareHist(hist_base, hist, cv2.HISTCMP_CORREL),cv2.compareHist(hist_base0, hist0, cv2.HISTCMP_CORREL))
                d2 = max(cv2.compareHist(hist_base, hist, cv2.HISTCMP_CHISQR),cv2.compareHist(hist_base0, hist0, cv2.HISTCMP_CHISQR))
                d3 = min(cv2.compareHist(hist_base, hist, cv2.HISTCMP_INTERSECT),cv2.compareHist(hist_base0, hist0, cv2.HISTCMP_INTERSECT))
                d4 = max(cv2.compareHist(hist_base, hist, cv2.HISTCMP_BHATTACHARYYA),cv2.compareHist(hist_base0, hist0, cv2.HISTCMP_BHATTACHARYYA))
                d5 = max(cv2.compareHist(hist_base, hist, cv2.HISTCMP_HELLINGER),cv2.compareHist(hist_base0, hist0, cv2.HISTCMP_HELLINGER))
                print d1,d2,d3,d4,d5, k
                pic_d.append((pic,k,d1,d2,d3,d4,d5))
            else:
                img = cv2.imread(pic,1)        
                img2 = img[0:954, 0:540/4]  
                img3 = img[0:954, -75+540/2:75+540/2]
                
                hist_0[frame_snap-first_pic] = cv2.calcHist([img2],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
                hist_0[frame_snap-first_pic] = cv2.normalize(hist_0[frame_snap-first_pic],hist_0[frame_snap-first_pic]).flatten()
                hist_00[frame_snap-first_pic] = cv2.calcHist([img3],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
                hist_00[frame_snap-first_pic] = cv2.normalize(hist_00[frame_snap-first_pic],hist_00[frame_snap-first_pic]).flatten()
                
                #print len(hist_0[frame_snap-first_pic])
                first_pic -= 1
                pic_d.append((pic,k,1000,-1000,1000,-1000,-1000))
                #plt.hist(img.ravel(),256,[0,256]); plt.show()
            '''

        pickle.dump(pic_d,open("./annotations/full_viewpoint_annotations_2016/histograms/"+video+"_new.pkl","wb"))
        pic_d = pickle.load(open("./annotations/full_viewpoint_annotations_2016/histograms/"+video+"_new.pkl","rb"))
        #print pic_d, len(pic_d)
else:
    text_file = ""
    for video in videos:
        pic_d = pickle.load(open("./annotations/full_viewpoint_annotations_2016/histograms/"+video+"_new.pkl","rb"))
        
        debug = True
        write = False
        fup = []
        fdo = []

        M = 2

        for stability in drange(-0.1,1.0,0.1):
            for epsilon in drange(0.0,1.0,0.025):
                if debug:
                    stability = 0.0
                    epsilon = 0.0

                problems = 0
                hproblems = 0
                flag_up = False
                flag_counter = 0
                flag_counter2 = -20
                old = -1
                c = 0
                cc = 0
                flagmax = -1
                flagmax2 = -1
                flagi = -1
                flagi2 = -1

                for i in range(2,len(pic_d)-2):


                    if old == -1:
                        s = 1
                        c += 1
                        if debug:
                            print "FRAME UP", i-1
                            text_file += str(i-1)+" "+str(2)+"\n"
                            fup.append(pic_d[i][0])

                            if flag_up:
                                print "PROBLEM-----------------------------------------"
                                problems += 1
                            flag_up = True
                        else:
                            if flag_up:
                                problems += 1
                            flag_up = True
                    else:

                        s = [pic_d[i-2][M],pic_d[i-1][M],pic_d[i-0][M],pic_d[i+1][M],pic_d[i+2][M]]
                        
                        for o in range(len(s)):
                            #s[o] = (np.median(s[o])+min(s[o]))/2
                            w = 0.65
                            s[o] = (w*np.median(s[o])+(1-w)*min(s[o]))

                        #if i > 10300 and i < 10500:
                        #    print s, i

                        sense = stability + epsilon
                        sense_low = stability - epsilon
                        # the ranges are the problem
                        p_val = 1
                        if s[2] > sense and (s[3] > sense or s[4] > sense) and (s[1] < sense_low and s[0] < sense_low):
                            
                            if debug:
                                if i-flag_counter < 20:
                                    continue
                                if flag_up:
                                    print "PROBLEM-----------------------------------------"
                                    problems += 1
                                elif i - flag_counter < 10:
                                    print "HUGE PROBLEM-----------------------------------------"
                                    hproblems += 1

                                print "FRAME UP", i+1
                                text_file += str(i+1)+" "+str(2)+"\n"
                                fup.append(pic_d[i][0])

                                if i-flag_counter > flagmax:
                                    flagmax = i-flag_counter
                                    flagi = i

                                flag_up = True
                                flag_counter = i

                            else:
                                if i - flag_counter < 20:
                                    continue
                                if flag_up:
                                    problems += 1
                                flag_up = True
                                if i - flag_counter < 10:
                                    hproblems += 1
                                flag_counter = i


                            c += 1
                        if s[2] < sense_low and (s[3] < sense_low or s[4] < sense_low) and (s[1] > sense and s[0] > sense):
                            if debug:
                                if i-flag_counter2 < 20:
                                    continue
                                if not flag_up:
                                    print "PROBLEM-----------------------------------------"
                                    problems += 1
                                print "FRAME DOWN", i+1
                                text_file += str(i+1)+" "+str(0)+"\n"
                                fdo.append(pic_d[i][0])

                                flag_up = False

                                if i-flag_counter2 > flagmax2:
                                    flagmax2 = i-flag_counter2
                                    flagi2 = i
                                flag_counter2 = i
                            else:
                                if i-flag_counter2 < 20:
                                    continue
                                if not flag_up:
                                    problems += 1
                                flag_up = False

                                flag_counter2 = i
                            cc += 1
                    old = 5
                if abs(c - cc) < 10 and c > 150 and problems < 10 and hproblems <= 10:
                    print "stable",c, cc, stability,epsilon, problems, hproblems
                if debug:
                    print c, cc, stability,epsilon, problems, hproblems
                    print flagmax, flagi, flagmax2, flagi2

                if debug:
                    if write:
                        file_name = str(video)+"_c.txt"
                        f = open("./annotations/full_viewpoint_annotations_2016/"+file_name, "w")
                        f.write(text_file)
                        #print text_file
                        f.close()
                        for i in fup:
                            shutil.copy2(i, './autoannotations/'+str(video)+"/")
                        for i in fdo:
                            shutil.copy2(i, './autoannotations/'+str(video)+"/")
                        for i in fup:
                            shutil.copy2(i, './autoannotations_up/'+str(video)+"/")
                        for i in fdo:
                            shutil.copy2(i, './autoannotations_do/'+str(video)+"/")
                    # recursively copy all fup and fdo into a folder and zip it

                    exit()


