import sys

sys.path.insert(0, './utils/')

import subprocess

import numpy as np
import argparse

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as models

import PIL
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import xml.etree.ElementTree as ET
import os
import math
import random
import pickle
import csv
import time

from datasets import CustomDatasetViewpointIntervals
from models import EncoderDecoderViewpoints, VGG_viewpoints

from extract_videos import extract_frames, deconstruct_video
from construct_video import construct_annotator
from IOUmetric import IoU_pairs

from viterbi import viterbi
import torch.backends.cudnn as cudnn
from magnificentoracle import MagnificentOracle

import excitation_bp as eb
# cudnn speedups
cudnn.benchmark = True
cudnn.fastest = True

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')

parser.add_argument("--arch",  "-a", metavar='ARCH', default='EB11', choices=["VGG","ED","EB","EB11"], help="model architecture: " + ' | '.join(["VGG","ED","EB"]))
parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/full_frame_data_2016/",help="Path that contains the dataset to be processed")
parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=1, help="cnn batch size")
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=12, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=10, help="Number of epochs to train for")
parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=1, help="How often should the model save")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=1, help="How often to test the model, default = 5")
parser.add_argument("--pretrained_framewise",metavar="PRETRAINED_framewise",default="/scratch/datasets/NFLsegment/experiments/viewpoint_framewise/models/model_epoch_10.pth",help="When extracting frames for labelling, points to the pretrained classifier file")
parser.add_argument("--pretrained_same_architecture",metavar="PRETRAINED_SAME_ARCHITECTURE",default=False,help="ED pretrained model")
parser.add_argument("--training",metavar="TRAINING",default=None,help="Training phase of the classifier is activated")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default=1e-6,help="Weight decay for the optimizer")
parser.add_argument("--viewpoints",metavar="VIEWPOINTS",default=3,type=int,help="How many viewpoints to train on")

parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to fill with logger information")

mylogger = None

# Utilities
MO = MagnificentOracle()

class viewpoint_classifier_ED():

    def __init__(self, model="EB11", path = None,viewpoints=3, interval_size=32, interval_samples_per_game=None,splitting="whole",overlap="consecutive"):
        
        interval_samples_per_game = 20000/interval_size 
        self.interval_size = interval_size
        self.model = model
        customset_test = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="whole",overlap="consecutive", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size,only_extract=True)

        self.testloader_acc = torch.utils.data.DataLoader(dataset=customset_test,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        
        # print len(self.testloader_acc) # 9128


        if (model == "VGG"):
            self.model = VGG_viewpoints(num_classes=viewpoints).cuda()
            self.model.soft = nn.LogSoftmax()
        elif (model == "ED"):
            self.model_VGG = VGG_viewpoints(num_classes=viewpoints, mode="features").cuda()
            self.model_VGG.soft = nn.LogSoftmax()
            self.model_VGG.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_framewise/models/model_epoch_10.pth"))
            mod = list(self.model_VGG.classifier.children())
            for i in range(3):
                mod.pop()
            self.model_VGG.classifier = torch.nn.Sequential(*mod)
            self.model_ED = EncoderDecoderViewpoints(max_len=interval_size).cuda() # not on multiple gpus, since it needs to not distribute the interval images
            self.model_VGG = nn.DataParallel(self.model_VGG,device_ids=[0,1,2,3]).cuda() # this is per image, so we can distribute over 4 gpus
            self.model_ED.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_intervals/models/model_epoch_2_32_ed.pth"))
            self.model_VGG.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_intervals/models/model_epoch_2_32_vgg.pth"))
        elif (model == "EB11"):
            self.model_VGG = VGG_viewpoints(num_classes=viewpoints, mode="features")
            self.model_VGG.soft = nn.LogSoftmax()
            self.model_VGG.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_framewise_VGG11/model_epoch_5.pth"))
        elif (model == "EB"):
            self.model_ED = EncoderDecoderViewpoints(max_len=interval_size).cuda()
            #self.model_ED.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_intervals/models/model_epoch_2_32_ed.pth"))
            self.model_ED.init_VGG(model="/scratch/datasets/NFLsegment/experiments/viewpoint_framewise/models/model_epoch_10.pth")

        if model != "EB11":
            self.optimizer = optim.Adam(list(self.model_VGG.parameters()) + list(self.model_ED.parameters()), weight_decay=float(args.weight_decay), lr=0.0001)
            self.criterion = nn.NLLLoss().cuda()
            mylogger.log(self.model_VGG)

        #mylogger.log(self.model_ED)
        print("EBBBB")

# Train NN
def train(epochs, vp):
    
    if not args.training:
        args.test_interval = 1
        args.save_interval = 10000
    batch_interval_print = 1


    for epoch in range(1, epochs + 1):  
        mylogger.log(" Epoch :"+str(epoch))
        vp.model_VGG.train()
        vp.model_ED.train()
        
        if (epoch % int(args.test_interval) == 0):
            acc(epoch,vp)
  

def acc(epoch, vp):

    save_consecutive = False
    produce_acc = True

    for loader in [vp.testloader_acc]:

        vp.model_VGG.eval()
        vp.model_ED.eval()

        loss = 0
        correct = 0
        all_examples = 0.0
        all_batches = 0
        conf_mat = np.zeros((3,3))
        
        for batch_idx, (data, target, video_name)  in enumerate(loader):

            all_batches += 1

            target.squeeze_()
            target = Variable(target.cuda())
            data = Variable(data.cuda()).squeeze(0)


            features = vp.model_VGG(data)
            output = vp.model_ED(features)

            loss += vp.criterion(output, target).data[0]
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            all_examples += data.size()[0]
            flat_pred = pred.cpu().numpy().flatten()
            np_target = target.data.cpu().numpy()

            if save_consecutive:
                for i in range(np_target.shape[0]):
                    mylogger.log(str(np_target[i])+str(flat_pred[i])+str(video_name))

def post_process():
    myfile = open("./experiments/viewpoint_intervals_extract/log_foundation.txt","r")
    lines = myfile.readlines()
    game_frames = {}
    incorrect, correct = 0, 0

    last_target = 2
    last_game = -1
    ddd = 0
    game_num = 0
    switch_gt = [0,0,game_num]
    switch = [0,0,game_num]
    change = {"02":[2,0],"10":[0,1],"21":[1,2]}
    remember = {"02":0, "10":0, "21":0}

    threshold_length = 6 # minimum length in a cycle of 2-0-1 forms, and out of all cycles, the smallest is 18
    KAPPA = 3
    KAPPA_target = 18
    last_targets = [2 for x in range(KAPPA_target)]
    last_predictions = [2 for x in range(KAPPA)]
    keep = False
    counters = 0

    target_change_frame = {"02":-1,"10":-1,"21":-1}
    target_minimum = {"02":1000,"10":1000,"21":1000}

    count_transitions = [0,0]
    tp, fp, fn = 0,0,0
    text_string = ""
    game_counter_lines = 0
    for i, line in enumerate(lines):
        if "2016" in line:
            ddd += 1
            line_decoded = line.split("-->")[1]
            line_decoded = line_decoded.split("('")
            target, prediction, game = int(line_decoded[0][0:1]), int(line_decoded[0][1:2]), line_decoded[1].split("'")[0]

            if game not in game_frames.keys():
                game_frames[game] = []

            if last_game != game or last_game == -1:
                print((len([x for x in text_string if x == "\n"]))/3, game_counter_lines/3)
                if last_game != -1:
                    #print(text_string)
                    myfile = open("./annotations/full_viewpoint_annotations_2016/"+last_game+"_c_estimated.txt","w")
                    myfile.write(text_string)
                else:
                    print(game)

                target_change_frame = {"02":-1,"10":-1,"21":-1}
                ddd = 0
                last_prediction = -1
                last_targets = [2 for x in range(KAPPA_target)]
                last_predictions = [2 for x in range(KAPPA)]
                text_string = "1 2\n"
                game_counter_lines = 0

            last_game = game

            #if not((last_predictions[KAPPA/2]+1) % 3 == prediction or last_predictions[KAPPA/2] == prediction):
            #    pass #prediction = (last_predictions[KAPPA/2]+1) % 3

            # Count how many proper changes occur in the dataset

            if (last_predictions[-1] + 1) % 3 == prediction and prediction != last_predictions[-1]:
                #if last_predictions[-1] != 0 or prediction != 1:
                count_transitions[0] += 1
                text_string += str(ddd-KAPPA+4)+" "+str(prediction)+"\n"
                #print("transition", ddd, game, last_predictions)
                #kp_correction = last_predictions[-1]
                #prediction = kp_correction
            elif (last_predictions[-1] + 1) % 3 != prediction and prediction != last_predictions[-1]:
                count_transitions[1] += 1
                #print(last_predictions[-1])
                prediction = last_predictions[-1]
                # corrective action

            if (last_predictions[KAPPA/2] != last_predictions[-1+KAPPA/2]):
                change[str(last_predictions[KAPPA/2])+str(last_predictions[-1+KAPPA/2])] = [last_predictions[-1+KAPPA/2],last_predictions[KAPPA/2]]
                remember[str(last_predictions[KAPPA/2])+str(last_predictions[-1+KAPPA/2])] = ddd-KAPPA/2
            
            if (last_targets[0] + 1) % 3 != last_targets[1] and last_targets[0] != last_targets[1]:
                print("ERROR", ddd, game)

            if last_targets[1] != last_targets[0]: # minimum time between two consecutive changes is 10 frames (a scoreboard view)
                
                game_counter_lines += 1
                str_change = str(last_targets[1])+str(last_targets[0])
                
                if target_change_frame[str_change] != -1:       
                    #if ddd-KAPPA_target-target_change_frame < minimum_length:
                    #print(ddd-KAPPA_target-target_change_frame[str_change], last_targets)             
                    target_minimum[str_change] = min(ddd-KAPPA_target-target_change_frame[str_change],target_minimum[str_change])
                    target_change_frame[str_change] = ddd-KAPPA_target
                    #print("min",target_minimum)
                else:
                    target_change_frame[str_change] = 1
                    #print("First frame", game)
                
                
                switch_gt[last_targets[1]] += 1
                #print("change",last_targets[1], last_targets, ddd-KAPPA_target)
                if abs(ddd-KAPPA_target - remember[str_change]) < threshold_length and change[str_change][0] == last_targets[0] and change[str_change][1] == last_targets[1]: # these frames are absolute, not relative to memory
                    switch[last_targets[1]] += 1
                    tp += 1
                else:
                    if last_targets[1] != 1:
                        #print("Mistake in change above 10 frames (min length , same as target memory):",str_change, remember[str_change], ddd-KAPPA_target, change[str_change][0], last_targets[0])
                        counters += 1
                    fn += 1

                
            for j in range(KAPPA-1):
                last_predictions[j] = last_predictions[j+1]
            last_predictions[KAPPA-1] = prediction


            for j in range(KAPPA_target-1):
                last_targets[j] = last_targets[j+1]
            last_targets[KAPPA_target-1] = target

            game_frames[game].append((target, prediction))

            if prediction != target:
                incorrect += 1
                keep = True
                #print(prev_predictions, last_targets)
            else:
                correct += 1

           

            #print("t:",target,"p:",prediction,"g:", game)
    fp = count_transitions[0]-tp
    #print(incorrect,"/",incorrect+correct," errors")
    print(switch,switch_gt)
    print("playtype vs scoreboard errors:",counters,"prediction errors:",2165-switch[0]+2163-switch[1]+2165-switch[2],"ground truth",2165-switch_gt[0]+2163-switch_gt[1]+2165-switch_gt[2])
    #print(minimum_length)
    print("Valid transitions that have been counted: ",count_transitions[0]," instead of ",sum(switch_gt), count_transitions[1]," are invalid")
    print(tp,fp,fn)

    print("Recall is : "+str(tp*1.0/(tp+fn)))
    print("Precision is : "+str(tp*1.0/(tp+fp)))

    myfile = open("./annotations/full_viewpoint_annotations_2016/"+last_game+"_c_estimated.txt","w")
    myfile.write(text_string)

def IoU():
    iou_all_games = 0
    iou_all_games0 = 0
    iou_all_games1 = 0
    iou_all_games2 = 0
    for game in ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108","2016121110","2016122411"]:
        gt = open("./annotations/full_viewpoint_annotations_2016/"+game+"_c.txt","r")
        prediction = open("./annotations/full_viewpoint_annotations_2016/"+game+"_c_estimated.txt","r")

        lines_gt = gt.readlines()
        lines_pd = prediction.readlines()
        print "Calculating IoU per class for game",game
        print game, "viewpoint imbalance is", len(lines_gt)/3.0-len(lines_pd)/3.0

        start_with = (1,2)
        gt_list = []
        for line in lines_gt:

            t = line.split(" ")
            t = (int(t[0])-1, int(t[1].strip("\n")))
            if t[0] == 0:
                continue

            gt_list.append((start_with, (t[0],start_with[1])))
            start_with = (t[0]+1,t[1])

        start_with = (1,2)
        pd_list = []
        for line in lines_pd:

            t = line.split(" ")
            t = (int(t[0])-1, int(t[1].strip("\n")))
            if t[0] == 0:
                continue

            pd_list.append((start_with, (t[0],start_with[1])))
            start_with = (t[0]+1,t[1])

        #print pd_list, gt_list

        mylogger.log("game: "+str(game)+" all classes "+str(IoU_pairs(pd_list, gt_list)))
        iou_all_games += IoU_pairs(pd_list, gt_list)
        mylogger.log("game: "+str(game)+" class view 0 "+str(IoU_pairs(pd_list, gt_list,classes=[0])))
        iou_all_games0 += IoU_pairs(pd_list, gt_list,classes=[0])
        mylogger.log("game: "+str(game)+" class view 1 "+str(IoU_pairs(pd_list, gt_list,classes=[1])))
        iou_all_games1 += IoU_pairs(pd_list, gt_list,classes=[1])
        mylogger.log("game: "+str(game)+" class scoreboard "+str(IoU_pairs(pd_list, gt_list,classes=[2])))
        iou_all_games2 += IoU_pairs(pd_list, gt_list,classes=[2])
    mylogger.log("all games, all classes "+str(iou_all_games/13.0))
    mylogger.log("all games, view 0 "+str(iou_all_games0/13.0))
    mylogger.log("all games, view 1 "+str(iou_all_games1/13.0))
    mylogger.log("all games, scoreboard "+str(iou_all_games2/13.0))

def EB_extraction(vp):
    print("EB extraction, 3 levels, sequence and non-sequence")

    for loader in [vp.testloader_acc]:

        vp.model_VGG.eval()

        eb.use_eb(True)

        print("Starting EB extraction")
        print(len(loader))

        counter = 0
        all_examples = 0
        last_video = "-"
        last_batch = 0
        
        n_image = 0
        for batch_idx, (data, target, video_name)  in enumerate(loader):
            
            #if video_name[0] == "2016090800": # already did it before
            #    continue

            if last_video != video_name[0]:
                print("video ", video_name[0])
                last_batch = 0

            last_video = video_name[0]
            if last_batch >= 32:
                continue
            else:
                last_batch += 1

            print(last_batch, batch_idx, video_name[0])

            # remove continue and exit()

            timer = time.time()
            target.squeeze_()
            target = Variable(target)
            data = data[0]
            data = Variable(data)

            #print data.size(), target.size(), video_name

            #features = vp.model_VGG(data)

            
            #print time.time()-timer
            output = vp.model_VGG(data)

            pred = output.data.max(1)[1] 
            all_examples += data.size()[0]
            flat_pred = pred.cpu().numpy().flatten()
            np_target = target.data.cpu().numpy()


            correct = pred.eq(target.data).sum()
            #print(output) # don't even need output

            if correct == 32:
                counter += 1
                print("all correct here")
            else:
                print(correct,"/",32 , "not correct...")
            
            #print time.time()-timer

            layer_top = list(vp.model_VGG.modules())[0].classifier[6] 
            layer_second = list(vp.model_VGG.modules())[0].classifier[4] 
            target_layer = list(vp.model_VGG.modules())[0].features[2] # 3,4,5 or 6
            top_dh = torch.zeros(32,3)

            #print np_target
            for i in range(32):
                top_dh[i,np_target[i]] = 1 # ground truth based contrastive signal
            #print top_dh

            mylogger.log("Using eb")
            grad = eb.contrastive_eb(vp.model_VGG, data, layer_top, layer_second, target=target_layer, top_dh=top_dh)
            mylogger.log("Time void from contrastive EB: {}".format(time.time()-timer))

            #print grad.numpy().shape

            grad_ = grad.numpy()[:,:,:,:]
            grad_ = np.mean(grad_,axis=1)

            prediction = pred.cpu().numpy()
            for j in range(32):
                mytensor = torch.from_numpy(grad_[j]).unsqueeze(0).unsqueeze(0)                
                print(mytensor.size())
                vutils.save_image(mytensor,"./experiments/viewpoint_EB_final_data_proper/sample_"+str(batch_idx).zfill(4)+"_"+str(j+n_image*32).zfill(4)+"_"+str(int(prediction[j]))+".png", normalize=True)
                print(data.size())
                vutils.save_image(data[j].data,"./experiments/viewpoint_EB_final_data_proper/sample_orig_"+str(batch_idx).zfill(4)+"_"+str(j+n_image*32).zfill(4)+"_"+str(int(prediction[j]))+".png", normalize=False)
            

            n_image += 1
            continue


            np.save("./experiments/viewpoint_EB_final_data_proper/"+str(last_batch)+"_"+str(video_name[0])+"_attentions.npy", grad_)
            np.save("./experiments/viewpoint_EB_final_data_proper/"+str(last_batch)+"_"+str(video_name[0])+"_original.npy", data.data.numpy())
            np.save("./experiments/viewpoint_EB_final_data_proper/"+str(last_batch)+"_"+str(video_name[0])+"_targets.npy", target.data.numpy())

            '''
            plt.figure(1)
            for i in range(32):
                img = grad_[i,:,:]
                plt.subplot(6,6,i+1)
                plt.imshow(img)
                plt.draw()
                plt.axis('off')
            plt.show()
            '''
            #print time.time()-timer

            
        print("Correct total sequences from 9128 to ", counter)

def main():

    # The basic core takes as input frames from the two camera viewpoints that look into the playing field
    # and learns proper kernels for recognizing the scene
    # This is a baseline to use as a pretrained classifier for recognizing entire playtypes in the advanced core

    global args, mylogger

    args = parser.parse_args()

    mylogger = MagnificentOracle()
    mylogger.set_log(logfile=args.log)
    mylogger.log("-dotted-line")  

    EB = True
    postprocess = False
    IoU = False

    if not EB:

        if ("postprocess" not in args.log):
            if ("IoU" not in args.log):
                vp = viewpoint_classifier_ED(args.arch, path=args.dataset_path, viewpoints = args.viewpoints)
                train(int(args.epochs), vp) 
            else:
                print("IoU scores per class")
                IoU()
        else:
            print("post-processing")
            post_process()
    else:
        vp = viewpoint_classifier_ED(args.arch, path=args.dataset_path, viewpoints = args.viewpoints)
        EB_extraction(vp)

if __name__ == "__main__":

    main()


