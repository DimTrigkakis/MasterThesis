import sys

sys.path.insert(0, './utils/')

import subprocess

import numpy as np
import argparse

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
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

from datasets import CustomDatasetViewpointIntervals, CustomDatasetViewpointFramewise
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

parser.add_argument("--arch",  "-a", metavar='ARCH', default='ED', choices=["VGG","ED","EB"], help="model architecture: " + ' | '.join(["VGG","ED","EB"]))
parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/full_frame_data_2016/",help="Path that contains the dataset to be processed")
parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=1, help="cnn batch size")
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=12, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=1, help="Number of epochs to train for")
parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=1, help="How often should the model save")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=1, help="How often to test the model, default = 5")
parser.add_argument("--training",metavar="TRAINING",default=None,help="Training phase of the classifier is activated")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default=1e-6,help="Weight decay for the optimizer")
parser.add_argument("--viewpoints",metavar="VIEWPOINTS",default=3,type=int,help="How many viewpoints to train on")

parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to fill with logger information")

mylogger = None

# Utilities
MO = MagnificentOracle()

class viewpoint_classifier_ED():

    def __init__(self, model="ED", path = None,viewpoints=3, interval_size=32, interval_samples_per_game=None,splitting="whole",overlap="consecutive"):

        interval_samples_per_game = 20000/interval_size 
        self.interval_size = interval_size

        # print len(self.testloader_acc) # 9128

        if (model == "VGG"):
            self.model_VGG = VGG_viewpoints(num_classes=viewpoints).cuda()
            self.model_VGG.soft = nn.LogSoftmax()
            self.model_VGG.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_framewise/models/model_epoch_10.pth"))
            #customset_validation = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="whole",overlap="consecutive", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size,only_extract=False)
            #self.valloader_acc = torch.utils.data.DataLoader(dataset=customset_validation,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)  
            #customset_validation2 = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="whole",overlap="sliding", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size,only_extract=False)
            #self.valloader_acc2 = torch.utils.data.DataLoader(dataset=customset_validation2,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)    
            customset_validation3 = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="20",overlap="consecutive", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size,only_extract=False)
            self.valloader_acc3 = torch.utils.data.DataLoader(dataset=customset_validation3,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)  
            #customset_validation4 = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="20",overlap="sliding", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size,only_extract=False)
            #self.valloader_acc4 = torch.utils.data.DataLoader(dataset=customset_validation4,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

        elif (model == "ED"):
            self.model_VGG = VGG_viewpoints(num_classes=viewpoints, mode="features").cuda()
            #self.model_VGG.soft = nn.LogSoftmax()
            #self.model_VGG.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_framewise/models/model_epoch_10.pth"))
            mod = list(self.model_VGG.classifier.children())
            for i in range(3):
                mod.pop()
            self.model_VGG.classifier = torch.nn.Sequential(*mod)
            self.model_ED = EncoderDecoderViewpoints(max_len=interval_size).cuda() # not on multiple gpus, since it needs to not distribute the interval images
            self.model_VGG = nn.DataParallel(self.model_VGG,device_ids=[0,1,2,3]).cuda() # this is per image, so we can distribute over 4 gpus
            self.model_ED.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_intervals_frozen/model/model_epoch_2_32_ed.pth"))
            self.model_VGG.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_intervals_frozen/model/model_epoch_2_32_vgg.pth"))

            customset_validation3 = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="whole",overlap="consecutive", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size,only_extract=False)
            self.valloader_acc3 = torch.utils.data.DataLoader(dataset=customset_validation3,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)  
           
        
        elif (model == "EB"):
            self.model_ED = EncoderDecoderViewpoints(max_len=interval_size).cuda()
            #self.model_ED.load_state_dict(torch.load("/scratch/datasets/NFLsegment/experiments/viewpoint_intervals/models/model_epoch_2_32_ed.pth"))
            self.model_ED.init_VGG(model="/scratch/datasets/NFLsegment/experiments/viewpoint_framewise/models/model_epoch_10.pth")

        if model != "EB" and args.training:
            self.optimizer = optim.Adam(list(self.model_VGG.parameters()) + list(self.model_ED.parameters()), weight_decay=float(args.weight_decay), lr=0.0001)
            self.criterion = nn.NLLLoss().cuda()
            mylogger.log(self.model_VGG)

        print(len(self.valloader_acc3))
        

# Train NN
def train(epochs, vp):
    
    if not args.training:
        args.test_interval = 1
        args.save_interval = 10000
    batch_interval_print = 1


    for epoch in range(1, 2):  
        #mylogger.log(" Epoch :"+str(epoch))
        #vp.model_VGG.train()
        #vp.model_ED.train()
        
        if (epoch % int(args.test_interval) == 0):
            acc(epoch,vp)
  

def acc(epoch, vp):


    for loader in [vp.valloader_acc3]:

        vp.model_VGG.eval()
        vp.model_ED.eval()

        loss = 0
        correct = 0.0
        all_examples = 0.0
        all_batches = 0
        conf_mat = np.zeros((3,3))
        
        for batch_idx, (data, target, video_name)  in enumerate(loader):

            print batch_idx
            target.squeeze_()
            target = Variable(target.cuda())
            data = Variable(data.cuda()).squeeze(0)

            features = vp.model_VGG(data)
            output = vp.model_ED(features)

            #loss += vp.criterion(output, target).data[0]
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            all_examples += data.size()[0]
            flat_pred = pred.cpu().numpy().flatten()
            np_target = target.data.cpu().numpy()

           
            #for i in range(np_target.shape[0]):
            #   conf_mat[int(np_target[i])][int(flat_pred[i])] += 1   

            for i in range(np_target.shape[0]):
                mylogger.log(str(np_target[i])+str(flat_pred[i])+str(video_name))   

            print(correct/all_examples)
        

        '''
        myset = "testing"
        print(correct/all_examples, correct)
        mylogger.log('\n Accuracy on '+myset+' set: , Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, all_examples,
            100.0 * correct / all_examples))

        mylogger.log("-- Confusion Matrix --")
        conf_mat = np.multiply(conf_mat,1/all_examples)
        mylogger.log(conf_mat)
        '''
from viterbi import viterbi

def post_process(folder=None):

    viterbify = True

    if viterbify:
        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/IoU_testing_prediction.txt","r")
        lines = myfile.readlines()
        # produce lines after viterbi here
        # recreate IoU_validation_prediction with viterbi

        last_game = -1
        game_frames = {}
        ddd = 0
        game_list = []
        difference = 0
        str_for_game = ""
        for i , line in enumerate(lines):
            if "2016" in line:
                ddd += 1
                line_decoded = line.split("-->")[1]
                line_decoded = line_decoded.split("('")
                target, prediction, game = int(line_decoded[0][0:1]), int(line_decoded[0][1:2]), line_decoded[1].split("'")[0]

                if last_game != game and last_game != -1:
                    # you need to gather the entire prediction and viterbify, then save it back
                    #print(len(game_list))
                    # viterbification

                    print(last_game, "last game was this")
                    numerical_instability = 10
                    for mmm in range(numerical_instability):
	                    obs = []
	                    for j in range(mmm*len(game_list)/numerical_instability,(mmm+1)*len(game_list)/numerical_instability):
	                        obs.append(game_list[j][1])


	                    s = [0,1,2]
	                    start_p = [0.3,0.3,0.3]
	                    k = 0.99 # 0.99 #emission p
	                    k2 = 0.9872 #0.9872 # transmission p
	                    e_t = [k, (1-k)/2, (1-k)/2]
	                    t_t = [k2, 1-k2, 0.00]

	                    e = [[e_t[0],e_t[1],e_t[2]],[e_t[2],e_t[0],e_t[1]],[e_t[1],e_t[2],e_t[0]]]
	                    t = [[t_t[0],t_t[1],0],[0,t_t[0],t_t[1]],[t_t[1],0,t_t[0]]]

	                    results = viterbi(obs,s,start_p,t,e)
	                    print(obs)
	                    print(results)
	                    results2 = obs
	                    #final_list = []
	                    count_mistakes = 0
	                    count_mistakes2 = 0
	                    for j in range(mmm*len(game_list)/numerical_instability,(mmm+1)*len(game_list)/numerical_instability):
	                        if game_list[j][0] != results[j-mmm*len(game_list)/numerical_instability]:
	                            count_mistakes += 1
	                        if game_list[j][0] != results2[j-mmm*len(game_list)/numerical_instability]:
	                            count_mistakes2 += 1
	                        
	                        #final_list.append((game_list[j][0],results[j],game_list[j][2]))
	                        str_for_game += " -->{}{}('{}',)\n".format(str(game_list[j][0]),str(results[j-mmm*len(game_list)/numerical_instability]), str(game_list[j][2]))
	                    #print str_for_game


                    difference += count_mistakes2 - count_mistakes

                    print(count_mistakes, count_mistakes2, len(game_list))

                    game_list = []

                #if prediction != target:
                #    print prediction, target, ddd
                game_list.append((target,prediction, game))


                last_game = game

        print(difference)

        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/IoU_testing_prediction_viterbi.txt","w")
        myfile.write(str_for_game)
        myfile.close()

    if viterbify:
        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/IoU_testing_prediction_viterbi.txt","r")
    else:
        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/IoU_testing_prediction.txt","r")

    lines = myfile.readlines()
    game_frames = {}
    incorrect, correct = 0, 0

    last_target = 2
    last_game = -1
    ddd = 0
    game_num = 0

    last_predictions = 2
    last_targets = 2
    counters = 0

    text_string = ""
    game_counter_lines = 0
    count_transitions = 0
    count_target_transitions = 0
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
                    if viterbify:
                        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/"+last_game+"_c_estimated_viterbi_testing.txt","w")
                    else:
                        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/"+last_game+"_c_estimated_testing.txt","w")
                    myfile.write(text_string)
                else:
                    print(game)

                target_change_frame = {"02":-1,"10":-1,"21":-1}
                ddd = 0
                last_prediction = -1
                last_targets = 2
                last_predictions = 2
                text_string = "1 2\n"
                game_counter_lines = 0

            last_game = game


            #if (last_predictions + 1) % 3 == prediction and prediction != last_predictions:
            #    count_transitions[0] += 1
            if prediction != last_predictions:
                text_string += str(ddd+1)+" "+str(prediction)+"\n"    
                count_transitions += 1        
            if target != last_targets:
                count_target_transitions += 1
                
            last_predictions = prediction
            last_targets = target

            game_frames[game].append((target, prediction))

            if prediction != target:
                incorrect += 1
                keep = True
                #print(prev_predictions, last_targets)
            else:
                correct += 1

           

            #print("t:",target,"p:",prediction,"g:", game)
    #print(incorrect,"/",incorrect+correct," errors")
    #print(minimum_length)
    if viterbify:
        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/"+last_game+"_c_estimated_viterbi_testing.txt","w") 
    else:
        myfile = open("./experiments/final_viewpoint_experiments/"+folder+"/"+last_game+"_c_estimated_testing.txt","w") 

    myfile.write(text_string)
    print(incorrect,"/",incorrect+correct," errors")
    print("FINAL TRANSITIOn", count_transitions , count_target_transitions)

    print(correct*1.0/(incorrect+correct))

def IoU(folder=None):
    difference = 0
    iou_all_games = 0
    iou_all_games0 = 0
    iou_all_games1 = 0
    iou_all_games2 = 0

    total_b, total_B = 0,0
    total_C = 0
    mylogger.log("------------- FOR FOLDER ---------- {}".format(folder))
    for game in ["2016121110","2016122411"]:#["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108"]:
        gt = open("./annotations/full_viewpoint_annotations_2016/"+game+"_c.txt","r")
        prediction = open("./experiments/final_viewpoint_experiments/"+folder+"/"+game+"_c_estimated_viterbi_testing.txt","r") 

        lines_gt = gt.readlines() 
        lines_pd = prediction.readlines()

        start_with = (1,2)
        gt_list = []
        for i, line in enumerate(lines_gt):
            #if i > len(lines_gt)*0.2:
            #    break

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

        difference += abs(len(pd_list)-len(gt_list))
        mylogger.log(game+" viewpoint imbalance is "+str((abs(len(pd_list)-len(gt_list)))/3.0))
        #print len(pd_list), len(gt_list)

        mylogger.log("game: "+str(game)+" all classes "+str(IoU_pairs(pd_list, gt_list)))
        IoU, Ilen, Blen, b, tpfp =  IoU_pairs(pd_list, gt_list)
        total_b += b
        total_B += Blen
        total_C += tpfp
        iou_all_games += IoU/Ilen
        mylogger.log("game: "+str(game)+" class view 0 "+str(IoU_pairs(pd_list, gt_list,classes=[0])))
        IoU, Ilen, Blen, b, _ =  IoU_pairs(pd_list, gt_list,classes=[0])
        iou_all_games0 += IoU/Ilen
        mylogger.log("game: "+str(game)+" class view 1 "+str(IoU_pairs(pd_list, gt_list,classes=[1])))
        IoU, Ilen, Blen, b, _ =  IoU_pairs(pd_list, gt_list,classes=[1])
        iou_all_games1 += IoU/Ilen
        mylogger.log("game: "+str(game)+" class scoreboard "+str(IoU_pairs(pd_list, gt_list,classes=[2])))
        IoU, Ilen, Blen, b, _ =  IoU_pairs(pd_list, gt_list,classes=[2])
        iou_all_games2 += IoU/Ilen
    mylogger.log("all games, all classes, recall is "+str(total_b*1.0/total_B))
    mylogger.log("all games, all classes, mAP is "+str(total_b*1.0/total_C))
    print("difference for folder is ", difference)
    mylogger.log("all games, all classes "+str(iou_all_games/2.0))
    mylogger.log("all games, view 0 "+str(iou_all_games0/2.0))
    mylogger.log("all games, view 1 "+str(iou_all_games1/2.0))
    mylogger.log("all games, scoreboard "+str(iou_all_games2/2.0))
    print(str(iou_all_games/2.0))

def EB_extraction(vp):
    print("EB extraction, 3 levels, sequence and non-sequence")

    for loader in [vp.testloader_acc]:

        vp.model_ED.eval()

        eb.use_eb(True)

        print("Starting EB extraction")
        print(len(loader))

        counter = 0
        all_examples = 0
        
        for batch_idx, (data, target, video_name)  in enumerate(loader):

            timer = time.time()
            target.squeeze_()
            target = Variable(target)
            data = data[0]
            data = Variable(data)

            #print data.size(), target.size(), video_name

            #features = vp.model_VGG(data)

            
            print time.time()-timer
            output = vp.model_ED(data)

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
            
            print time.time()-timer

            layer_top = list(vp.model_ED.modules())[0].classifier[6] 
            layer_second = list(vp.model_ED.modules())[0].classifier[4] 
            target_layer = list(vp.model_ED.modules())[0].features[0] # 3,4,5 or 6
            top_dh = torch.zeros(32,3)

            print np_target
            for i in range(32):
                top_dh[i,np_target[i]] = 1 # ground truth based contrastive signal
            print top_dh

            mylogger.log("Using eb")
            grad = eb.contrastive_eb(vp.model_ED, data, layer_top, layer_second, target=target_layer, top_dh=top_dh)
            mylogger.log("Time void from contrastive EB: {}".format(time.time()-timer))

            print grad.numpy().shape
            grad_ = grad.numpy()[:,:,:,:]
            grad_ = np.mean(grad_,axis=1)

            np.save("./experiments/viewpoint_EB_data/"+str(batch_idx)+"_"+str(video_name[0])+"_attentions.npy", grad_)
            np.save("./experiments/viewpoint_EB_data/"+str(batch_idx)+"_"+str(video_name[0])+"_original.npy", data.data.numpy())
            np.save("./experiments/viewpoint_EB_data/"+str(batch_idx)+"_"+str(video_name[0])+"_targets.npy", target.data.numpy())

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
            print time.time()-timer
            
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

    EB = False
    postprocess = False
    IoU_ = True

    if not EB:
        if not postprocess:
            if not IoU_:
                vp = viewpoint_classifier_ED(args.arch, path=args.dataset_path, viewpoints = args.viewpoints)
                train(int(args.epochs), vp) 
            else:
                for folder in ["ED32_vgg_frozen"]: #["framewise","ED32","ED196","ED32_vgg_frozen"]:
                    IoU(folder=folder)
        else:
            for folder in ["ED32_vgg_frozen"]: #["framewise","ED32","ED196","ED32_vgg_frozen"]:
                print("post-processing")
                post_process(folder=folder)
    else:
        vp = viewpoint_classifier_ED(args.arch, path=args.dataset_path, viewpoints = args.viewpoints)
        EB_extraction(vp)

if __name__ == "__main__":

    main()


