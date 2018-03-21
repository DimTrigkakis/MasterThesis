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

from datasets import CustomDatasetViewpointIntervals
from models import EncoderDecoderViewpoints, VGG_viewpoints

from extract_videos import extract_frames, deconstruct_video
from construct_video import construct_annotator

from viterbi import viterbi
import torch.backends.cudnn as cudnn
from magnificentoracle import MagnificentOracle

# cudnn speedups
cudnn.benchmark = True
cudnn.fastest = True

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')

parser.add_argument("--arch",  "-a", metavar='ARCH', default='ED', choices=["VGG","ED"], help="model architecture: " + ' | '.join(["VGG","ED"]))
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

    def __init__(self, model="VGG", path = None,viewpoints=3, interval_size=32, interval_samples_per_game=None,splitting="whole",overlap="consecutive"):
        
        interval_samples_per_game = 20000/interval_size 
        self.interval_size = interval_size
        customset_train = CustomDatasetViewpointIntervals(path = path,subset_type="training",viewpoints = viewpoints,splitting="whole",overlap="consecutive", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size)
        #customset_test = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="whole",overlap="consecutive", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size)
        #customset_test_v2 = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="whole",overlap="sliding", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size)
        customset_test_v3 = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="20",overlap="consecutive", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size)
        #customset_test_v4 = CustomDatasetViewpointIntervals(path = path,subset_type="testing",viewpoints = viewpoints,splitting="20",overlap="sliding", interval_samples_per_game = interval_samples_per_game,interval_size=interval_size)

        self.trainloader = torch.utils.data.DataLoader(dataset=customset_train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
        self.trainloader_acc = torch.utils.data.DataLoader(dataset=customset_train,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        #self.testloader_acc = torch.utils.data.DataLoader(dataset=customset_test,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        #self.testloader_acc_v2 = torch.utils.data.DataLoader(dataset=customset_test_v2,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        self.testloader_acc_v3 = torch.utils.data.DataLoader(dataset=customset_test_v3,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        #self.testloader_acc_v4 = torch.utils.data.DataLoader(dataset=customset_test_v4,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        

        #print len(self.trainloader_acc) # 858
        #print len(self.trainloader)
        #print len(self.testloader_acc) # 193
        #print len(self.testloader_acc_v2) # 1967
        #print len(self.testloader_acc_v3) # 185
        #print len(self.testloader_acc_v4) # 1833


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

        self.optimizer = optim.Adam(self.model_ED.parameters(), weight_decay=float(args.weight_decay), lr=0.0001)
        self.criterion = nn.NLLLoss().cuda()
        mylogger.log(self.model_VGG)
        mylogger.log(self.model_ED)

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

        
        if args.training:
            for batch_idx, (data,target, video) in enumerate(vp.trainloader):

                target.squeeze_()
                target = Variable(target.cuda())

                #img = MO.visual_tensor(data)

                data = Variable(data.cuda()).squeeze(0)

                print(data.size())

                features = vp.model_VGG(data)
                output = vp.model_ED(features)

                print(output.size())

                loss = vp.criterion(output, target)
                
                vp.optimizer.zero_grad()
                loss.backward()
                vp.optimizer.step()

                pred = output.data.cpu().max(1)[1] 

                if batch_idx % batch_interval_print == batch_interval_print-1:
                    mylogger.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, len(vp.trainloader),
                        100. * batch_idx / len(vp.trainloader), loss.data[0]))
        
        if (epoch % int(args.test_interval) == 0):
            acc(epoch,vp)

        if (args.save_dir is not None and epoch % int(args.save_interval) == 0):
            directory = args.save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(vp.model_VGG.state_dict(), os.path.join(directory, "model_epoch_" + str(epoch)+"_"+str(vp.interval_size)+"_vgg.pth"))   
            torch.save(vp.model_ED.state_dict(), os.path.join(directory, "model_epoch_" + str(epoch)+"_"+str(vp.interval_size)+"_ed.pth"))   

def acc(epoch, vp):

    for loader in [ vp.trainloader_acc, vp.testloader_acc_v3]:

        vp.model_VGG.eval()
        vp.model_ED.eval()

        loss = 0
        correct = 0
        all_examples = 0.0
        all_batches = 0
        conf_mat = np.zeros((3,3))
        
        for batch_idx, (data, target, video)  in enumerate(loader):

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
           
            for i in range(np_target.shape[0]):
                conf_mat[int(np_target[i])][int(flat_pred[i])] += 1           

        myset = "testing"
        if loader == vp.trainloader_acc:
            myset = "training"

        mylogger.log('\n Accuracy on '+myset+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss/all_batches, correct, all_examples,
            100. * correct / all_examples))

        mylogger.log("-- Confusion Matrix --")
        conf_mat = np.multiply(conf_mat,1/all_examples)
        mylogger.log(conf_mat)

def main():

    # The basic core takes as input frames from the two camera viewpoints that look into the playing field
    # and learns proper kernels for recognizing the scene
    # This is a baseline to use as a pretrained classifier for recognizing entire playtypes in the advanced core

    global args, mylogger

    args = parser.parse_args()

    mylogger = MagnificentOracle()
    mylogger.set_log(logfile=args.log)
    mylogger.log("-dotted-line")  

    # TO-DO
    vp = viewpoint_classifier_ED(args.arch, path=args.dataset_path, viewpoints = args.viewpoints)
    train(int(args.epochs), vp) 

if __name__ == "__main__":

    main()


