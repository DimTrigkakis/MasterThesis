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

from datasets import CustomDataset, VideoDataset, CustomDatasetViewpointRaw, CustomDatasetViewpointFramewise
from datasets import compute_mean_std
from datasets import extract_xmls
from models import VGG_viewpoints, EncoderDecoderViewpoints

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

parser.add_argument("--arch",  "-a", metavar='ARCH', default='VGG', choices=["VGG","ED"], help="model architecture: " + ' | '.join(["VGG","ED"]))
parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/full_frame_data_2016/",help="Path that contains the dataset to be processed")
parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=1, help="cnn batch size")
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=12, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=10, help="Number of epochs to train for")
parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=1, help="How often should the model save")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=1, help="How often to test the model, default = 5")
parser.add_argument("--pretrained_model",metavar="PRETRAINED_MODEL",default=None,help="When extracting frames for labelling, points to the pretrained classifier file")
parser.add_argument("--pretrained_same_architecture",metavar="PRETRAINED_SAME_ARCHITECTURE",default=None,help="When we want to remove the last layer of an imagenet pretrained model, and replace it for our own tasks")
parser.add_argument("--compute_mstd",metavar="COMPUTE_MSTD",default=False,help="If set to True, will compute values for the mean and std of the dataset instead of training/testing a classifier")
parser.add_argument("--training",metavar="TRAINING",default=None,help="Training phase of the classifier is activated")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default=1e-6,help="Weight decay for the optimizer")
parser.add_argument("--viewpoints",metavar="VIEWPOINTS",default=3,type=int,help="How many viewpoints to train on")

parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to fill with logger information")

mylogger = None

# Utilities
MO = MagnificentOracle()

class viewpoint_classifier():

    def weighted_sampling(self,path=None,viewpoints=3):

        if viewpoints == 3:
            if not os.path.isfile("./experiments/viewpoint_framewise/sampling_weights_three_viewpoints.p"):
                customset_preprocess = CustomDatasetViewpointFramewise(path = args.dataset_path,subset_type="training", retrieve_images=False,viewpoints = viewpoints)
                self.processloader = torch.utils.data.DataLoader(dataset=customset_preprocess,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))

                sample_views = [] # when you start

                for batch_idx, (imgs, label) in enumerate(self.processloader):
                    sample_views.append(label.numpy()[0][0])

                class_presence = [0, 0, 0]

                for view in sample_views:
                    class_presence[view] += 1

                for i in range(len(class_presence)):
                    class_presence[i] /= len(sample_views)*1.0

                class_weights = [0 for i in range(len(sample_views))]
                for i in range(len(sample_views)):
                    class_weights[i] = 1.0/class_presence[sample_views[i]]

                class_weights = [i for i in class_weights]

                # Finished with sampler weighting
                sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights,len(self.processloader),replacement=True)
                pickle.dump(sampler,open("./experiments/viewpoint_framewise/sampling_weights_three_viewpoints.p","wb"))
            else:
                sampler = pickle.load(open("./experiments/viewpoint_framewise/sampling_weights_three_viewpoints.p","rb"))
            return sampler


    def __init__(self, model="VGG", path = None,viewpoints=3):

        self.sampler = self.weighted_sampling(path=path,viewpoints=viewpoints)

        customset_train = CustomDatasetViewpointFramewise(path = path,subset_type="training",viewpoints = viewpoints)
        customset_test = CustomDatasetViewpointFramewise(path = path,subset_type="testing",viewpoints = viewpoints)

        self.trainloader = torch.utils.data.DataLoader(dataset=customset_train,sampler=self.sampler,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
        self.trainloader_acc = torch.utils.data.DataLoader(dataset=customset_train,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        self.testloader_acc = torch.utils.data.DataLoader(dataset=customset_test,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

        #print len(self.trainloader) #434
        #print len(self.trainloader_acc) #434
        #print len(self.testloader_acc) #405, no scoreboard augmentation

        if (model == "VGG"):
            self.model = VGG_viewpoints(num_classes=viewpoints).cuda()
            self.model.soft = nn.LogSoftmax()
            if args.pretrained_same_architecture is not None:
                self.model.load_state_dict(torch.load(args.pretrained_same_architecture))
        elif (model == "ED"):
            self.model = EncoderDecoderViewpoints()

        
        if args.pretrained_model != None:
            if args.pretrained_same_architecture is not None:
                self.model.load_state_dict(torch.load(args.pretrained_model))
            else:
                if args.arch == "vgg":
                    self.model.soft = None
                    classifier = list(self.model.classifier.children())
                    classifier.pop()
                    classifier.append(torch.nn.Linear(4096,1000))
                    new_classifier = torch.nn.Sequential(*classifier)
                    self.model.classifier = new_classifier
                    self.model.load_state_dict(torch.load(args.pretrained_model))
                    classifier = list(self.model.classifier.children())
                    classifier.pop()
                    classifier.append(torch.nn.Linear(4096,2))
                    new_classifier = torch.nn.Sequential(*classifier)
                    self.model.classifier = new_classifier
                    self.model.soft = nn.LogSoftmax()
                else:
                    self.model.fc = nn.Linear(512, 1000)
                    self.model.load_state_dict(torch.load(args.pretrained_model))
                    self.model.fc = nn.Linear(512,2)     
        
   
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=float(args.weight_decay), lr=0.0001)
        self.criterion = nn.NLLLoss().cuda()
        mylogger.log(self.model)

# Train NN
def train(epochs, vp):
    
    if not args.training:
        args.test_interval = 1
        args.save_interval = 10000
    batch_interval_print = 1

    #vp.model = nn.DataParallel(vp.model,device_ids=[0]).cuda()

    for epoch in range(1, epochs + 1):  
        mylogger.log(" Epoch :"+str(epoch))
        vp.model.train()

        
        if args.training:
            for batch_idx, (data,target) in enumerate(vp.trainloader):

                if data.size()[0] != args.batch_size: # Incomplete batches require more memory so we ignore one batch
                    continue

                target.squeeze_()
                target = Variable(target.cuda())

                #img = MO.visual_tensor(data)

                data = Variable(data.cuda())
                output = vp.model(data)
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
            torch.save(vp.model.state_dict(), os.path.join(directory, "model_epoch_" + str(epoch)+".pth"))     

def acc(epoch, vp):

    for loader in [ vp.trainloader_acc, vp.testloader_acc]:

        vp.model.eval()

        loss = 0
        correct = 0
        all_examples = 0.0
        all_batches = 0
        conf_mat = np.zeros((3,3))
        
        for batch_idx, (data, target)  in enumerate(loader):

            if data.size()[0] != args.batch_size: # Incomplete batches require more memory so we ignore one batch. Since we do this before all_examples += data.size()[0], we can safely measure the accuracy
                continue

            all_batches += 1

            target.squeeze_()
            target = Variable(target.cuda())
            data = Variable(data.cuda())
            output = vp.model(data)
            loss += vp.criterion(output, target).data[0]
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            all_examples += data.size()[0]
            flat_pred = pred.cpu().numpy().flatten()
            np_target = target.data.cpu().numpy()
           
            for i in range(np_target.shape[0]):
                conf_mat[int(np_target[i])][int(flat_pred[i])] += 1  

            print(correct, all_examples)         

        myset = "testing"
        if loader == vp.trainloader_acc:
            myset = "training"

        mylogger.log('\n Accuracy on '+myset+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss/all_batches, correct, all_examples,
            100. * correct / all_examples))

        mylogger.log("-- Confusion Matrix --")
        conf_mat = np.multiply(conf_mat,1/all_examples)
        mylogger.log(conf_mat)

def compute_mstd(args):
    customset_viewpoint = CustomDatasetViewpointFramewise(path = args.dataset_path,subset_type="mstd")
    mstd_loader = torch.utils.data.DataLoader(dataset=customset_viewpoint,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    customset_viewpoint.compute_mean_std(mstd_loader)

def main():

    # The basic core takes as input frames from the two camera viewpoints that look into the playing field
    # and learns proper kernels for recognizing the scene
    # This is a baseline to use as a pretrained classifier for recognizing entire playtypes in the advanced core

    global args, mylogger

    args = parser.parse_args()
        
    if args.compute_mstd: 
        compute_mstd(args)
        return

    mylogger = MagnificentOracle()
    mylogger.set_log(logfile=args.log)
    mylogger.log("-dotted-line")  

    vp = viewpoint_classifier(args.arch, path=args.dataset_path, viewpoints = args.viewpoints)  
    train(int(args.epochs), vp) 

if __name__ == "__main__":

    main()


