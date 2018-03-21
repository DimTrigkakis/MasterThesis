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

from datasets import CustomDataset, VideoDataset, CustomDatasetViewpoint
from datasets import compute_mean_std
from datasets import extract_xmls
from models import AlexNet, VGG_viewpoints, ResNet, EncoderDecoderViewpoints

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

parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/baseline_training_data/",help="Path that contains the dataset to be processed")
parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=256, help="cnn batch size")
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=12, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=20, help="Number of epochs to train for")
parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=4, help="How often should the model save")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=1, help="How often to test the model, default = 5")
parser.add_argument("--dataset_index",metavar="DATASET_INDEX",type=int,default=0,choices=[0],help="Choose among customly made 80-20% training/testing splits for the datasets, default = 0")
parser.add_argument("--compute_mstd",metavar="COMPUTE_MSTD",default=False,help="If set to True, will compute values for the mean and std of the dataset instead of training/testing a classifier")
parser.add_argument("--training",metavar="TRAINING",default=None,help="Training phase of the classifier is activated")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default=1e-6,help="Weight decay for the optimizer")

# Utilities
MO = MagnificentOracle()

class viewpoint_classifier():

    def weighted_sampling(self,dataset_index=0,path=None):

        if not os.path.isfile("./results/intermediate_data/sampling_weights_three_viewpoints.p"):
            customset_preprocess = CustomDatasetViewpoint(path = args.dataset_path,subset_type="training",dataset_index=dataset_index, retrieve_images=False,viewpoints = 3)
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
            m = 2*len(sample_views)
            class_weights = [i/m for i in class_weights]

            # Finished with sampler weighting
            sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights,len(self.processloader),replacement=True)
            pickle.dump(sampler,open("./results/intermediate_data/sampling_weights_three_viewpoints.p","wb"))
        else:
            sampler = pickle.load(open("./results/intermediate_data/sampling_weights_three_viewpoints.p","rb"))
        return sampler


    def __init__(self, dataset_index=0, path = None):

        self.sampler = self.weighted_sampling(dataset_index=dataset_index,path=path)

        customset_train = CustomDatasetViewpoint(path = path,subset_type="training",dataset_index=dataset_index,viewpoints = 3)
        customset_test = CustomDatasetViewpoint(path = path,subset_type="testing",dataset_index=dataset_index,viewpoints = 3)

        self.trainloader = torch.utils.data.DataLoader(pin_memory=True,dataset=customset_train,sampler=self.sampler,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
        self.trainloader_acc = torch.utils.data.DataLoader(dataset=customset_train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
        self.testloader_acc = torch.utils.data.DataLoader(dataset=customset_test,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)

        self.trainloader_seg = torch.utils.data.DataLoader(pin_memory=True,dataset=customset_train,batch_size=256,shuffle=False,num_workers=args.num_workers)
        self.trainloader_seg_encoder = torch.utils.data.DataLoader(pin_memory=True,dataset=customset_train,batch_size=1024,shuffle=False,num_workers=args.num_workers)
        self.trainloader_acc_seg = torch.utils.data.DataLoader(dataset=customset_train,batch_size=96,shuffle=False,num_workers=args.num_workers)
        self.trainloader_acc_seg_encoder = torch.utils.data.DataLoader(dataset=customset_train,batch_size=96*32,shuffle=False,num_workers=args.num_workers)
        self.testloader_acc_seg = torch.utils.data.DataLoader(dataset=customset_test,batch_size=256,shuffle=False,num_workers=args.num_workers)
        self.testloader_acc_seg_encoder = torch.utils.data.DataLoader(dataset=customset_test,batch_size=1024,shuffle=False,num_workers=args.num_workers)

        self.model_vgg = VGG_viewpoints(num_classes=3).cuda()
        self.model_ed = EncoderDecoderViewpoints().cuda()

        # Trained vgg loading, comment to disable
        self.model_vgg= nn.DataParallel(self.model_vgg,device_ids=[0,1,2,3]).cuda()
        self.model_vgg.load_state_dict(torch.load("./results/viewpoint_models/vgg_viewpoint_ED_prepared/model_epoch_2.pth"))
        #self.model_vgg = self.model_vgg.module
        #self.model_vgg.cuda()
        mod = list(self.model_vgg.module.classifier.children())
        mod.pop()
        mod.pop()
        mod.pop()
        new_classifier = torch.nn.Sequential(*mod)
        self.model_vgg.module.new_classifier = new_classifier
        print self.model_vgg
        # Trained ED loading, comment to disable

        self.model_ed.load_state_dict(torch.load("./results/viewpoint_models/vgg_viewpoint_ED_disjointed/model_ed_epoch_20.pth"))
        print self.model_ed

        '''
        if args.pretrained_model != None:
            if args.pretrained_same_architecture:
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
        '''     
   
        self.optimizer_vgg = optim.Adam(self.model_vgg.parameters(), weight_decay=float(args.weight_decay), lr=0.0001)
        self.optimizer_ED = optim.Adam(self.model_ed.parameters(), weight_decay=float(args.weight_decay), lr=0.0001)

# Train NN

def encoder_viewpoint(epochs, vp):
    if not args.training:
        args.test_interval = 1
        args.save_interval = 10000
    batch_interval_print = 1

    vp.criterion = nn.NLLLoss().cuda()

    featurescape = []
    for epoch in range(1, epochs + 1):  
        MO.log(" Epoch :"+str(epoch))
        vp.model_ed.train()

        if args.training:

            if epoch == 1:
                vp.model_vgg.train()
                for batch_idx, (data,target) in enumerate(vp.trainloader_seg): # 256 frames per batch, to be turned to 256 later
                    target.squeeze_()
                    target = Variable(target.cuda())
                    data = Variable(data.cuda())

                    output, features = vp.model_vgg(data)
                    fd = features.data.cpu()
                    print fd.size()[0]
                    for i in range(fd.size()[0]):
                        featurescape.append(fd[i].unsqueeze_(0))
                print "Features calculated"

            for batch_idx, (data,target) in enumerate(vp.trainloader_seg_encoder):

                if batch_idx == 39:
                    break
                target.squeeze_()
                target = Variable(target.cuda())
                myfeatures = featurescape[batch_idx*1024:(batch_idx+1)*1024]
                myfeatures = Variable(torch.cat(myfeatures,0)).cuda()
                encodings = vp.model_ed(myfeatures)
                correct = encodings.data.cpu().max(1)[1].eq(target.data.cpu()).cpu().sum()
                loss = vp.criterion(encodings, target)
                vp.optimizer_ED.zero_grad()
                loss.backward()
                vp.optimizer_ED.step()
                print correct, "/ 1024"
                
                pred = output.data.cpu().max(1)[1] 

                if batch_idx % batch_interval_print == batch_interval_print-1:
                    MO.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, len(vp.trainloader_seg_encoder),
                        100. * batch_idx / len(vp.trainloader_seg_encoder), loss.data[0]))


            if (args.save_dir is not None and epoch % int(args.save_interval) == 0):
                directory = args.save_dir
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(vp.model_ed.state_dict(), os.path.join(directory, "model_ed_epoch_" + str(epoch)+".pth"))       

def encoder_viewpoint_testing(vp):
    batch_interval_print = 1

    vp.criterion = nn.NLLLoss().cuda()

    for epoch in range(1,2):  
        MO.log(" Epoch :"+str(epoch))
        vp.model_vgg.eval()
        vp.model_ed.eval()

        corrects = 0
        batches = 0
        feature_realm = []

        featurescape = []
        for batch_idx, (data,target) in enumerate(vp.testloader_acc_seg): # 256 frames per batch, to be turned to 256 later
            target.squeeze_()
            target = Variable(target.cuda())
            data = Variable(data.cuda())

            output, features = vp.model_vgg(data)
            fd = features.data.cpu()
            for i in range(fd.size()[0]):
                featurescape.append(fd[i].unsqueeze_(0))
        print "Features calculated"

        change_en = 0
        change_gt = 0
        for batch_idx, (data,target) in enumerate(vp.testloader_acc_seg_encoder):

            if batch_idx == 5:
                    break
            target.squeeze_()
            target = Variable(target.cuda())
            myfeatures = featurescape[batch_idx*1024:(batch_idx+1)*1024]
            myfeatures = Variable(torch.cat(myfeatures,0)).cuda()
            encodings = vp.model_ed(myfeatures)
            batches += target.size()[0]
            true_encoding = encodings.data.cpu()[0:target.size()[0]].max(1)[1]
            true_target = target.data.cpu()
            correct = true_encoding.eq(true_target).cpu().sum()
            corrects += correct
            for i in range(1,1024):
                if true_encoding[i][0] != true_encoding[i-1][0]:
                    change_en += 1
                if true_target[i] != true_target[i-1]:
                    change_gt += 1

            print correct, "/ "+str(target.size()[0])
        print corrects, " / "+str(batches)

def train_viewpoint(epochs, vp):
    
    if not args.training:
        args.test_interval = 1
        args.save_interval = 10000
    batch_interval_print = 1

    vp.criterion = nn.NLLLoss().cuda()
    vp.model_vgg= nn.DataParallel(vp.model_vgg,device_ids=[0,1,2,3]).cuda()

    for epoch in range(1, epochs + 1):  
        MO.log(" Epoch :"+str(epoch))
        vp.model_vgg.train()

        if args.training:
            for batch_idx, (data,target) in enumerate(vp.trainloader):

                target.squeeze_()
                target = Variable(target.cuda())
                data = Variable(data.cuda())

                output = vp.model_vgg(data)
                
                loss = vp.criterion(output, target)
                
                vp.optimizer_vgg.zero_grad()
                loss.backward()
                vp.optimizer_vgg.step()

                pred = output.data.cpu().max(1)[1] 

                if batch_idx % batch_interval_print == batch_interval_print-1:
                    MO.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, len(vp.trainloader),
                        100. * batch_idx / len(vp.trainloader), loss.data[0]))
        
        if (epoch % int(args.test_interval) == 0):
            if len(vp.testloader_acc) > 0:
                acc(epoch,vp)

        if (args.save_dir is not None and epoch % int(args.save_interval) == 0):
            directory = args.save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(vp.model_vgg.state_dict(), os.path.join(directory, "model_epoch_" + str(epoch)+".pth"))     

def acc(epoch, vp):

    for loader in [ vp.trainloader_acc, vp.testloader_acc]:

        vp.model_vgg.eval()

        loss = 0
        correct = 0
        all_examples = 0.0
        all_batches = 0
        conf_mat = np.zeros((3,3))
        
        for batch_idx, (data, target)  in enumerate(loader):
            all_batches += 1

            target.squeeze_()
            target = Variable(target.cuda())
            data = Variable(data.cuda())
            output = vp.model_vgg(data)
            loss += vp.criterion(output, target).data[0]
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            all_examples += data.data.cpu().numpy().shape[0]
            flat_pred = pred.cpu().numpy().flatten()
            np_target = target.data.cpu().numpy()
           
            for i in range(np_target.shape[0]):
                conf_mat[int(np_target[i])][int(flat_pred[i])] += 1           

        MO.log('\n Accuracy on set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss/all_batches, correct, all_examples,
            100. * correct / all_examples))

        MO.log("-- Confusion Matrix --")
        conf_mat = np.multiply(conf_mat,1/all_examples)
        MO.log(conf_mat)

def compute_mstd(args):
    customset_viewpoint = CustomDatasetViewpoint(path = args.dataset_path,subset_type="mstd",dataset_index=args.dataset_index)
    mstd_loader = torch.utils.data.DataLoader(dataset=customset_viewpoint,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    customset_viewpoint.compute_mean_std(mstd_loader)

def main():

    # The basic core takes as input frames from the two camera viewpoints that look into the playing field
    # and learns proper kernels for recognizing the scene
    # This is a baseline to use as a pretrained classifier for recognizing entire playtypes in the advanced core

    global args

    args = parser.parse_args()
        
    if args.compute_mstd: 
        compute_mstd(args)
        return

    vp = viewpoint_classifier(dataset_index=int(args.dataset_index), path=args.dataset_path)    
    #train_viewpoint(int(args.epochs), vp) 
    #encoder_viewpoint(int(args.epochs),vp)
    encoder_viewpoint_testing(vp)

if __name__ == "__main__":

    main()


