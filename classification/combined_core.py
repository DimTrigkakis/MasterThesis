# BEST OF
'''
CUDA_VISIBLE_DEVICES=2 python advanced_core.py --arch dimo --training True --batch_size 4 --num_workers 32 --log ./logs/LSTM/LSTM_toy.txt --pretrained_model ./results/viewpoint_models/experiments/resnet_2_baseline_1e6_yes/model_15.pth --pretrained_finetuning True --lr_cnn 0.001 --lr_rnn 0.01 --lr_class 0.01 --weight_decay 0'''

'''
# Kill all user processes : kill `ps -o pid= -N T`
'''
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

from threading import Thread, Lock

from datasets import CustomDataset, VideoDataset, CustomDatasetViewpointRaw, CustomMasterDatasetPlays
from datasets import compute_mean_std
from datasets import extract_xmls
from models import AlexNet, VGG, ResNet, EncoderDecoderViewpoints, DimoAutoSequence, VGG_viewpoints

from extract_videos import extract_frames, deconstruct_video
from construct_video import construct_annotator

from viterbi import viterbi
from magnificentoracle import MagnificentOracle

import excitation_bp as eb


# No more "too many open files" problem
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')

parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/full_frame_data_2016/",help="Path that contains the dataset to be processed")
parser.add_argument("--dataset_index",metavar="DATASET_INDEX",default=0,help="Choose among customly made 90-10% training/testing splits for the datasets, default = 0")

parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=16, help="Use the batch size default: %d" % (2))
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=12, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=60, help="Number of epochs to train for, default = 24")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default="1e-6",help="Weight decay for the optimizer")

parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=8, help="How often should the model save, default = 10")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=2, help="How often to test the model, default = 4")

parser.add_argument("--pretrained_model",metavar="PRETRAINED_MODEL",default=None,help="Point to a pretrained viewpoint model")
parser.add_argument("--saved_model",metavar="SAVED_MODEL",default=None,help="Point to a pretrained playtype model")
parser.add_argument("--viewpoint_model",metavar="SAVED_MODEL",default=None,help="Viewpoint model segmentation")
parser.add_argument("--training",metavar="TRAINING",default=False,help="Training phase of the classifier is activated")
parser.add_argument("--categories",metavar="CATEGORIES",default="two",help="If categories are chunked, there's only 3 categories with PASS, RUSH, OTHER")

parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to fill with logger information")

parser.add_argument("--pretrained_same_architecture",metavar="pretrained_same_architecture", default=True, help="Whether the viewpoint model is the same as the one trained")

parser.add_argument("--arch",  "-a", metavar='ARCH', default='ED', choices=["alex", "vgg","resnet","ED"], help="model architecture: " + ' | '.join(["alex", "vgg","resnet"]))

mylogger = None

class playtype_classifier():
    def __init__(self, model="dimo" ,dataset_index=0):

        # First, create the weighted sampler by analyzing the dataset and ascribing proper class weights

        self.num_inner_nodes = 2
        self.length_of_sequence = 16       

        pretrained_model_selection = args.pretrained_model
        #pretrained_model_selection = "./results/viewpoint_models/vgg_viewpoint_experiments/v1/model_epoch_5.pth"
        self.model = DimoAutoSequence(pretrained_model=pretrained_model_selection,num_inner_nodes=self.num_inner_nodes,max_len=self.length_of_sequence)
        self.model = nn.DataParallel(self.model,device_ids=[0,1,2,3]).cuda()
        
        self.model.load_state_dict(torch.load(args.saved_model))

        mylogger.log("Playtype classifier ready")
        print self.model


class viewpoint_classifier():

    def __init__(self, model,dataset_index=0, path = None,viewpoints=3):

        
        if (model == "alex"):
            self.model = AlexNet()
        elif (model == "vgg"):
            self.model = VGG(num_classes=2)
        elif (model == "resnet"):
            self.model = ResNet()
        elif (model == "ED"):
            self.model_ED = EncoderDecoderViewpoints()

        self.model_vgg = VGG_viewpoints(num_classes=3).cuda()
        self.model_ed = EncoderDecoderViewpoints().cuda()

        self.model_vgg= nn.DataParallel(self.model_vgg,device_ids=[0,1,2,3]).cuda()
        self.model_vgg.load_state_dict(torch.load("./results/viewpoint_models/vgg_viewpoint_ED_prepared/model_epoch_2.pth"))
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

def acc_test(epoch, pt, vp, gl, video="2"):
    
    pt.model.eval()
    vp.model_vgg.eval()
    vp.model_ed.eval()

    conf_mat = np.zeros((3,3))

    batch_interval_print = 1


    for epoch in range(1,2):  

        corrects = 0
        batches = 0
        feature_realm = []

        featurescape = []
        timer = time.time()
        v = False
        k = False
        if (v):
            for batch_idx, (data,target) in enumerate(gl.view_loader): # 1024 frames per batch, to be turned to 256 later
                target.squeeze_()
                target = Variable(target.cuda())
                data = Variable(data.cuda())

                output, features = vp.model_vgg(data)
                fd = features.data.cpu()
                for i in range(fd.size()[0]):
                    featurescape.append(fd[i].unsqueeze_(0))
            print "Features calculated"
            print time.time()-timer
            timer = time.time()

            change_en = 0
            change_gt = 0
            myencodings = []
            for batch_idx, (data,target) in enumerate(gl.view_loader_ed):

                target.squeeze_()
                target = Variable(target.cuda())
                myfeatures = featurescape[batch_idx*1024:(batch_idx+1)*1024]
                myfeatures = Variable(torch.cat(myfeatures,0)).cuda()
                encodings = vp.model_ed(myfeatures)
                batches += target.size()[0]
                true_encoding = encodings.data.cpu()[0:target.size()[0]].max(1)[1]
                myencodings.append(true_encoding)

                true_target = target.data.cpu()
                correct = true_encoding.eq(true_target).cpu().sum()
                corrects += correct
                '''
                for i in range(1,1024):
                    if true_encoding[i][0] != true_encoding[i-1][0]:
                        change_en += 1
                    if true_target[i] != true_target[i-1]:
                        change_gt += 1
                '''

                print correct, "/ "+str(target.size()[0])
            print corrects, " / "+str(batches)
            print time.time()-timer
            pickle.dump(myencodings,open("./encodings_"+str(video)+".pik","w"))
        elif k:
            myencodings = pickle.load(open("./encodings_"+str(video)+".pik","r"))
            # put encoding into temporary file with c_estimation instead of c
            old = -1
            j = 1
            with open("./annotations/full_viewpoint_annotations_2016/"+str(video)+"_c_estimation.txt","w") as f:
                for i in range(len(myencodings)):
                    s = myencodings[i].size()[0]
                    for j in range(s):
                        alpha = int(myencodings[i][j].numpy()[0])
                        if alpha != old and ((alpha == 1 and old == 0) or old == -1 or (alpha == 2 and old) == 1 or (alpha == 0 and old == 2)):
                            f.write(str(i*1024+j)+" "+str(alpha)+str("\n"))
                            print str(alpha)+" "+str(i*1024+j)
                            old = alpha
        else:
            correct = 0
            data_size = 0
            for batch_idx, (imgs, play_type) in enumerate(gl.test_acc_loader):

                batch_size = play_type.size()[0]
                play_type = Variable(play_type.cuda())
                imgs = Variable(imgs.cuda())

                output = pt.model(imgs)
                play_type = play_type.squeeze()

                prediction = output.data.cpu().max(1)[1].numpy()
                prediction = np.squeeze(prediction)

                play_type_target = play_type.data.cpu().numpy()
                correct += np.sum(prediction == play_type_target)
                data_size += batch_size

                mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction, play_type_target, np.sum(prediction == play_type_target),int(batch_size)))
            train_loss = 0
            mylogger.log('Training set, Average loss: {:.4f} Accuracy: {}/{} ({:.0f}%)'.format(train_loss/data_size, correct, data_size,100.0*correct/(data_size))) 


    '''
    for batch_idx, (imgs, play_type) in enumerate(gl.test_acc_loader):
        print batch_idx
        print "here"
        batch_size = play_type.size()[0]
        play_type = Variable(play_type.cuda())
        imgs = Variable(imgs.cuda())
        output = pt.model(imgs)
        play_type = play_type.squeeze()

        prediction = output.data.cpu().max(1)[1].numpy()
        prediction = np.squeeze(prediction)
        loss = pt.criterion(output, play_type)

        play_type_target = play_type.data.cpu().numpy()
        correct += np.sum(prediction == play_type_target)
        train_loss += loss.data.cpu().numpy()[0]

        data_size += batch_size
       
        for i in range(play_type_target.shape[0]):
            conf_mat[int(play_type_target[i])][int(prediction[i])] += 1   

        if random.random() > 0.95:
            mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction, play_type_target, np.sum(prediction == play_type_target),int(batch_size)))
    '''
    
    #mylogger.log('Testing set, Average loss: {:.4f} Accuracy: {}/{} ({:.0f}%)'.format(train_loss/data_size, correct, data_size,100.0*correct/(data_size))) 

    #conf_mat = np.multiply(conf_mat,1.0/data_size)
    #mylogger.log('Conf matrix')
    #mylogger.log(conf_mat)

class global_loader():
    def __init__(self,path):
        # Extract all play information (frame segments)
        # Extract all viewpoint information (viewpoint segments)
        # Run viewpoint classifier on raw frame segments
        # After extracting view segments run playtype classifier
        customset_test = CustomMasterDatasetPlays(path = path,subset_type="testing",dataset_index=0, categories="chunked")
        self.test_acc_loader = torch.utils.data.DataLoader(dataset=customset_test,batch_size=16,shuffle=False,num_workers=args.num_workers)
        customset_test_vp = CustomDatasetViewpointRaw(path = path,subset_type="testing",dataset_index=0, viewpoints=3)
        self.view_loader = torch.utils.data.DataLoader(dataset=customset_test_vp,batch_size=64,shuffle=False,num_workers=args.num_workers)
        customset_test_vp = CustomDatasetViewpointRaw(path = path,subset_type="testing",dataset_index=0, viewpoints=3)
        self.view_loader_ed = torch.utils.data.DataLoader(dataset=customset_test_vp,batch_size=1024,shuffle=False,num_workers=args.num_workers)
                
def main():
    global args, mylogger
     
    args = parser.parse_args() 
    mylogger = MagnificentOracle()
    mylogger.set_log(logfile=args.log)
    mylogger.log("-dotted-line")  


    customset_test = CustomMasterDatasetPlays(path = args.dataset_path,subset_type="testing",dataset_index=0, categories="chunked")
    test_acc_loader = torch.utils.data.DataLoader(dataset=customset_test,batch_size=16,shuffle=False,num_workers=args.num_workers)
    print len(customset_test), len(test_acc_loader)
    #gl = global_loader(args.dataset_path)
    #pt = playtype_classifier(dataset_index=int(args.dataset_index))
    #vp = viewpoint_classifier(model="ED",dataset_index=int(args.dataset_index))
    #acc_test(args.epochs,pt,vp,gl,video="2016091100")

main()
