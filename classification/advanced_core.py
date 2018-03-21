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

from datasets import CustomDataset, VideoDataset, CustomDatasetPlays
from datasets import compute_mean_std
from datasets import extract_xmls
from models import AlexNet, VGG, ResNet, DimoLSTM, DimoResNet

from extract_videos import extract_frames, deconstruct_video
from construct_video import construct_annotator

from viterbi import viterbi
from magnificentlogger import MagnificentLogger
# No more "too many open files" problem
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')

parser.add_argument("--arch",  "-a", metavar='ARCH', default='resnet', choices=["alex", "vgg","resnet","dimo"], help="model architecture: " + ' | '.join(["alex", "vgg","resnet"]))
parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/full_frame_data_2016/",help="Path that contains the dataset to be processed")
parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=64, help="Use the batch size default: %d" % (64))
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=64, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=32, help="Number of epochs to train for, default = 32")
parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=10, help="How often should the model save, default = 10")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=4, help="How often to test the model, default = 4")
parser.add_argument("--dataset_index",metavar="DATASET_INDEX",default=0,help="Choose among customly made 80-20% training/testing splits for the datasets, default = 0")
parser.add_argument("--pretrained_model",metavar="PRETRAINED_MODEL",default=None,help="When extracting frames for labelling, points to the pretrained classifier file")
parser.add_argument("--pretrained_finetuning",metavar="PRETRAINED_FINETUNING",default=False,help="When we want to remove the last layer of an imagenet pretrained model, and replace it for our own tasks")
parser.add_argument("--training",metavar="TRAINING",default=False,help="Training phase of the classifier is activated")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default="1e-6",help="Weight decay for the optimizer")
parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to fill with logger information")
parser.add_argument("--lr_rnn",metavar="LR_RNN", default="1e-4", help="learning rate rnn")
parser.add_argument("--lr_cnn",metavar="LR_CNN", default="1e-4", help="learning rate cnn")
parser.add_argument("--lr_class",metavar="LR_CLASS", default="1e-4", help="learning rate classifier")
parser.add_argument("--categories",metavar="CATEGORIES",default="chunked",help="If categories are chunked, there's only 3 categories with PASS, RUSH, OTHER")
parser.add_argument("--dist",metavar="DIST",default=0,type=int,help="Distribution for the lstm time window subsampling")
parser.add_argument("--optim",metavar="OPTIM",default="Adam",help="Optimizer to use")
parser.add_argument("--pretrained_lstm",metavar="PRETRAINED_LSTM",default=None,help="Instead of importing a resnet for the cnn, import a resulting model for the entire model")

mylogger = None

class viewpoint_classifier():
    def __init__(self, model,dataset_index=0,video_target = None):

        # First, create the weighted sampler by analyzing the dataset and ascribing proper class weights

        customset_preprocess = CustomDatasetPlays(path = args.dataset_path,subset_type="training",dataset_index=dataset_index,categories=args.categories, retrieve_images=False)
        self.processloader = torch.utils.data.DataLoader(dataset=customset_preprocess,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))

        sample_plays = [] # when you start

        for batch_idx, (imgs, play_type) in enumerate(self.processloader):
            sample_plays.append(play_type.cpu().numpy()[0][0])

        cat = 3
        if args.categories == "unchunked":
            cat = 13
        if args.categories == "two":
            cat = 2
        self.cat = cat
        class_presence = [0 for i in range(self.cat)]

        for play in sample_plays:
            class_presence[play] += 1

        for i in range(cat):
            class_presence[i] /= len(sample_plays)*1.0

        customset_train = CustomDatasetPlays(path = args.dataset_path,subset_type="training",dataset_index=dataset_index,categories=args.categories,dist=args.dist)
        self.trainset = customset_train
        mylogger.log("Loaded {} dataset with {} number of plays".format(customset_train.subset_type, customset_train.maxlength))

        customset_test = CustomDatasetPlays(path = args.dataset_path,subset_type="testing",dataset_index=dataset_index,categories=args.categories,dist=args.dist)
        self.testset = customset_test
        mylogger.log("Loaded {} dataset with {} number of plays".format(customset_test.subset_type, customset_test.maxlength))

        class_weights = [0 for i in range(len(sample_plays))]
        for i in range(len(sample_plays)):
            class_weights[i] = 1.0/class_presence[sample_plays[i]]
        m = sum(class_weights)
        class_weights = [i*1.0/m for i in class_weights]

        # Finished with sampler weighting
        sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights,len(self.processloader),replacement=True)
        self.trainloader = torch.utils.data.DataLoader(dataset=customset_train,sampler=sampler,batch_size=int(args.batch_size),shuffle=True,num_workers=int(args.num_workers))
        self.train_acc_loader = torch.utils.data.DataLoader(dataset=customset_train,sampler=None,batch_size=int(args.batch_size),shuffle=False,num_workers=int(args.num_workers))
        self.testloader = torch.utils.data.DataLoader(dataset=customset_test,batch_size=int(args.batch_size),shuffle=False,num_workers=int(args.num_workers))  
   
        if (model == "alex"):
            self.cnn_model = AlexNet()
        elif (model == "vgg"):
            self.cnn_model = VGG()
        elif (model == "dimo"):
            self.cnn_model = DimoResNet(num_classes = cat)

        self.input_feature_dimension = 512
        self.n_hidden = 512
        self.n_hidden_classifier = 32
        self.n_categories = cat
        self.lstm_sequences = 1
        self.bidirectional = 2
        self.lstm_model = DimoLSTM(input_size=self.input_feature_dimension, hidden_size=self.n_hidden, num_layers=self.lstm_sequences, bidirectional=(False if self.bidirectional == 1 else True))
        self.classifier = nn.Sequential(nn.Linear(self.n_hidden*2,self.n_hidden_classifier), nn.Linear(self.n_hidden_classifier,self.n_categories), nn.LogSoftmax())   

        if args.pretrained_model != None:
            if args.pretrained_lstm == None:
                if args.pretrained_finetuning == False: # Pretrained imagenet
                    self.cnn_model.fc = nn.Linear(512,1000) 
                    self.cnn_model.load_state_dict(torch.load(args.pretrained_model))
                    self.cnn_model.fc = None
                else:
                    if args.arch != "vgg":
                        self.cnn_model.fc = nn.Linear(512,3) # Pretrained viewpoint
                        self.cnn_model.load_state_dict(torch.load(args.pretrained_model))
                        self.cnn_model.fc = None
                    else:
                        classifier = list(self.cnn_model.classifier.children())
                        classifier.pop()
                        classifier.append(torch.nn.Linear(4096,2))
                        new_classifier = torch.nn.Sequential(*classifier)
                        self.cnn_model.classifier = new_classifier
                        self.cnn_model.load_state_dict(torch.load(args.pretrained_model))
                        self.cnn_model.soft = nn.LogSoftmax()
                        print self.model
            else: # lstm
                self.cnn_model.fc = None
                self.cnn_model.load_state_dict(torch.load(args.pretrained_lstm+"/cnn_model_30.pth")) 
                self.lstm_model.load_state_dict(torch.load(args.pretrained_lstm+"/lstm_model_30.pth"))
                self.classifier.load_state_dict(torch.load(args.pretrained_lstm+"/classifier_30.pth"))

        else: # tabula rasa
            self.cnn_model.fc = None

        self.cnn_model.cuda()    
        self.lstm_model.cuda()
        self.classifier.cuda()

        mylogger.log("-dotted-line")   
        mylogger.log("Using architecture: {}".format(args.arch))
        mylogger.log("Using batch size: {}".format(args.batch_size))
        mylogger.log("Using worker count: {}".format(args.num_workers))
        mylogger.log("Using epoch count: {}".format(args.epochs))
        mylogger.log("Using dataset index: {}".format(args.dataset_index)) 
        mylogger.log("Using model  <{}>".format(args.pretrained_model))
        if args.pretrained_finetuning:
            mylogger.log("Using finetuning")
        mylogger.log("Using weight decay: {}".format(args.weight_decay))
        if args.training:
            mylogger.log("Training schedule type <training>")
        mylogger.log("Using categories  <{}>".format(args.categories))

        mylogger.log("Using learning rates  <{},{},{}>".format(args.lr_cnn,args.lr_rnn,args.lr_class))
        mylogger.log("Using subsampling window type {}".format(args.dist))
        mylogger.log("Using optimizer {}".format(args.optim))
        mylogger.log("Using hyperparameters lstm hidden: {}, classifier hidden: {}, lstm sequences: {}, lstm bidirectional: {}".format(self.n_hidden,self.n_hidden_classifier,self.lstm_sequences,self.bidirectional))


        # Choose parameters to train
        parameters = [
                {'params': self.lstm_model.parameters(), 'lr':float(args.lr_rnn)},
                {'params': self.cnn_model.parameters(), 'lr': float(args.lr_cnn)},
                {'params': self.classifier.parameters(), 'lr': float(args.lr_class)}]
                
        if args.optim == "SGD":
            self.optimizer = optim.SGD(parameters, weight_decay=float(args.weight_decay), momentum=0.5,nesterov=True)
        elif args.optim == "Adam":
            self.optimizer = optim.Adam(parameters, weight_decay=float(args.weight_decay))
        self.criterion = nn.CrossEntropyLoss().cuda()

        mylogger.log("Viewpoint classifier ready")

# Train NN

def train(epochs, vp):
    
    mylogger.log("-dotted-line")
    mylogger.log("Initializing training")

    criterion = nn.NLLLoss()
    loss_batch_skip = 1

    if not args.training:
        args.test_interval = 1
        args.save_interval = 2
        epochs = 1

    for epoch in range(1, epochs + 1): 
        mylogger.log("Beginning of epoch <{}>".format(epoch))  

        vp.cnn_model.train()
        vp.lstm_model.train()
        vp.classifier.train()
    

        if args.training:    
            timer = time.time()
            for batch_idx, (imgs, play_type) in enumerate(vp.trainloader):

                # init
                play_type = Variable(play_type.cuda())
                play_type = play_type.squeeze(1)

                imgs = Variable(imgs.cuda())
                shape = imgs.size()
                imgs = imgs.view(-1,shape[2],shape[3],shape[4])
                batch_size = shape[0]

                #  cnn
                myoutput = vp.cnn_model(imgs)

                # lstm
                lstm_chunks = []
                for i in range(int(batch_size)):
                    indices = Variable(torch.LongTensor(range(i*vp.trainset.frame_select,(i+1)*(vp.trainset.frame_select))).cuda())
                    lstm_chunk = torch.index_select(myoutput, 0, indices).data
                    lstm_chunk = lstm_chunk.unsqueeze_(1)
                    lstm_chunks.append(lstm_chunk)

                lstm_batch = Variable(torch.cat(lstm_chunks,1))
                hidden_rnn = vp.lstm_model.init_hidden(int(batch_size),vp.lstm_sequences, vp.bidirectional)
                output_final, hidden_final = vp.lstm_model(lstm_batch, hidden_rnn)
                output_final = output_final.select(0,vp.trainset.frame_select-1)

                # classifier
                output_final = vp.classifier(output_final)
                
                loss = criterion(output_final, play_type)

                vp.optimizer.zero_grad()
                loss.backward()
                vp.optimizer.step()

                if loss.data.cpu().numpy() != loss.data.cpu().numpy():
                    print "BAD LOSS, NAN"

                if batch_idx % loss_batch_skip == 0:
                    mylogger.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx , len(vp.trainloader),
                        100. * batch_idx / len(vp.trainloader), loss.data.cpu().numpy()[0]/int(args.batch_size)))

        #mylogger.log("Time void from epoch: {}".format(time.time()-timer))

        if (args.save_dir is not None and epoch % int(args.save_interval) == 0):
            directory = args.save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
             
            mylogger.log("--saving model, epoch : "+str(epoch)+"--")
            torch.save(vp.cnn_model.state_dict(), os.path.join(directory, "cnn_model_" + str(epoch)+".pth"))
            torch.save(vp.lstm_model.state_dict(), os.path.join(directory, "lstm_model_" + str(epoch)+".pth"))
            torch.save(vp.classifier.state_dict(), os.path.join(directory, "classifier_" + str(epoch)+".pth"))

        if (epoch % int(args.test_interval) == 0):
            if args.training:
                mylogger.log("--training accuracy--")
                acc_train(epoch, vp)
            if len(vp.testloader.dataset) > 0:
                mylogger.log("--testing accuracy--")
                acc_test(epoch,vp)        

def acc_train(epoch, vp):
    
    vp.cnn_model.eval()
    vp.lstm_model.eval()
    vp.classifier.eval()

    train_loss = 0
    correct = 0
    data_size = 0.0

    criterion = nn.NLLLoss()

    for batch_idx, (imgs, play_type) in enumerate(vp.train_acc_loader):

        # init
        play_type = Variable(play_type.cuda())
        play_type = play_type.squeeze(1)

        imgs = Variable(imgs.cuda())
        shape = imgs.size()
        imgs = imgs.view(-1,shape[2],shape[3],shape[4])
        batch_size = shape[0]

        #  cnn
        myoutput = vp.cnn_model(imgs)

        # lstm
        lstm_chunks = []
        for i in range(int(batch_size)):
            indices = Variable(torch.LongTensor(range(i*vp.trainset.frame_select,(i+1)*(vp.trainset.frame_select))).cuda())
            lstm_chunk = torch.index_select(myoutput, 0, indices).data
            lstm_chunk = lstm_chunk.unsqueeze_(1)
            lstm_chunks.append(lstm_chunk)

        lstm_batch = Variable(torch.cat(lstm_chunks,1))
        hidden_rnn = vp.lstm_model.init_hidden(int(batch_size),vp.lstm_sequences,vp.bidirectional)
        output_final, hidden_final = vp.lstm_model(lstm_batch, hidden_rnn)
        output_final = output_final.select(0,vp.trainset.frame_select-1)

        # classifier
        output_final = vp.classifier(output_final)

        prediction = output_final.data.cpu().max(1)[1].numpy()
        prediction = np.squeeze(prediction)
        play_type_target = play_type.data.cpu().numpy()

        loss = criterion(output_final, play_type)

        correct += np.sum(prediction == play_type_target)
        train_loss += loss.data.cpu().numpy()[0]

        data_size += batch_size

        if random.random() > 0.95:
            mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction, play_type_target, np.sum(prediction == play_type_target),int(batch_size)))
    
    mylogger.log('Training set, Average loss: {:.4f} Accuracy: {}/{} ({:.0f}%)'.format(train_loss/data_size, correct, data_size,100.0*correct/(data_size))) 

def acc_test(epoch, vp):

    vp.cnn_model.eval()
    vp.lstm_model.eval()
    vp.classifier.eval()

    test_loss = 0
    correct = 0
    data_size = 0.0

    criterion = nn.NLLLoss()
      
    conf_mat = np.zeros((vp.cat,vp.cat))

    window_i = 11 # 0
    window_size = 1 # 12

    for batch_idx, (imgs, play_type) in enumerate(vp.testloader):
        # init
        play_type = Variable(play_type.cuda())
        play_type = play_type.squeeze(1)

        # I-J window processing
        # remove frames outside I-J range
        I = window_i
        J = window_size+window_i
        window_index = torch.LongTensor(range(I,J))
        imgs = torch.index_select(imgs,1,window_index)

        imgs = Variable(imgs.cuda())
        shape = imgs.size()
        imgs = imgs.view(-1,shape[2],shape[3],shape[4])

        batch_size = shape[0]
        true_frame_select = shape[1]

        #  cnn
        myoutput = vp.cnn_model(imgs)

        # lstm
        lstm_chunks = []
        for i in range(int(batch_size)):
            indices = Variable(torch.LongTensor(range(i*true_frame_select,(i+1)*(true_frame_select))).cuda())
            lstm_chunk = torch.index_select(myoutput, 0, indices).data
            lstm_chunk = lstm_chunk.unsqueeze_(1)
            lstm_chunks.append(lstm_chunk)

        lstm_batch = Variable(torch.cat(lstm_chunks,1))
        hidden_rnn = vp.lstm_model.init_hidden(int(batch_size),vp.lstm_sequences,vp.bidirectional)
        output_final, hidden_final = vp.lstm_model(lstm_batch, hidden_rnn)
        output_final = output_final.select(0,true_frame_select-1)

        # classifier
        output_final = vp.classifier(output_final)

        prediction = output_final.data.cpu().max(1)[1].numpy()
        prediction = np.squeeze(prediction)
        play_type_target = play_type.data.cpu().numpy()

        loss = criterion(output_final, play_type)

        correct += np.sum(prediction == play_type_target)
        test_loss += loss.data.cpu().numpy()[0]

        data_size += batch_size

        for i in range(batch_size):
            prediction = np.asarray(prediction)*np.ones(1)
            conf_mat[int(play_type_target[i])][int(prediction[i])] += 1 

    mylogger.log('Testing set, Average loss: {:.4f} Accuracy: {}/{} ({:.0f}%)'.format(test_loss/data_size, correct, data_size,100.0*correct/(data_size))) 

    conf_mat = np.multiply(conf_mat,1.0/data_size)
    mylogger.log(conf_mat)
        
def main():
    global args, mylogger
     
    args = parser.parse_args() 
    mylogger = MagnificentLogger()#logtext=args.log)
    mylogger.log("-dotted-line")  
    vp = viewpoint_classifier(args.arch,dataset_index=int(args.dataset_index))

    train(int(args.epochs), vp)

if __name__ == "__main__":

    main()


