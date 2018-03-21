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

from datasets import CustomDataset, VideoDataset
from datasets import compute_mean_std
from datasets import extract_xmls
from models import AlexNet, VGG, ResNet

from extract_videos import extract_frames, deconstruct_video
from construct_video import construct_annotator

from viterbi import viterbi


parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')

parser.add_argument("--arch",  "-a", metavar='ARCH', default='resnet', choices=["alex", "vgg","resnet"], help="model architecture: " + ' | '.join(["alex", "vgg","resnet"]))
parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./training_data/",help="Path that contains the dataset to be processed")
parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=64, help="Juse the batch size default: %d" % (64))
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=64, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=50, help="Number of epochs to train for, default = 50")
parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=5, help="How often should the model save")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=5, help="How often to test the model, default = 5")
parser.add_argument("--dataset_index",metavar="DATASET_INDEX",default=0,choices=["25","50","75","0","1","2","3","4","5","6","7","8","9","10","11"],help="Choose among customly made 80-20% training/testing splits for the datasets, default = 0")

parser.add_argument("--label_dataset",type=bool,metavar="LABEL_DATASET",default=False,help="If this is set to True, we produce labels every <frame_skip> frames and save them in the dataset folder")
parser.add_argument("--pretrained_model",metavar="PRETRAINED_MODEL",default=None,help="When extracting frames for labelling, points to the pretrained classifier file")
parser.add_argument("--pretrained_finetuning",metavar="PRETRAINED_FINETUNING",default=False,help="When we want to remove the last layer of an imagenet pretrained model, and replace it for our own tasks")
parser.add_argument("--frame_skip",metavar="FRAME_SKIP",type=int,default=1,help="Number of frames to skip while creating the labels for a video given a pre-trained classifier")
parser.add_argument("--compute_mstd",metavar="COMPUTE_MSTD",default=False,help="If set to True, will compute values for the mean and std of the dataset instead of training/testing a classifier")
parser.add_argument("--video", metavar="VIDEO", default=None, help="Video to classify")
parser.add_argument("--use_existing_frames", type=bool, default=False, help="Whether to use previously extracted frames")
parser.add_argument("--extract_frames_path",metavar="EXTRACT_FRAMES_PATH",default=None,help="If this is set to a path, the classifier runs in testing mode on the frames of a video pointed to by the path and extracts labels every <frame_skip> frames")
parser.add_argument("--pickle_folder",metavar="PICKLE_FOLDER",default=None,help="Folder to save either extracted frame labels, or frames and labels for a video file")
parser.add_argument("--extract_to_xml_folder",metavar="EXTRACT_TO_XML_FOLDER",default=None,help="When set to a path, we create xml annotations for every .jpg file in that path")
parser.add_argument("--segment_label_folder",metavar="SEGMENT_LABEL_FOLDER",default=None,help="When set to a path, we use the labels to split the video into segments that contain the same label")
parser.add_argument("--extract_ground_truth",metavar="EXTRACT_GROUND_TRUTH",default=False,help="When set to true, it will extract the dataset's ground truth annotations into a segmented viewpoint pkl file")
parser.add_argument("--viterbi_trainer",metavar="VITERBI_TRAINER",default=None,help="Setting this to anything finds all the training video label viterbi transition probabilities")
parser.add_argument("--conformal",metavar="CONFORMAL", default=None, help="Setting this to anything means we want to find the MCL for each class")
parser.add_argument("--video_frame_annotation",metavar="VIDEO_FRAME_ANNOTATION",default=None, help="Given a video, extract all its frames in the target folder")
parser.add_argument("--training",metavar="TRAINING",default=None,help="Training phase of the classifier is activated")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default="1e-6",help="Weight decay for the optimizer")
parser.add_argument("--video_target",metavar="VIDEO_TARGET",default=None,help="Conformal video target")

# Uses

# 1. train a classifier (arch, dataset_path, dataset_index, batch_size, num_workers, epochs, save_dir, save_interval, test_interval)
# 2. computing mstd (compute_mstd, ...)

# 3. extracting a video into frames and classifying them: 
# COMMAND python core.py --pretrained_model ./trained_models/vgg_2/model_10.pth  --extract_frames_path ./results/frame_labelling/2016091100/frames/ --pickle_folder ./results/frame_labelling/2016091100/labels/ --video ./data/videos_2016/2016091100.mp4
# MODIFY --use_existing_frames if extracting the frames worked, but labelling them with a pretrained classifier did not

# 4. extracting xml files for a dataset (not required to annotate a video, just for the dataset for training a classifier)
# COMMAND python core.py --extract_to_xml_folder ./results/frame_labelling/2016090800/frames

# 5. segmenting the labels into continuous parts
# COMMAND python core.py --segment_label_folder ./results/frame_labelling/2016090800/labels/ --pickle_folder ./results/frame_labelling/2016121111_resnet_2/labels/

# 6. extracting ground truth into continuous parts (give a dataset folder, a dataset index containing the video in the training set and a pickle folder)
# ...

# 7. constructing a video:
# COMMAND = python construct_video.py --frame_label_folder ./results/frame_labelling/2016091100/ --construct_video True --add_annotations True

# 8. extracting only frames for annotation:
# COMMAND = python core.py --video_frame_annotation <folder path to extract frames> --video <source video>

class viewpoint_classifier():
    def __init__(self, model,dataset_index=0,video_target = None):

        if args.video == None:
            
            self.video_target = video_target
            customset_train = CustomDataset(path = args.dataset_path,subset_type="training",dataset_index=dataset_index,video_target = video_target)
            customset_test = CustomDataset(path = args.dataset_path,subset_type="testing",dataset_index=dataset_index, video_target = video_target)
        
            self.trainloader = torch.utils.data.DataLoader(dataset=customset_train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
            self.testloader = torch.utils.data.DataLoader(dataset=customset_test,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)    
        else:
            video_dataset = VideoDataset(video=args.video, batch_size=args.batch_size,
                                        frame_skip=int(args.frame_skip),image_folder=args.extract_frames_path, use_existing=args.use_existing_frames)
            
            self.videoloader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=1,shuffle=False,num_workers=args.num_workers)

   
        if (model == "alex"):
            self.model = AlexNet()
        elif (model == "vgg"):
            self.model = VGG()
        elif (model == "resnet"):
            self.model = ResNet()

        if args.pretrained_model != None:
            if args.pretrained_finetuning == False:
                self.model.load_state_dict(torch.load(args.pretrained_model))
            else:
                print "DEBUG : Make it load only part of the resnet model"
                #print(self.model)
                #self.model.load_state_dict(torch.load(args.pretrained_model))
                #for param in self.model.parameters():
                #    param.requires_grad = False
                self.model.fc = nn.Linear(512, 1000)
                #print(self.model)
                self.model.load_state_dict(torch.load(args.pretrained_model))
                self.model.fc = nn.Linear(512,3)
                #print(self.model)
                
        self.model.cuda()        
        print "Using weight decay: ",args.weight_decay
        self.optimizer = optim.SGD(self.model.parameters(), weight_decay=float(args.weight_decay),lr=0.01, momentum=0.9,nesterov=True)
        self.criterion = nn.CrossEntropyLoss().cuda()

# Train NN

def train(epochs, vp):
    
    for epoch in range(1, epochs + 1):    
        vp.model.train()
    

        if not args.training:
            args.test_interval = 1
            args.save_interval = 10000

        if args.training:
            for batch_idx, (data, target, frame_num, game_id) in enumerate(vp.trainloader):
                target = target.view(-1)
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output = vp.model(data)
                loss = vp.criterion(output, target)
                
                vp.optimizer.zero_grad()
                loss.backward()
                vp.optimizer.step()

                pred = output.data.max(1)[1] 


                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(vp.trainloader.dataset),
                        100. * batch_idx / len(vp.trainloader), loss.data[0]))
        
        if (epoch % int(args.test_interval) == 0):

            if args.training:
                print "--training accuracy--"
                acc_train(epoch, vp)
            if len(vp.testloader) > 0:
                print "--testing accuracy--"
                acc_test(epoch,vp)

        if (args.save_dir is not None and epoch % int(args.save_interval) == 0):
            directory = args.save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            print("--saving model, epoch : "+str(epoch)+"--")
            torch.save(vp.model.state_dict(), os.path.join(directory, "model_" + str(epoch)+".pth"))
   
 


def conformal_labelling(vp, buckets, p_vals = None, test=False):

    print "Conformal prediction on ", vp.video_target
    ''' 
    vp.model.eval()
    prediction_list = {}
    
    
    softmax = nn.Softmax().cuda()

    for frame_index, (data, target, frame_num, game_id) in enumerate(vp.testloader):
        #print target, frame_num 
        data, target  = data.cuda(), target.cuda()
        data, target  = Variable(data, volatile=True), Variable(target)        
        
        output = vp.model(data)
        soft_output = softmax(output)
        pred = soft_output.cpu().data.max(1)[1]
        #print output, soft_output, pred, target
        for i in range(len(frame_num)):
            #print output[i], soft_output[i], pred[i], target[i]
            #print output[i].cpu().data.numpy()
            o =  output[i].cpu().data.numpy()
            s = soft_output[i].cpu().data.numpy()
            p = pred.numpy()[i]
            t =  target[i].cpu().data.numpy()
            f = frame_num[i]
            prediction_list[str(f)] = (o,s,p,t)
        
    print "DONE"
    
    
    direct = "./conformities/"+vp.video_target
    
    
    if not os.path.isdir(direct):
           os.makedirs(direct)
    with open(direct+"/prediction_list.pkl","wb") as f:
        pickle.dump(prediction_list,f)
    '''

    if test == False:
        direct = "./conformities/"+vp.video_target
        p_list = {}
        o_list = []
        s_list = []
        t_list = []

        prediction_list = pickle.load(open(direct+"/prediction_list.pkl","rb"))  
        for frame in sorted(prediction_list):
                p_list[int(frame)] = prediction_list[frame][2][0]
                o_list.append(prediction_list[frame][0])
                s_list.append(prediction_list[frame][1])
                t_list.append(prediction_list[frame][3][0])
                 
                p = prediction_list[frame][1]
                buckets[0].append(p[0])
                buckets[1].append(p[1])
                buckets[2].append(p[2])

        p_seg = viterbi_preprocessing(p_list)
    else:
        direct = "./conformities/"+vp.video_target
        p_list = {}
        o_list = []
        s_list = []
        t_list = []

        prediction_list = pickle.load(open(direct+"/prediction_list.pkl","rb"))

        '''for key in prediction_list:
            #print key, int(key)
            s = prediction_list.pop(key)
            prediction_list[int(key)] = s
            #prediction_list[int(key)] = prediction_list.pop(key) '''

        '''for key in iter(sorted(prediction_list.items(), key=lambda x: int(x[0]))):
            print key'''
        
        for data in iter(sorted(prediction_list.items(), key=lambda x: int(x[0]))):
                frame = int(data[0])
                p_list[frame] = data[1][2][0]
                o_list.append(data[1][0])
                s_list.append(data[1][1])
                t_list.append(data[1][3][0])

        #print len(p_list)
        p_seg = viterbi_preprocessing(p_list)        
        
        i = 0
        print len(p_seg), len(prediction_list) 
        #print prediction_list
        actual_acc = 0
        for data in iter(sorted(prediction_list.items(), key=lambda x: int(x[0]))):
            
            if i > len(p_seg)-1:
                break 
            frame = int(data[0])
            
            p_value = data[1][1]

            for j in range(3):

                #print buckets[j]
                pp = p_value[j]
                s = 0
                for k in range(len(buckets[j])-1):
                    #print pp, buckets[j][k]
                    
                    if pp > buckets[j][k] and pp < buckets[j][k+1]:
                        s = k
                        #print "TERMINATION"
                        break
                        
                #print k

                p_value[j] = k*1.0 / len(buckets[j])
                #print p_value[j]

            pred = data[1][2]
            target = data[1][3]
            pred_f = p_seg[i]
            i += 1
            #print " "
            
            #print pred_f, target, pred
            #print frame
            #print prediction_list[frame], ",", 
            correct = 0
            if pred_f == target:
                #print "S",
                correct = 1
                actual_acc += 1
            else:
                pass
                #print "OXO",pred_f, target,
            #print p_value
            
            for j in range(3):
                p_vals[correct][j].append(p_value[j])
                
        print "FINAL ACCURACY", actual_acc*100.0/len(prediction_list)
            
 
    # print o_list, s_list, t_list 
    # print buckets

    

     
    
    

    
    # The ones below use all prediction files from above
    
    # Here you will compare the ground truth to the viterbied list
    # for every frame, you have whether it belongs in "correct" or "incorrect" category and
    # you also have its output / softmaxed scores. 

    # Thus you can find the median, mean and std for both categories        

def extract_labels(vp, pickle_folder=None):
  
    vp.model.eval()
    # run the classifier on all frames from the testloader using <frame_skip> to obtain labels
    prediction_list = []
    softmax = nn.Softmax().cuda()
    prediction_list = {}
    for frame_index, (data, target, frame_num, game_id) in enumerate(vp.testloader):
        output = vp.model(data)
        output = softmax(output)
        pred = output.data.max(1)[1]
        mypred = pred.cpu().numpy()
        prediction_list.append(mypred[0][0])    
        prediction_list[frame_index] = mypred[0][0]  
    
    # Save the prediction list for the frames
    with open(pickle_folder+"/predictions.pkl","wb") as f:
        pickle.dump(prediction_list,f)

def extract_video(vp,frame_skip=1,pickle_folder=None):

    vp.model.eval()
    # run the classifier on all frames from the testloader using <frame_skip> to obtain labels
    softmax = nn.Softmax().cuda()
    prediction_list = {}
    frames_checked = 0
    output_list = {}

    for batch_idx, (data, frame_indices) in enumerate(vp.videoloader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        
        output = vp.model(data)
        raw_scores = output.data
        raw_scores = raw_scores.cpu().numpy()
        
        output2 = softmax(output)
        scores = output2.data
        scores = scores.cpu().numpy()    
    
        pred = output2.data.max(1)[1]
        mypred = pred.cpu().numpy()

        np_frames = frame_indices.cpu().numpy()

        for i in range(np_frames.shape[0]):
            prediction_list[np_frames[i]] = mypred[i, 0]  
            output_list[np_frames[i]] = (raw_scores[i], scores[i])

            frames_checked += 1
            if (frames_checked % 10 == 0):
                print("Checked %d frames" % (frames_checked))  
         
    print(prediction_list)
    # Save the prediction list for the frames
    print(os.path.join(pickle_folder, "viewpoint.pkl"))
    
    if not os.path.isdir(pickle_folder):
            os.makedirs(pickle_folder) 
    with open(os.path.join(pickle_folder, "output.pkl"),"wb") as f:
        pickle.dump(output_list,f)
    with open(os.path.join(pickle_folder, "viewpoint.pkl"),"wb") as f:
        pickle.dump(prediction_list,f)   

    with open(os.path.join(pickle_folder, 'viewpoint.csv'), 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in prediction_list.items():
           writer.writerow([key, value])
            


# DEBUG MOVE
from threading import Thread, Lock
def acc_test(epoch, vp):
    vp.model.eval()

    test_loss = 0
    correct = 0

    logsoftmax = nn.LogSoftmax().cuda()
    confusions = {}
    confusion_transitions = {}
    mutex = Lock()
    
    for data, target, frame_num, game_id in vp.testloader:
        target = target.view(-1)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = vp.model(data)
        
        test_loss += F.nll_loss(logsoftmax(output), target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

        flat_pred = pred.cpu().numpy().flatten()
        np_target = target.data.cpu().numpy()
        mutex.acquire()
        try:
            for i in range(np_target.shape[0]):
            
                if game_id[i] not in confusions:
                    conf_mat = np.zeros((3,3))
                    confusions[game_id[i]] = conf_mat
                if game_id[i] not in confusion_transitions:
                    conf_mat = np.zeros((9,9))
                    confusion_transitions[game_id[i]] = conf_mat

        finally:
                mutex.release()
            
         
        prev_pred = flat_pred[0]
        prev_gt = np_target[0]
        for i in range(np_target.shape[0]):
            cc = confusions[game_id[i]]
            cc[np_target[i], flat_pred[i]] += 1
            
            if i > 0:
                from_index = prev_pred % 3+flat_pred[i]*3
                to_index = prev_gt % 3 + np_target[i]*3
                dc = confusion_transitions[game_id[i]]
                dc[to_index, from_index] += 1

            prev_pred = flat_pred[i]
            prev_gt = np_target[i]

    test_loss = test_loss
    test_loss /= len(vp.testloader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(vp.testloader.dataset),
        100. * correct / len(vp.testloader.dataset)))
    print("-- Confusion Matrices --")

    ultimate_matrix = np.zeros((3,3))
    for key in confusions:
        print key
        print(confusions[key])
        ultimate_matrix = np.add(ultimate_matrix,confusions[key])

    print "average matrix"
    print ultimate_matrix
    print np.multiply(ultimate_matrix,1.0/len(vp.testloader.dataset))
         
    for key in confusion_transitions:
        print key
        print confusion_transitions[key]

def acc_train(epoch, vp):
    
    vp.model.eval()

    test_loss = 0
    correct = 0

    logsoftmax = nn.LogSoftmax().cuda()
    for data, target, frame_num, game_id in vp.trainloader:
        target = target.view(-1)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = vp.model(data)
        
        test_loss += F.nll_loss(logsoftmax(output), target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(vp.trainloader) # loss function already averages over batch size
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(vp.trainloader.dataset),
        100. * correct / len(vp.trainloader.dataset)))

def compute_mstd(args):
    customset = CustomDataset(path = args.dataset_path,subset_type="mstd",dataset_index=args.dataset_index)
    mstd_loader = torch.utils.data.DataLoader(dataset=customset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    compute_mean_std(mstd_loader)

# Use viterbi to smooth out results in the segmentation input
def viterbi_preprocessing(segmentation):
    
    # These numbers have not been trained, but hand-picked for debugging 
    p = [0.33,0.33,0.33] # priors 
    # t = [[0.99,0.005,0.005],[0.005,0.99,0.005],[0.005,0.005,0.99]] # based on transitions in the training set
    # e = [[0.98,0.01,0.01],[0.01,0.98,0.01],[0.01,0.01,0.98]] # based on the confusion matrix for the training set
    e = [[1439.0/(1446),0,7.0/1446],[33.0/1480,1446.0/1480,1.0/1480],[1.0/1181,21.0/1181,1159.0/1181]]
    s = [0,1,2]
    t = [[0.31/0.31507,0.005/0.31507,0.00007/0.31507],[0.00001/0.31501,0.31/0.31501,0.005/0.31501],[0.005/0.37501,0.00001/0.37501,0.37/0.37501]]   
    
    # There are underflows occuring when the number of observations is over 7k
    m = max(segmentation.iterkeys())
    segment_length = m/7000
    segmentation_list_viterbi = []

    for iteration in range(segment_length+1):

        obs_fragment_list = []
        from_key = 1 + 7000*iteration
        to_key = min(from_key+7000-1,m)
        
        for key in sorted(segmentation):
            
            #print "Current key", key, "from key", from_key, "to key", to_key
            if int(key) >= from_key and int(key) <= to_key:
                #print "SUCCESS"
                obs_fragment_list.append(segmentation[key])
                #print "key : ",key,"SUCCESS"
            
        #print len(obs_fragment_list) 
         
        obs_fragment_list = viterbi(obs_fragment_list,s,p,t,e)
        for frag in obs_fragment_list:
            segmentation_list_viterbi.append(frag)
    
    #print segmentation_list_viterbi
    
    return segmentation_list_viterbi

# Segment a pickled viewpoint file into the corresponding segmented viewpoint file. Labels go from being per frame to being per segment
def segment_video(target_folder, pickle_folder,ground_truth=False,frame_skip=1):
    
    if not ground_truth:
        segmentation = pickle.load(open(target_folder+"/viewpoint.pkl","rb"))
    else:
        segmentation = pickle.load(open(target_folder+"/gt_viewpoint.pkl","rb"))
    print segmentation
    if not ground_truth: 
        segmentation = viterbi_preprocessing(segmentation)
 
    initial_frame = False
    first_frame = None
    last_frame = None
    segments = []
    change = False
    
    for idx, label in enumerate(segmentation):
        print idx
         
        if not initial_frame:
            initial_frame = True
            first_frame = (label, idx)

        change = False
        if idx+frame_skip >= len(segmentation) or segmentation[idx+frame_skip] != segmentation[idx]:
            
            change = True       
        
        if change:
            second_frame = (segmentation[idx],idx)
            segment = (first_frame, second_frame)
            segments.append(segment)
            if idx+frame_skip != len(segmentation):
                first_frame = (segmentation[idx+frame_skip],idx+frame_skip)
                  
    if not os.path.isdir(pickle_folder):
            os.makedirs(pickle_folder)    
            
    if not ground_truth:
        with open(os.path.join(pickle_folder, "viewpoint_segments.pkl"),"wb") as f:
            pickle.dump(segments,f)    
    else:
        with open(os.path.join(pickle_folder,"gt_viewpoint_segments.pkl"),"wb") as f:
            pickle.dump(segments,f)
    print segments
    # Open pickle file, segment video frames into label-consistent parts with frame_start, frame_finish
    # and save in an appropriate format
    return

# Use this function so that you can write the entire dataset's ground truth labels into a viewpoint_segments.pkl file 
# that can be used to find the IoU against training/testing videos with annotations
def extract_ground_truth(pickle_folder,dataset_index,dataset_folder, video_target = None):
    
    dataset = CustomDataset(path = args.dataset_path,subset_type="training",dataset_index=dataset_index,video_target=video_target)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=False,num_workers=args.num_workers)

    prediction_list = {}
    
    for data, target, frame_num, game_id in dataloader:
        
        fn = int(frame_num.numpy()[0])
        tn = int(target.numpy()[0])
        
        prediction_list[fn] = tn

    with open(os.path.join(pickle_folder,"gt_viewpoint.pkl"),"wb") as f:
        pickle.dump(prediction_list,f)

def viterbi_training():

    dataset = CustomDataset(path = args.dataset_path,subset_type="training",dataset_index=2)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=128,shuffle=False,num_workers=args.num_workers)

    transition_matrix = np.zeros((3,3))
    tn_p = None
    i = 0
    for data, target, frame_num in dataloader:
        
        tn = target.numpy()
        if tn_p != None:   
            for i in range(128):
                transition_matrix[tn_p[i],tn[i]] = transition_matrix[tn_p[i],tn[i]]+1
        tn_p = tn
        i+=1
        
        if i % 100 == 0:
            print np.multiply(transition_matrix,1.0/i)    

def conformal_prediction(vp):

    vp.model.eval()
    dataset = CustomDataset(path = args.dataset_path,subset_type="training",dataset_index=args.dataset_index)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=128,shuffle=False,num_workers=args.num_workers)
    dataset2 =  CustomDataset(path = args.dataset_path,subset_type="testing",dataset_index=args.dataset_index)
    dataloader2 = torch.utils.data.DataLoader(dataset=dataset2, batch_size=128, shuffle=False, num_workers=args.num_workers)

    MCL = [[],[],[]]
    # only include samples that have : correct classification with low p-value
    # or incorrect classification
    # ask xu
    for data, target, frame_num, game_id in dataloader:
        tn = target.numpy()
        data = data.cuda()
        data = Variable(data)
        output = vp.model(data)
        print output
        for i in range(128):
            MCL[tn[i]].append(output[i])
    for data, target, frame_num, game_id in dataloader2:
        tn = target.numpy()
        data = data.cuda()
        data = Variable(data)
        output = vp.model(data)
        print output
        for i in range(128):
            MCL[tn[i]].append(output[i])

    pickle.dump(MCL,open("./MCL.pkl","wb"))
    print MCL
    MCL = [MCL[0].sort, MCL[1].sort, MCL[2].sort]
    
    # train-test 2014
    for epsilon in range(0,11):
        e = epsilon/10.0
        # for all testing samples in a dataset
        # calculate number of samples that the p classification of the sample beats in its class
        # normalize by the number of samples
        # this is the p-value of the confidence of the sample
        # if p > 1-e save it into a new list

    # include all bad samples (p value is too low or false classification with high confidence)
    

        
def main():
    global args

    args = parser.parse_args()

    if args.viterbi_trainer != None:
        viterbi_training()
        return

    if args.segment_label_folder != None:
        segment_video(args.segment_label_folder, args.pickle_folder)
        return
    
    if args.extract_ground_truth:
        extract_ground_truth(args.pickle_folder,int(args.dataset_index),args.dataset_path)
        segment_video(args.pickle_folder, args.pickle_folder,ground_truth=True)
        return
    
    if args.conformal != None:
        #directories = [name for name in os.listdir("training_data") if os.path.isdir(name)]
        dirs = os.walk("./training_data").next()[1]
        print dirs
        
        print "CONFORMAL"

        buckets = [[],[],[]]    

        train_buckets = False
        if train_buckets:
            for mydir in dirs:
                if mydir == "aug":
                    continue
                
                
                vp = viewpoint_classifier(args.arch,dataset_index=int(args.dataset_index), video_target = mydir)
                if mydir not in ["2014101912","2014091411","2016091808","2016112710"]:
                    conformal_labelling(vp,buckets)
                
            print len(buckets[0]), len(buckets[1]), len(buckets[2])
            pickle.dump( buckets, open( "buckets.p", "wb" ) )    
        
        buckets = pickle.load(open("buckets.p","rb"))
        for i in range(3):
            buckets[i].sort()
        #print buckets
        epsilon = 0.1
        p_vals = [[[],[],[]],[[],[],[]]]
 
        for mydir in dirs:
            if mydir in ["2014101912","2014091411","2016112710"]:
                vp = viewpoint_classifier(args.arch,dataset_index=int(args.dataset_index), video_target = mydir)
                conformal_labelling(vp, buckets,p_vals = p_vals, test=True)

        #print p_vals

        # For all buckets of correct-class combinations, find mean, median and std
        
        for c in range(2):
            for e in range(3):
                d = p_vals[c][e]
                print len(d)
                #print d[0:30] 
                #print d[-20:-1]
                means =  sum(d, 0.0) / len(d)
                
                medians = np.median(np.array(d)) 
                stds = np.std(np.array(d))
                print means, medians, stds

        print "ALL DONE"
        return   
        

    if args.compute_mstd:        
        compute_mstd(args)
        return
    

    if args.video_frame_annotation != None:
        
        deconstruct_video(args.video,args.video_frame_annotation,frame_skip=args.frame_skip)
        print "deconstruction complete"
        #filename = os.path.basename(args.video).split(".")[0]
        #directory_frames = args.video_frame_annotation+"/"+filename        
        #construct_annotator(directory_frames,directory_frames,filename) 
        #print "construction complete"
        #subprocess.check_call(["./up.sh","upload",directory_frames+"/video.avi","./video_annotations/"+filename+".avi"])
        #print "upload complete"
        return 

    if not args.label_dataset and args.video == None:
        vp = viewpoint_classifier(args.arch,dataset_index=int(args.dataset_index))    
        train(int(args.epochs), vp) 
    elif args.video != None:
        vp = viewpoint_classifier(args.arch)
        
        if (args.pickle_folder == None):
            args.pickle_folder = args.extract_frames_path

        extract_video(vp, args.frame_skip, pickle_folder=args.pickle_folder)    
    

if __name__ == "__main__":

    main()


