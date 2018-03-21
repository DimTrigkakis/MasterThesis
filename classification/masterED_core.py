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
import torchvision.utils as vutils
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

from datasets import MasterPlaysets
from models import DimoAutoSequence
from magnificentoracle import MagnificentOracle

import excitation_bp as eb

# No more "too many open files" problem
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')

parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/full_frame_data_2016/",help="Path that contains the dataset to be processed")

parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=16, help="Use the batch size default: %d" % (2))
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=12, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=100, help="Number of epochs to train for, default = 24")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default=0,help="Weight decay for the optimizer")#"1e-6"

parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=1, help="How often should the model save, default = 2")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=20, help="How often to test the model, default = 1")

parser.add_argument("--pretrained_model",metavar="PRETRAINED_MODEL",default=None,help="Point to a pretrained viewpoint model")
parser.add_argument("--saved_model",metavar="SAVED_MODEL",default=None,help="Point to a pretrained playtype model")
parser.add_argument("--training",metavar="TRAINING",default=False,help="Training phase of the classifier is activated")
parser.add_argument("--subsampling",default=0,help="Subsampling type for classifiers")
parser.add_argument("--viewpoint_included",default="both",help="Select from 'both' viewpoints, '0' or '1'")

parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to fill with logger information")

mylogger = None

class playtype_classifier():
    def __init__(self, model="dimo" ,dataset_index=0):

        # First, create the weighted sampler by analyzing the dataset and ascribing proper class weights

        print "Initializing playtype classifier"

        final_length = 16
        view = "both"

        first_save = False
        sampler = None
        if first_save: # takes > 1 minute to even compute, compare with 5 seconds load time
            customset_preprocess = MasterPlaysets(path = args.dataset_path,subset_type="training", retrieve_images=False, subsampling=None,fill=None, from_file="gt", part="part", view=view) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
            
            self.processloader = torch.utils.data.DataLoader(dataset=customset_preprocess,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))

            class_presence = [0, 0, 0]
            for batch_idx, (imgs, play_type) in enumerate(self.processloader):
                class_presence[play_type.cpu().numpy()[0][0]] += 1
            
            sample_weights = [ (1.0 / class_presence[play_type.cpu().numpy()[0][0]]) for (imgs, play_type) in self.processloader]

            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights,len(self.processloader), replacement=True)

            # mstd = MasterPlaysets(path = args.dataset_path,subset_type="training", retrieve_images=True, subsampling=None,fill=None, from_file="gt", part="part", view="both") # gt vs est, whole vs. part, 'both' vs '0' vs '1'
            # self.processloader_weighted = torch.utils.data.DataLoader(sampler=sampler,dataset=mstd,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))
            # customset_preprocess.compute_mean_std(self.processloader_weighted)       

            print "Loading datasets"
            customset_train = MasterPlaysets(path = args.dataset_path,subset_type="training", retrieve_images=True, subsampling=None,fill=None, from_file="gt", part="part", view=view) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
            customset_test = MasterPlaysets(path = args.dataset_path,subset_type="testing", retrieve_images=True, subsampling=None,fill=None, from_file="gt", part="part", view=view) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
            
            customsets = [customset_preprocess, customset_train, customset_test]
            pickle.dump(customsets,open("./junk/customset_v0.pkl","wb"))
        else:
            customsets = pickle.load(open("./junk/customset_v0.pkl","rb"))
            customset_preprocess, customset_train, customset_test = customsets
            self.processloader = torch.utils.data.DataLoader(dataset=customset_preprocess,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))
            class_presence = [0, 0, 0]
            for batch_idx, (imgs, play_type) in enumerate(self.processloader):
                class_presence[play_type.cpu().numpy()[0][0]] += 1
            sample_weights = [ (1.0 / class_presence[play_type.cpu().numpy()[0][0]]) for (imgs, play_type) in self.processloader]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights,len(self.processloader), replacement=True)

        customset_train.proper_length = final_length
        customset_test.proper_length = final_length

        self.trainloader = torch.utils.data.DataLoader(dataset=customset_train,batch_size=int(args.batch_size),shuffle=False,num_workers=int(args.num_workers)) 
        self.train_acc_loader = torch.utils.data.DataLoader(dataset=customset_train,batch_size=int(args.batch_size),shuffle=False,num_workers=int(args.num_workers)) 
        #self.trainloader = torch.utils.data.DataLoader(sampler=sampler,dataset=customset_train,batch_size=int(args.batch_size),num_workers=int(args.num_workers)) 
        #self.train_acc_loader = torch.utils.data.DataLoader(sampler=sampler,dataset=customset_train,batch_size=int(args.batch_size),num_workers=int(args.num_workers))
        #self.train_acc_loader = torch.utils.data.DataLoader(dataset=customset_train,batch_size=int(args.batch_size),shuffle=False,num_workers=int(args.num_workers))
        self.test_acc_loader = torch.utils.data.DataLoader(dataset=customset_test,batch_size=int(args.batch_size),shuffle=False,num_workers=int(args.num_workers))  

        self.lr = 0.0001
        mylogger.log("-dotted-line")   
        mylogger.log("Using worker count: {}".format(args.num_workers))
        mylogger.log("Using epoch count: {}".format(args.epochs))
        mylogger.log("Using model  <{}>".format(args.pretrained_model))
        mylogger.log("Using weight decay: {}".format(args.weight_decay))
        if args.training:
            mylogger.log("Training schedule type <training>")
        mylogger.log("Using learning rate  <{}>".format(self.lr))

        pretrained_model_selection = args.pretrained_model
        #pretrained_model_selection = "./results/viewpoint_models/vgg_viewpoint_experiments/v1/model_epoch_5.pth"

        print "Building model"
        self.num_inner_nodes = 2
        self.length_of_sequence = final_length
        self.model = DimoAutoSequence(pretrained_model=pretrained_model_selection,num_inner_nodes=self.num_inner_nodes,max_len=self.length_of_sequence).cuda()
        self.model = nn.DataParallel(self.model,device_ids=[0,1,2,3]).cuda()
        if args.saved_model != None:
            self.model.load_state_dict(torch.load(args.saved_model))

        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=float(args.weight_decay), lr= self.lr)
        self.criterion = nn.NLLLoss().cuda()
        mylogger.log("Playtype classifier ready")
        mylogger.log(self.model)

# Train NN

def train(epochs, pt):
    
    mylogger.log("-dotted-line")
    mylogger.log("Initializing training")

    loss_batch_skip = 1

    if not args.training:
        args.test_interval = 1
        args.save_interval = 2
        epochs = 1

    for epoch in range(1, epochs + 1): 

        mylogger.log("Beginning of epoch <{}>".format(epoch)) 


        if args.training:    
            pt.model.train()
            pt.model.eval()

            #pt.optimizer = optim.Adam(pt.model.parameters(), weight_decay=float(args.weight_decay), lr= pt.lr*(0.5 ** (epoch // 20)))
            # print pt.lr*(0.5 ** (epoch // 8))

            timer = time.time()
            # print(epoch)
        
            for batch_idx, (imgs, play_type) in enumerate(pt.trainloader):

                if batch_idx >= 1:
                    continue

                    #if batch_idx == len(pt.trainloader) - 1 or batch_idx > 1:
                    #    continue

                    #print imgs.size(), play_type.size(), play_type

                play_type = Variable(play_type.cuda())
                imgs = Variable(imgs.cuda())

                for j in range(1500):
                    output = pt.model(imgs)
                    prediction = output.data.cpu().max(1)[1].numpy()
                    play_type_ = play_type.squeeze()

                    loss = pt.criterion(output, play_type_)

                    prediction = np.squeeze(prediction)
                    play_type_target = play_type_.data.cpu().numpy()
                    correct = np.sum(prediction == play_type_target)
                    print(epoch, batch_idx, prediction, play_type_target, loss.data.cpu().numpy()[0]/int(args.batch_size), correct)

                    
                    pt.optimizer.zero_grad()
                    loss.backward()
                    pt.optimizer.step()                
                    
                    if loss.data.cpu().numpy() != loss.data.cpu().numpy():
                        print "BAD LOSS, NAN"
                        exit()

                    if batch_idx % loss_batch_skip == 0:
                        mylogger.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx , len(pt.trainloader),
                            100. * batch_idx / len(pt.trainloader), loss.data.cpu().numpy()[0]/int(args.batch_size)))
            
            mylogger.log("Time void from epoch: {}".format(time.time()-timer))

        '''
        if (args.save_dir is not None and epoch % int(args.save_interval) == 0):
            directory = args.save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
             
            mylogger.log("--saving model, epoch : "+str(epoch)+"--")
            torch.save(pt.model.state_dict(), os.path.join(directory, "entire_model_" + str(epoch)+".pth"))
        
        if (epoch % int(args.test_interval) == 0):
            acc(epoch, pt)    
        '''

        

def acc(epoch, pt):
    

    pt.model.eval()

    train_loss = 0
    correct = 0
    data_size = 0.0

    for loader in [pt.train_acc_loader, pt.test_acc_loader]:
        conf_mat = np.zeros((3,3))
        if loader == pt.test_acc_loader:
            continue
        for batch_idx, (imgs, play_type) in enumerate(loader):
            if batch_idx >= 1:
                continue

            #if batch_idx == len(loader) - 1 :
            #    continue
            #print(imgs.size(), play_type.size())

            batch_size = play_type.cpu().size()[0]
            play_type = Variable(play_type, volatile = True, requires_grad = False).cuda()
            imgs = Variable(imgs, volatile = True, requires_grad = False).cuda()

            output = pt.model(imgs)
            print(output)
            play_type = play_type.squeeze()

            prediction = output.data.cpu().max(1)[1].numpy()
            #prediction = np.squeeze(prediction)
            loss = pt.criterion(output, play_type)

            play_type_target = play_type.data.cpu().numpy()
            correct += np.sum(prediction == play_type_target)
            train_loss += loss.data.cpu().numpy()[0]

            data_size += batch_size

            for i in range(batch_size):
                print(prediction.shape, play_type_target.shape, batch_size, i, batch_idx, len(loader))
                conf_mat[int(play_type_target[i])][int(prediction[i])] += 1 

            if random.random() > 0.95:
                mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction, play_type_target, np.sum(prediction == play_type_target),int(batch_size)))
        
        mylogger.log('Set, Average loss: {:.4f} Accuracy: {}/{} ({:.0f}%)'.format(train_loss/data_size, correct, data_size,100.0*correct/(data_size))) 
        conf_mat = np.multiply(conf_mat,1.0/data_size)
        mylogger.log('Conf matrix')
        mylogger.log(conf_mat)

def final_expectation(sq):

    sq.model = sq.model.module
    sq.model.cpu()
    sq.model.eval()
    

    eb.use_eb(True)

    for batch_idx, (sequence, target) in enumerate(sq.exciteloader):

        picnum = sq.length_of_sequence

        mean=[0.3886, 0.4391, 0.3203]
        std=[0.2294, 0.2396, 0.2175]
        unnormalize = transforms.Normalize(
            mean=[-0.3886/0.2294, -0.4391/0.2396, -0.3203/0.2175],
            std=[1/0.2294, 1/0.2396, 1/0.2175]
        )

        sequence = Variable(sequence)
        target = Variable(target)

        '''
        mysequence = sequence[0].cpu().data

        for i in range(picnum):
            img = mysequence[i,:,:,:]
            img = unnormalize(img)
            img = img.permute(1,2,0)
            plt.subplot(math.sqrt(picnum),math.sqrt(picnum),i+1)
            plt.imshow(img.numpy())
            plt.draw()
            plt.axis('off')
        plt.show()
        '''

        output_model = sq.model(sequence)
        pred = output_model.data.max(1)[1]
        print(pred)
        print(target)

        if sq.model.decode:
            layer_top = list(sq.model.modules())[0].batch_to_classifier[0]
            layer_second = list(sq.model.modules())[0].final_classifier[1]
            target_layer = list(sq.model.modules())[0].features[2]
        else:
            layer_top = list(sq.model.modules())[0].decoder_to_classifier[3]
            layer_second = list(sq.model.modules())[0].decoder_to_classifier[1]
            target_layer = list(sq.model.modules())[0].features[2]            

        for label_choice in [0,1]:

            top_dh = torch.zeros(1, layer_top.out_features)
            top_dh[0,label_choice] = 1

            # since sequence is a series of images, it does not batch and send one signal but sends multiple signals
            timer = time.time()
            sq.optimizer.zero_grad()
            grad = eb.contrastive_eb(sq.model, sequence, layer_top, layer_second, target=target_layer, top_dh=top_dh)
            #grad = eb.eb(sq.model, sequence, layer_top, target=target_layer, top_dh=top_dh)
            mylogger.log("Time void from contrastive EB: {}".format(time.time()-timer))
            #print(grad)
            #print(torch.max(grad))

            grad_ = grad.numpy()[:,:,:,:]
            grad_ = np.mean(grad_,axis=1)

            mylogger.log(pred)
            mylogger.log(target)

            plt.figure(1)
            for i in range(picnum):
                img = grad_[i,:,:]
                plt.subplot(math.sqrt(picnum),math.sqrt(picnum),i+1)
                plt.imshow(img)
                plt.draw()
                plt.axis('off')
            plt.show()

        mysequence = sequence[0].cpu().data

        for i in range(picnum):
            img = mysequence[i,:,:,:]
            img = unnormalize(img)
            img = img.permute(1,2,0)
            plt.subplot(math.sqrt(picnum),math.sqrt(picnum),i+1)
            plt.imshow(img.numpy())
            plt.draw()
            plt.axis('off')
        plt.show()

        # it kind off works
        exit()   

        pred = output.data.max(1)[1] 
        
        layer_top = list(vp.VGG.modules())[0].classifier[6] # output layer of network
        layer_second = list(vp.VGG.modules())[0].classifier[4]
        target_layer = list(vp.VGG.modules())[0].features[2]

        prediction = pred.cpu().numpy()[0][0]

        # excite the prediction neuron

        #self.mean = [ 0.54005252  ,0.51546471  ,0.48611783]
        #self.std = [ 0.42398592  ,0.4154406   ,0.43313607]
        unnormalize = transforms.Normalize(mean=[-0.53/0.4,-0.50/0.4,-0.47/0.4],std=[1/0.4,1/0.4,1/0.4])
        topil = transforms.ToPILImage()
        chunk_of_tensor = unnormalize(first_image[0])
        img = topil(chunk_of_tensor)

        plt.figure(1)
        plt.subplot(4,4,1)
        plt.title("Prediction: "+str(my_keys[prediction]))
        plt.imshow(img)
        plt.draw()
        plt.axis('off')

        for i in range(30,40):

            top_dh = torch.zeros(1, layer_top.out_features)

            idd = -1
            if i != 37:
                idd = i ; top_dh[0, idd] = 1
            else:
                idd = first_target; top_dh[0, idd] = 1

            grad = eb.contrastive_eb(vp.VGG, datum, layer_top, layer_second ,target=target_layer, top_dh=top_dh)
            
            grad_ = grad.numpy()[0,:,:,:]
            grad_ = np.mean(grad_,axis=0)

            ix = i-28
            # ix from 0 to 8
            plt.subplot(4,4,ix)
            plt.axis('off')
            plt.title("Contrast: "+str(my_keys[idd]))
            plt.imshow(grad_)
            plt.draw()

        plt.show()

def final_expectation_dataset(sq):

    sq.model.eval()
    sq.model = sq.model.module
    sq.model.cpu()
    sq.model.eval()
    print sq.model

    eb.use_eb(True)

    for batch_idx, (sequence, target) in enumerate(sq.exciteloader_train):

        np.save("./data/full_frame_data_2016_orig_trinary/"+str(batch_idx)+"_seq.npy", sequence.numpy())
        np.save("./data/full_frame_data_2016_orig_trinary/"+str(batch_idx)+"_tar.npy", target.numpy())
        continue

        timer_init = time.time()
        picnum = sq.length_of_sequence

        mean=[0.3886, 0.4391, 0.3203]
        std=[0.2294, 0.2396, 0.2175]
        unnormalize = transforms.Normalize(
            mean=[-0.3886/0.2294, -0.4391/0.2396, -0.3203/0.2175],
            std=[1/0.2294, 1/0.2396, 1/0.2175]
        )

        # save sequence and target here


        sequence = Variable(sequence)
        target = Variable(target)

        mylogger.log("Making a prediction")
        output_model = sq.model(sequence)
        print output_model, target
        mylogger.log("Made the prediction")
        pred = output_model.data.max(1)[1]

        prediction = pred.numpy()
        mytarget = target.data.cpu().numpy()

        if prediction != mytarget:
            continue
        
        layer_top = list(sq.model.modules())[0].decoder_to_classifier[6]
        layer_second = list(sq.model.modules())[0].decoder_to_classifier[4]
        target_layer = list(sq.model.modules())[0].features[2] 
        top_dh = torch.zeros(sequence.size()[0], layer_top.out_features)

        for i in range(sequence.size()[0]):
            top_dh[i,mytarget[i][0]] = 1 # ground truth based contrastive signal

        print top_dh

        timer = time.time()
        mylogger.log("Using eb")
        sq.optimizer.zero_grad()
        grad = eb.contrastive_eb(sq.model, sequence, layer_top, layer_second, target=target_layer, top_dh=top_dh)
        mylogger.log("Time void from contrastive EB: {}".format(time.time()-timer))

        print grad.numpy().shape
        grad_ = grad.numpy()[:,:,:,:]
        grad_ = np.mean(grad_,axis=1)

        np.save("./data/full_frame_data_2016_npy_trinary/"+str(batch_idx)+".npy", grad_)

        print time.time()-timer_init
        '''
        plt.figure(1)
        for i in range(picnum):
            img = grad_[i,:,:]
            plt.subplot(math.sqrt(picnum),math.sqrt(picnum),i+1)
            plt.imshow(img)
            plt.draw()
            plt.axis('off')
        plt.show()

        mysequence = sequence[0].cpu().data

        for i in range(picnum):
            img = mysequence[i,:,:,:]
            img = unnormalize(img)
            img = img.permute(1,2,0)
            plt.subplot(math.sqrt(picnum),math.sqrt(picnum),i+1)
            plt.imshow(img.numpy())
            plt.draw()
            plt.axis('off')
        plt.show()
        '''
                
def main():
    global args, mylogger
     
    args = parser.parse_args() 
    mylogger = MagnificentOracle()
    mylogger.set_log(logfile=args.log)
    mylogger.log("-dotted-line")  

    pt = playtype_classifier()
    train(int(args.epochs), pt)
    #final_expectation(pt)
    #acc_test(1,pt)   # 91% testing acc at 48 epochs for binary, 85% testing acc at 48 epochs for trinary
    #final_expectation_dataset(pt)


if __name__ == "__main__":

    main()


