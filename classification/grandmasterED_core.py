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
from shutil import copyfile

from datasets import MasterPlaysets
from models import TCNPlaytypes
from magnificentoracle import MagnificentOracle

import excitation_bp as eb

import torch.backends.cudnn as cudnn
from IOUmetric import IoU_pairs

cudnn.benchmark = True
cudnn.fastest = True

# No more "too many open files" problem
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')

parser.add_argument("--dataset_path", metavar='DATASET_PATH', default="./data/full_frame_data_2016/",help="Path that contains the dataset to be processed")

parser.add_argument("--batch_size", "-b", metavar="BATCH_SIZE", default=2, help="Use the batch size default: %d" % (2))
parser.add_argument("--num_workers", metavar="NUM_WORKERS", default=16, help="Default number of workers for data loader")
parser.add_argument("--epochs", "-e", metavar="EPOCHS", default=100, help="Number of epochs to train for, default = 24")
parser.add_argument("--weight_decay",metavar="WEIGHT_DECAY",default=1e-6,help="Weight decay for the optimizer")#"1e-6"

parser.add_argument("--save_dir", metavar="SAVE_DIR", default=None, help="Directory to save to, defaults to None")
parser.add_argument("--save_interval", metavar="SAVE_INTERVAL", default=1, help="How often should the model save, default = 2")
parser.add_argument("--test_interval", metavar="TEST_INTERVAL", default=1, help="How often to test the model, default = 1")

parser.add_argument("--pretrained_model",metavar="PRETRAINED_MODEL",default=None,help="Point to a pretrained viewpoint model")
parser.add_argument("--saved_model",metavar="SAVED_MODEL",default=None,help="Point to a pretrained playtype model")
parser.add_argument("--training",metavar="TRAINING",default=False,help="Training phase of the classifier is activated")

parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to fill with logger information")

mylogger = None

def ground_truth_files():
    full_datasets = MasterPlaysets(path = args.dataset_path,subset_type="entire", retrieve_images=False, from_file="gt", part="whole", view="both", proper_length = -1) # gt vs est, whole vs. part, 'both' vs '0' vs '1'            

class playtype_classifier():
    def __init__(self, model="TCN"):

        print "Initializing playtype classifier"

        # Hyperparameters #############################################
        proper_length = 32 # interval in each sample
        view = "both"   # both or 0 or 1
        from_file = "gt" # otherwise, if it's pred, it should assign majority labels to all intervals
        vgg_type = "D"
        part = "whole" # or whole
        encoder_nodes = 256 # 256 better
        num_nodes = [64,96]
        self.lr = args.batch_size*1e-5*0.6

        self.proper_length = proper_length
        frozen = False
        if frozen:
            self.lr = args.batch_size*1e-6
        #################################################

        string_file = str(proper_length)+"_"+view+"_"+from_file+"_"+part+".pkl"
        mylogger.log(string_file+"_"+vgg_type+"_"+str(num_nodes[0])+"_"+str(num_nodes[1])+"_"+str(encoder_nodes)+"_"+str(args.batch_size)+"_"+str(self.lr))

        # Part is "part" since we want to only use the validation set. In the end we can test the final testing accuracy by setting part to "whole"

        excitement = True

        if not excitement:
            if not os.path.isfile("./dataset_preparations/"+string_file):
                
                print "creating file in ./dataset_preparations to retrieve dataset easily"

                # takes > 1 minute to even compute, compare with 5 seconds load time
                customset_preprocess = MasterPlaysets(path = args.dataset_path,subset_type="training", retrieve_images=False, from_file=from_file, part=part, view=view, proper_length = proper_length) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
                
                self.processloader = torch.utils.data.DataLoader(dataset=customset_preprocess,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))

                class_presence = [0, 0, 0]
                for batch_idx, (imgs, play_type,_) in enumerate(self.processloader):
                    class_presence[play_type.cpu().numpy()[0][0]] += 1
                
                sample_weights = [ (1.0 / class_presence[play_type.cpu().numpy()[0][0]]) for (imgs, play_type,_) in self.processloader]

                sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights,len(self.processloader), replacement=True)

                # mstd = MasterPlaysets(path = args.dataset_path,subset_type="training", retrieve_images=True, subsampling=None,fill=None, from_file="gt", part="part", view="both") # gt vs est, whole vs. part, 'both' vs '0' vs '1'
                # self.processloader_weighted = torch.utils.data.DataLoader(sampler=sampler,dataset=mstd,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))
                # customset_preprocess.compute_mean_std(self.processloader_weighted)       

                print "Loading datasets"
                customset_train = MasterPlaysets(path = args.dataset_path,subset_type="training", retrieve_images=True, from_file=from_file, part=part, view=view, proper_length = proper_length) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
                customset_val = MasterPlaysets(path = args.dataset_path,subset_type="testing", retrieve_images=True, from_file=from_file, part=part, view=view, proper_length = proper_length) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
                customset_test = MasterPlaysets(path = args.dataset_path,subset_type="testing", retrieve_images=True, from_file=from_file, part="whole", view=view, proper_length = proper_length) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
                
                customsets = [customset_preprocess, customset_train, customset_val, customset_test]
                pickle.dump(customsets,open("./dataset_preparations/"+string_file,"wb"))

            else:

                customsets = pickle.load(open("./dataset_preparations/"+string_file,"rb"))
                #customset_test = customsets[0]
                customset_preprocess, customset_train, customset_val, customset_test = customsets
                self.processloader = torch.utils.data.DataLoader(dataset=customset_preprocess,batch_size=int(1),shuffle=False,num_workers=int(args.num_workers))
                class_presence = [0, 0, 0]
                for batch_idx, (imgs, play_type,_) in enumerate(self.processloader):
                    class_presence[play_type.cpu().numpy()[0][0]] += 1
                sample_weights = [ (1.0 / class_presence[play_type.cpu().numpy()[0][0]]) for (imgs, play_type,_) in self.processloader]
                sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights,len(self.processloader), replacement=True)

            customset_train.proper_length = proper_length
            customset_test.proper_length = proper_length
            customset_val.proper_length = proper_length

            self.trainloader = torch.utils.data.DataLoader(sampler=sampler, dataset=customset_train,batch_size=int(args.batch_size),shuffle=True,num_workers=int(args.num_workers)) 
            self.train_acc_loader = torch.utils.data.DataLoader(dataset=customset_train,batch_size=1,shuffle=False,num_workers=int(args.num_workers)) 
            self.val_acc_loader = torch.utils.data.DataLoader(dataset=customset_val,batch_size=1,shuffle=False,num_workers=int(args.num_workers))  
            self.test_acc_loader = torch.utils.data.DataLoader(dataset=customset_test,batch_size=1,shuffle=False,num_workers=int(args.num_workers))  
        else:
            #customset_excite = MasterPlaysets(path = args.dataset_path,subset_type="exciting", retrieve_images=True, from_file=from_file, part="whole", view=view, proper_length = proper_length) # gt vs est, whole vs. part, 'both' vs '0' vs '1'
            #pickle.dump(customset_excite,open("./dataset_preparations/exciteloader.pkl","wb"))

            customset_excite = pickle.load(open("./dataset_preparations/exciteloader.pkl","rb")) 
            self.exciteloader_train = torch.utils.data.DataLoader(dataset=customset_excite,batch_size=1,shuffle=False,num_workers=int(args.num_workers))
            print(len(self.exciteloader_train))

        print "Building model"

        self.model = TCNPlaytypes(pretrained_model=args.pretrained_model,max_len=proper_length,vgg_type=vgg_type, num_nodes=num_nodes, encoder_nodes=encoder_nodes).cuda()
        self.model = nn.DataParallel(self.model,device_ids=[0,1]).cuda()
        if args.saved_model != None:
            self.model.load_state_dict(torch.load(args.saved_model))

        if frozen == True:
            # go through model arguments first
            mylist = [{'params' : self.model.module.encoder.parameters()},{'params' :self.model.module.decoder_to_classifier.parameters()} ,
            {'params' : self.model.module.features.parameters(), 'lr' : 0},{'params' :self.model.module.classifier.parameters(), 'lr' : 0}]
            print(mylist)
            #optimizer = optim.Adam([param for name, param in model.state_dict().iteritems()
            #                if 'foo' in name], lr=args.lr)

            self.optimizer = optim.Adam(mylist, lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(),weight_decay=float(args.weight_decay), lr= self.lr)

        self.criterion = nn.NLLLoss().cuda()
        if args.training:
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


    '''
    pt.trainloader.retrieve_images=False
    smallest = 100
    count_50 = 0
    for batch_idx, (imgs, play_type) in enumerate(pt.train_acc_loader):
        print(imgs.size()[1])
        if imgs.size()[1] < smallest:
            smallest = imgs.size()[1]
        if imgs.size()[1] < 32:
            count_50 += 1
    print(smallest, count_50)
    exit()
    '''

    for epoch in range(1, epochs + 1): 

        mylogger.log("Beginning of epoch <{}>".format(epoch)) 

        timer = time.time()
        if args.training:    
            pt.model.train()

            #pt.model.eval() #?

            #pt.optimizer = optim.Adam(pt.model.parameters(), weight_decay=float(args.weight_decay), lr= pt.lr*(0.5 ** (epoch // 20)))
            # print pt.lr*(0.5 ** (epoch // 8))

            # print(epoch)
        
            for batch_idx, (imgs, play_type,_) in enumerate(pt.trainloader):

                if batch_idx == len(pt.trainloader) - 1:
                    continue
                
                play_type = Variable(play_type.cuda())
                imgs = Variable(imgs.cuda())
                true_epoch = 0
                #print imgs.size(), play_type.size(), play_type

                

                output = pt.model(imgs)
                prediction = output.data.cpu().max(1)[1].numpy()
                play_type_ = play_type.squeeze()

                loss = pt.criterion(output, play_type_)

                prediction = np.squeeze(prediction)
                play_type_target = play_type_.data.cpu().numpy()
                correct = np.sum(prediction == play_type_target)
                print(epoch, batch_idx, prediction, play_type_target, loss.data.cpu().numpy()[0]/int(args.batch_size), correct, "/", args.batch_size)

                pt.optimizer.zero_grad()
                loss.backward()
                pt.optimizer.step()                
                
                if random.random() > 0.95:
                    mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction, play_type_target, np.sum(prediction == play_type_target),int(play_type.cpu().size()[0])))

                if loss.data.cpu().numpy() != loss.data.cpu().numpy():
                    print "BAD LOSS, NAN"
                    exit()

                if batch_idx % loss_batch_skip == 0:
                    mylogger.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx , len(pt.trainloader),
                        100. * batch_idx / len(pt.trainloader), loss.data.cpu().numpy()[0]/int(args.batch_size)))
            

        
        if (args.save_dir is not None and epoch % int(args.save_interval) == 0):
            directory = args.save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
             
            mylogger.log("--saving model, epoch : "+str(epoch)+"--")
            torch.save(pt.model.state_dict(), os.path.join(directory, "entire_model_" + str(epoch)+".pth"))
        
        if (epoch % int(args.test_interval) == 0):
            acc(epoch, pt)    
        
        mylogger.log("Time void from epoch: {}".format(time.time()-timer))
        

def acc(epoch, pt):
    

    pt.model.eval()

    print(len(pt.test_acc_loader))
    for loader in [pt.test_acc_loader]:#[pt.train_acc_loader, pt.val_acc_loader]:#[pt.test_acc_loader]:#[pt.train_acc_loader, pt.val_acc_loader]:
    	
        conf_mat = np.zeros((3,3))
        train_loss = 0
        correct = 0
        data_size = 0.0


        for batch_idx, (imgs, play_type, _) in enumerate(loader):
            #if batch_idx >= 1:
            #    continue

            
            if loader == pt.train_acc_loader:
                if batch_idx == len(loader) - 1:
                    continue

                #print(imgs.size(), play_type.size())

                #print(imgs.size())
                batch_size = play_type.cpu().size()[0]
                play_type = Variable(play_type, volatile = True, requires_grad = False).cuda()
                imgs = Variable(imgs, volatile = True, requires_grad = False).cuda()

                output = pt.model(imgs)
                play_type = play_type.squeeze(1)

                prediction = output.data.cpu().max(1)[1].numpy()
                prediction = np.squeeze(prediction)
                loss = pt.criterion(output, play_type)

                play_type_target = play_type.data.cpu().numpy()
                correct += np.sum(prediction == play_type_target)
                train_loss += loss.data.cpu().numpy()[0]

                data_size += batch_size

                for i in range(batch_size):
                    conf_mat[int(play_type_target)][int(prediction)] += 1 

                if random.random() > 0.95:
                    mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction, play_type_target, np.sum(prediction == play_type_target),int(batch_size)))
            elif loader == pt.val_acc_loader:


                if batch_idx == len(loader) - 1:
                    continue

                # Break batch into segments of proper length
                # remove the /4 for non overlapping
                remainder = int(imgs.size()[1]-pt.proper_length) % int(pt.proper_length/4)

                mypredictions = [0,0,0]
                for t in range(0,int((imgs.size()[1]-pt.proper_length)/(pt.proper_length/4))):
                    #print(int(imgs.size()[1]/(pt.proper_length/4)), remainder)
                    imgs_temp = imgs[:,int(remainder/2)+t*(pt.proper_length/4):int(remainder/2)+t*(pt.proper_length/4)+pt.proper_length]

                    #vutils.save_image(imgs_temp[0],"./myfigure{}.jpg".format(t))
                

                    #print(imgs.size())

                    batch_size = play_type.cpu().size()[0]

                    play_type_fragment = Variable(play_type, volatile = True, requires_grad = False).cuda()
                    imgs_fragment = Variable(imgs_temp, volatile = True, requires_grad = False).cuda()

                    #print(imgs_temp.size(), imgs.size())

                    output = pt.model(imgs_fragment)
                    play_type_fragment = play_type_fragment.squeeze(1)

                    prediction = output.data.cpu().max(1)[1].numpy()
                    prediction = np.squeeze(prediction)
                    loss = pt.criterion(output, play_type_fragment)

                    play_type_target = play_type_fragment.data.cpu().numpy()
                    train_loss += loss.data.cpu().numpy()[0]

                    mypredictions[int(prediction)] += 1

                prediction_final = mypredictions.index(max(mypredictions))

                print("my prediction in this case is {} for imgs from {} to {}".format(prediction_final, _[0], _[pt.proper_length]))
                #base1, base2 = os.path.basename(_[0][0]),os.path.basename(_[int(remainder/2)+int(((imgs.size()[1]-pt.proper_length)/(pt.proper_length/4)-1)*(pt.proper_length/4)+pt.proper_length)][0])
                #print(base1) # save in this file
                #mylogger.log("{} {} {}".format(int(os.path.splitext(base1)[0]), int(os.path.splitext(base2)[0]), prediction_final, play_type_target[0]))
                #if "2016121110" in _[0][0]:
                #    f_1 = open("./experiments/final_playtype_experiments/2016121110_pred.txt","a")
                #    f_1.write("{} {} {} {}\n".format(int(os.path.splitext(base1)[0]), int(os.path.splitext(base2)[0]), prediction_final, play_type_target[0]))
                #    f_1.close()
                #else:
                #    f_2 = open("./experiments/final_playtype_experiments/2016122411_pred.txt","a")
                #    f_2.write("{} {} {} {}\n".format(int(os.path.splitext(base1)[0]), int(os.path.splitext(base2)[0]), prediction_final, play_type_target[0]))
                #   f_2.close()

                data_size += batch_size
                if prediction_final == play_type_target:
                    correct += 1
                for i in range(batch_size):
                    conf_mat[int(play_type_target)][prediction_final] += 1 

                if random.random() > 0.8:
                    mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction_final, play_type_target, np.sum(prediction_final == play_type_target),int(batch_size)))
            else:
                if batch_idx == len(loader) - 1:
                    continue

                # Break batch into segments of proper length
                # remove the /4 for non overlapping
                remainder = int(imgs.size()[1]-pt.proper_length) % int(pt.proper_length/4)

                mypredictions = [0,0,0]
                for t in range(0,int((imgs.size()[1]-pt.proper_length)/(pt.proper_length/4))):
                    #print(int(imgs.size()[1]/(pt.proper_length/4)), remainder)
                    imgs_temp = imgs[:,int(remainder/2)+t*(pt.proper_length/4):int(remainder/2)+t*(pt.proper_length/4)+pt.proper_length]

                    #vutils.save_image(imgs_temp[0],"./myfigure{}.jpg".format(t))
                

                    #print(imgs.size())

                    batch_size = play_type.cpu().size()[0]

                    play_type_fragment = Variable(play_type, volatile = True, requires_grad = False).cuda()
                    imgs_fragment = Variable(imgs_temp, volatile = True, requires_grad = False).cuda()

                    #print(imgs_temp.size(), imgs.size())

                    output = pt.model(imgs_fragment)
                    play_type_fragment = play_type_fragment.squeeze(1)

                    prediction = output.data.cpu().max(1)[1].numpy()
                    prediction = np.squeeze(prediction)
                    loss = pt.criterion(output, play_type_fragment)

                    play_type_target = play_type_fragment.data.cpu().numpy()
                    train_loss += loss.data.cpu().numpy()[0]

                    mypredictions[int(prediction)] += 1

                prediction_final = mypredictions.index(max(mypredictions))

                print("my prediction in this case is {} for imgs from {} to {}".format(prediction_final, _[0], _[pt.proper_length]))
                base1, base2 = os.path.basename(_[0][0]),os.path.basename(_[int(remainder/2)+int(((imgs.size()[1]-pt.proper_length)/(pt.proper_length/4)-1)*(pt.proper_length/4)+pt.proper_length)][0])
                print(base1) # save in this file
                mylogger.log("{} {} {}".format(int(os.path.splitext(base1)[0]), int(os.path.splitext(base2)[0]), prediction_final, play_type_target[0]))
                if "2016121110" in _[0][0]:
                    f_1 = open("./experiments/final_playtype_experiments/2016121110_34_pred_pred.txt","a")
                    f_1.write("{} {} {} {}\n".format(int(os.path.splitext(base1)[0]), int(os.path.splitext(base2)[0]), prediction_final, play_type_target[0]))
                    f_1.close()
                else:
                    f_2 = open("./experiments/final_playtype_experiments/2016122411_34_pred_pred.txt","a")
                    f_2.write("{} {} {} {}\n".format(int(os.path.splitext(base1)[0]), int(os.path.splitext(base2)[0]), prediction_final, play_type_target[0]))
                    f_2.close()

                data_size += batch_size
                if prediction_final == play_type_target:
                    correct += 1
                for i in range(batch_size):
                    conf_mat[int(play_type_target)][prediction_final] += 1 

                if random.random() > 0.0:
                    mylogger.log('Prediction : {} vs. {} : {} / {}'.format(prediction_final, play_type_target, np.sum(prediction_final == play_type_target),int(batch_size)))
       

        myset = "training"
        if loader == pt.test_acc_loader:
            myset = "testing"
        mylogger.log('{} set, Average loss: {:.4f} Accuracy: {}/{} ({:.0f}%)'.format(myset, train_loss/data_size, correct, data_size,100.0*correct/(data_size))) 
        conf_mat = np.multiply(conf_mat,1.0/data_size)
        mylogger.log('Conf matrix')
        mylogger.log(conf_mat)

def final_expectation(sq):

    sq.model.eval()
    sq.model = sq.model.module
    sq.model.cpu()
    sq.model.eval()
    

    eb.use_eb(True)
    sq.model.eval()

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

def final_expectation_dataset(pt):

    pt.model.eval()
    pt.model = pt.model.module
    pt.model.cpu()
    pt.model.eval()
    print pt.model

    eb.use_eb(True)
    pt.model.eval()

    for batch_idx, (imgs, play_type,_) in enumerate(pt.exciteloader_train):

        counter = 0
        all_examples = 0
        last_video = "-"
        last_batch = 0

        if batch_idx == len(pt.exciteloader_train) - 1:
            continue

        mypredictions = [0,0,0]

        n_image = 0
        for t in range(0,int((imgs.size()[1]-pt.proper_length)/(pt.proper_length))):
            #print(int(imgs.size()[1]/(pt.proper_length/4)), remainder)
            imgs_temp = imgs[:,t*(pt.proper_length):t*(pt.proper_length)+pt.proper_length]
            _temp = _[:][t*(pt.proper_length):t*(pt.proper_length)+pt.proper_length]

            #vutils.save_image(imgs_temp[0],"./myfigure{}.jpg".format(t))
        

            #print(imgs.size())


            batch_size = play_type.cpu().size()[0]

            play_type_fragment = Variable(play_type, volatile = True, requires_grad = False)
            imgs_fragment = Variable(imgs_temp, volatile = True, requires_grad = False)

            #print(imgs_temp.size(), imgs.size())

            output = pt.model(imgs_fragment)
            play_type_fragment = play_type_fragment.squeeze(1)

            prediction = output.data.cpu().max(1)[1].numpy()
            prediction = np.squeeze(prediction)

            play_type_target = play_type_fragment.data.cpu().numpy()

            mypredictions[int(prediction)] += 1
            #print("Done for t=",t," and batch id=",batch_idx," and video ", _[t*(pt.proper_length)])

            layer_top = list(pt.model.modules())[0].decoder_to_classifier[6]
            layer_second = list(pt.model.modules())[0].decoder_to_classifier[4]
            target_layer = list(pt.model.modules())[0].features[1] 
            top_dh = torch.zeros(imgs.size()[0], layer_top.out_features)

            
            top_dh[0,play_type_target[0]] = 1 # ground truth based contrastive signal
            #print(top_dh)

            #prediction_final = mypredictions.index(max(mypredictions))

            if prediction != play_type_target:
                #print("not a match",prediction, play_type_target, int(prediction), int(play_type_target[0]))
                n_image += 1
                continue
            else:
                pass
                #print(int(prediction),"matched",int(play_type_target[0]))

            imgs_temp = Variable(imgs_temp)
            pt.optimizer.zero_grad()
            grad = eb.contrastive_eb(pt.model, imgs_temp, layer_top, layer_second, target=target_layer, top_dh=top_dh)

            print grad.numpy().shape
            grad_ = grad.numpy()[:,:,:,:]
            grad_ = np.mean(grad_,axis=1)

            for j in range(pt.proper_length):
                mytensor = torch.from_numpy(grad_[j]).unsqueeze(0).unsqueeze(0)                
                print(mytensor.size())
                vutils.save_image(mytensor,"./experiments/playtype_EB_final_data/sample_"+str(batch_idx).zfill(4)+"_"+str(j+n_image*pt.proper_length).zfill(4)+"_"+str(int(prediction))+".png", normalize=True)
                copyfile(str(_temp[j][0]), "./experiments/playtype_EB_final_data/sample_orig_"+str(batch_idx).zfill(4)+"_"+str(j+n_image*pt.proper_length).zfill(4)+"_"+str(int(prediction))+".jpg")                

            n_image += 1

        continue

        if prediction_final == play_type_target:
            print("CORRECT!" , prediction_final, play_type_target)

        exit()

        

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


def IoU():
    difference = 0
    iou_all_games = 0
    iou_all_games0 = 0
    iou_all_games1 = 0
    iou_all_games2 = 0

    total_b, total_B = 0,0
    total_C = 0

    for game in ["2016121110","2016122411"]:#["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108"]:
        gt = open("./experiments/final_playtype_experiments/gt/"+game+".txt","r")
        prediction = open("./experiments/final_playtype_experiments/"+game+"_34_pred_pred.txt","r")


        lines_gt = gt.readlines() 
        lines_pd = prediction.readlines()

        gt_list = []
        for i, line in enumerate(lines_gt):
            #if i > len(lines_gt)*0.2:
            #    break

            t = line.split(" ")
            t = (int(t[0])-1, int(t[1])-1, int(t[2].strip("\n")))
            #if t[0] == 0:
            #    continue

            gt_list.append(((t[0],t[2]), (t[1],t[2])))

        pd_list = []
        for line in lines_pd:

            t = line.split(" ")
            t = (int(t[0])-1, int(t[1])-1+9, int(t[2].strip("\n")))
            #if t[0] == 0:
            #    continue

            pd_list.append(((t[0],t[2]), (t[1],t[2])))

        print(pd_list)
        print(gt_list)


        difference += abs(len(pd_list)-len(gt_list))
        print(difference)


        mylogger.log(game+" viewpoint imbalance is "+str((abs(len(pd_list)-len(gt_list)))/3.0))
        #print len(pd_list), len(gt_list)

        mylogger.log("game: "+str(game)+" all classes "+str(IoU_pairs(pd_list, gt_list)))
        IoU, Ilen, Blen, b, tpfp =  IoU_pairs(pd_list, gt_list)
        total_b += b
        total_B += Blen
        total_C += tpfp
        iou_all_games += IoU/Ilen
        mylogger.log("game: "+str(game)+" class 0 "+str(IoU_pairs(pd_list, gt_list,classes=[0])))
        IoU, Ilen, Blen, b, _ =  IoU_pairs(pd_list, gt_list,classes=[0])
        iou_all_games0 += IoU/Ilen
        mylogger.log("game: "+str(game)+" class 1 "+str(IoU_pairs(pd_list, gt_list,classes=[1])))
        IoU, Ilen, Blen, b, _ =  IoU_pairs(pd_list, gt_list,classes=[1])
        iou_all_games1 += IoU/Ilen
        mylogger.log("game: "+str(game)+" class 2"+str(IoU_pairs(pd_list, gt_list,classes=[2])))
        IoU, Ilen, Blen, b, _ =  IoU_pairs(pd_list, gt_list,classes=[2])
        iou_all_games2 += IoU/Ilen
    mylogger.log("all games, all classes, recall is "+str(total_b*1.0/total_B))
    mylogger.log("all games, all classes, mAP is "+str(total_b*1.0/total_C))
    print("difference for folder is ", difference)
    mylogger.log("all games, all classes "+str(iou_all_games/2.0))
    mylogger.log("all games, class 0 "+str(iou_all_games0/2.0))
    mylogger.log("all games, class 1 "+str(iou_all_games1/2.0))
    mylogger.log("all games, class 2 "+str(iou_all_games2/2.0))
    print(str(iou_all_games/2.0))

def main():
    global args, mylogger
     
    args = parser.parse_args() 
    mylogger = MagnificentOracle()
    mylogger.set_log(logfile=args.log)
    mylogger.log("-dotted-line")  

    #ground_truth_files()

    pt = playtype_classifier()
    #print("All done")
    #IoU()
    #train(int(args.epochs), pt)
    #final_expectation(pt)
    #acc_test(1,pt)   # 91% testing acc at 48 epochs for binary, 85% testing acc at 48 epochs for trinary
    final_expectation_dataset(pt)


if __name__ == "__main__":

    main()


