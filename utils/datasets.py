import numpy as np

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.utils.data as data
import PIL
from scipy.misc import imread, imresize

import glob
import os
import csv
import pickle
 
import re
import random
import xml.etree.ElementTree as ET
import csv

from collections import defaultdict
import fnmatch

from PIL import Image
import math

# Dataset preparation
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def compute_mean_std(loader): 

    c_sums = np.array([0,0,0])
    batches = 0
    for batch, (data,label,frame_num, game_id) in enumerate(loader):
        batches += 1
        img_np = data.numpy()
        b,c,h,w = img_np.shape
        c_sums = np.add(np.multiply(np.sum(img_np,(0,2,3)),1.0/(b*h*w)),c_sums)

    c_sums = np.multiply(c_sums,1.0/batches)
    print 'means: ',c_sums
    
    # broadcasting image means
    c_sums = np.expand_dims(c_sums,axis=1)
    c_sums = np.expand_dims(c_sums,axis=2)
    c_sums = np.expand_dims(c_sums,axis=3)
    
    d_sums = np.array([0,0,0])
    batches = 0
    for batch, (data,label, frame_num, game_id) in enumerate(loader):
        
        batches += 1
        img_np = data.numpy()
        img_np = np.swapaxes(img_np,0,1)
        c,b,h,w = img_np.shape
        adds = np.add(img_np,np.multiply(c_sums,-1))
        sums = np.sum(np.square(adds),(1,2,3))
        d_sums = np.add(np.multiply(sums,1.0/(b*h*w)),d_sums)
        d_sums = np.add( np.multiply(np.sum(np.square(np.add(img_np,np.multiply(c_sums,-1))),(1,2,3)),1.0/(b*h*w))  ,d_sums)
    d_sums = np.multiply(d_sums,1.0/batches)
    d_sums = np.sqrt(d_sums)
    print 'standard deviations: ',d_sums
    
    # Update these last, so they do not affect the trainloading transforms
    means = c_sums
    sds = d_sums

class VideoDataset(data.Dataset):

    transform_type = None

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset

    def proper_transform(self):

        normalize = transforms.Normalize(
            mean=[0.3886, 0.4391, 0.3203],
            std=[0.2294, 0.2396, 0.2175]
        )

        normal_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor(),
            normalize
        ])

        
        if self.transform_type == "normal":
            return normal_transform

    def __init__(self, video=None, batch_size=1, frame_skip=5,scale=256,random_crop=[224,224],image_folder=None, use_existing=False):
        
        self.random_crop = random_crop
        self.scale = scale
        self.image_folder = image_folder
        self.video = video

        # Are we sure we want a random cropping for the frame, unlike testing time where it is a center crop?
        self.transform_type = "normal"

        normalize = transforms.Normalize(
            mean=[0.3886, 0.4391, 0.3203],
            std=[0.2294, 0.2396, 0.2175]
        )

    
        self.t = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])


        #self.t = self.proper_transform()

        #making sure that the image folder ends with a / which is needed for the system call        
        image_folder = os.path.join(image_folder, "")
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)

        print("ffmpeg -i %s -vf \"select=not(mod(n\\,%d))\" -vsync vfr " % (video, frame_skip + 1))
        print("ffmpeg -i %s -vf \"select=not(mod(n\\,%d))\" -vsync vfr %s%%08d.png" % (video, frame_skip + 1, image_folder))
        if not use_existing:        
            os.system("ffmpeg -i %s -vf \"select=not(mod(n\\,%d))\" -vsync vfr %s%%08d.png" % (video, frame_skip + 1, image_folder)) 
        
        self.namelist = glob.glob(image_folder+"/*.png")
        self.maxlength = len(self.namelist)
        print "Loaded dataset of length: ",self.maxlength

    def __getitem__(self, index):

        img_name = self.namelist[index]
        #image name is the form: /some/files/here/video_name/00000546.png" so split by files, get number before .png
        frame_num = int(img_name.split("/")[-1].split(".")[0])
        img = PIL.Image.open(img_name)
        img_tensor = self.t(img)

        return img_tensor,frame_num

    def __len__(self):
        return len(self.namelist)

class CustomDataset(data.Dataset):
    
    namelist = None
    path = None
    balanced = False
    subset_type = None
    random_crop = None
    center_crop = None
    scale = None
    transform_type = None

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.3886, 0.4391, 0.3203],
            std=[0.2294, 0.2396, 0.2175]
        )

        compute_mstd_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor() # To test the compute_mean_std method, normalize the dataset to see if the mean and std are 0s and 1s respectively
        ])

        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.transform_type == "mstd":
            return compute_mstd_transform
        elif self.transform_type == "train":
            return train_transform
        elif self.transform_type == "test":
            return test_transform       
    
    def custom_subset(self,dataset_index=0, video_target = None):

        print self.path
        if self.subset_type == None:
            
            self.namelist = glob.glob(self.path+"/**/*.*")
        else:
            if dataset_index == 0:
                training_games = ["aug","2014092112","2014091410","2014091411","2014100600","2014122107","2014090400","2014112305"]
                testing_games = ["2014091408","2014101912"]
            elif dataset_index == 1:
                training_games = ["aug","2014091408","2014092112","2014091410","2014101912","2014091411","2014122107","2014112305"]
                testing_games = ["2014100600","2014090400"]
            elif dataset_index == 2:
                training_games = ["aug","2014091408","2014092112","2014091410","2014101912","2014091411","2014122107","2014112305","2014100600","2014090400"]
                testing_games = []
            elif dataset_index == 3:
                training_games = ["2014091408"]
                testing_games = []                            
            elif dataset_index == 4:
                training_games = []
                testing_games = ["2016090800"]
            elif dataset_index == 5:
                training_games = []
                testing_games = ["2016091100"]
            elif dataset_index == 6:
                training_games = []
                testing_games = ["2016092500"]
            elif dataset_index == 7:
                training_games = []
                testing_games = ["2016090800","2016091100","2016092500"]
            elif dataset_index == 8:
                training_games =["aug","2014091408","2014092112","2014091410","2014101912","2014091411","2014122107","2014112305","2014100600","2014090400","2016112710","2016091801","2016100201","2016091808"]
                testing_games = ["2016091100"]
            elif dataset_index == 9:
                training_games =["aug","2014091408","2014092112","2014091410","2014101912","2014091411","2014122107","2014112305","2014100600","2014090400","2016112710","2016091801","2016100201","2016091808"]
                testing_games = ["2016092500"]
            elif dataset_index == 10:
                training_games =["aug","2014091408","2014092112","2014091410","2014101912","2014091411","2014122107","2014112305","2014100600","2014090400","2016112710","2016091801","2016100201","2016091808"]
                testing_games = ["2016091100","2016092500"]
            elif dataset_index == 11:
                training_games = []
                testing_games = ["2016091100","2016092500","2016112710","2016091801","2016100201","2016091808"]
            elif dataset_index == 25: # conformal prediction dataset
                training_games = ["aug","2014090400","2014091408","2014091400","2014092112","2014100600","2014112305","2014122107","2016090800","2016091100","2016091801","2016092500","2016100201","2016100202"]
                testing_games = ["2014101912","2014091411","2016091808","2016112710"]
            elif dataset_index == 50:
                training_games = []
                testing_games = ["2016091808"]
            elif dataset_index == 75:
                training_games = []
                testing_games = ["2016112710"]
            
            if video_target != None:
                training_games = []
                testing_games = [video_target]
            
            all_images = glob.glob(self.path+"/**/*.*")
            
            for img_name in all_images:
                basename = os.path.basename(img_name)
                name = basename.split(".")[0] 
                game_id, frame, label = name.split("_")
                label = int(label)
                
                mytype = "normal"
                if "aug" in img_name:
                    mytype = "aug"
                
                in_dataset = ((self.subset_type == "training" or self.subset_type == "mstd") and \
                (game_id in training_games or mytype in training_games)) or \
                (self.subset_type == "testing" and (game_id in testing_games or mytype in testing_games))
                 
                if (self.subset_type == "training" and self.balanced):
                    if (in_dataset):                    
                        self.namelist[label].append(img_name)
                else:
                    if (in_dataset):
                        self.namelist.append(img_name)
                 
    def __init__(self, path=None, video_target = None, scale=256, random_crop=[224,224], center_crop=[224,224], balanced=False, subset_type=None, dataset_index=0, file_type=".jpg"):
        
        self.path = path
        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.balanced = balanced
        self.subset_type = subset_type
        self.file_type = file_type
        self.video_target = video_target

        if subset_type == "training":
            self.transform_type = "train"
        elif subset_type == "testing" or None:
            self.transform_type = "test"
        elif subset_type == "mstd":
            self.transform_type = "mstd"            

        if (subset_type == "training" and balanced):
            self.namelist = defaultdict(list)
        else:
            self.namelist = []

        self.custom_subset(dataset_index, video_target = video_target)

        if balanced==True:
            self.namelist = balanced_subset()
        
        if (not balanced or not self.subset_type == "training"): 
            self.namelist.sort(key=natural_keys)

        self.maxlength = len(self.namelist)
        print "Loaded ",subset_type," dataset of length: ",self.maxlength

    '''
    DEPRECATED

    # img_name is assumed to be the complete path
    def extract_xml_label(self,img_name):
        
        xml_name = "."+img_name.split(".")[1]+".xml"
        tree = ET.parse(xml_name)
        root = tree.getroot()
        label_int = -1
        frame_int = -1
        for child in root:
            if child.tag == "label":
                label_int = int(child.text)        
            if child.tag == "frame":
                frame_int = int(child.text)
        return label_int, frame_int

    # img_name is assumed to be the complete path
    def get_filename_codes(self,img_name):
        img_name_base = os.path.basename(img_name)
        codes = img_name_base.split("_")
        start = 1

        mytype = "normal"
        if "scoreboard" in img_name_base.split("_")[0]:
            start += 1
            mytype = "scoreboard"

        game_code = codes[start]
        frame = codes[start+1]
        label = codes[start+2].split(".")[0] 
        return (mytype, game_code, frame, label)
    '''
    
    def __getitem__(self, index):

        if (self.subset_type == "training" and self.balanced):
            rand_label = random.choice(self.namelist.keys())            
            img_name = random.choice(self.namelist[rand_label])
        else:
            img_name = self.namelist[index]

        img = PIL.Image.open(img_name)
        t = self.proper_transform()
        img_tensor = t(img)
        basename = os.path.basename(img_name)
        name = basename.split(".")[0]
        game_id, frame, label = name.split("_") 
        label = int(label)
        label = np.array([label])

        return img_tensor, label, frame, game_id

    def __len__(self):
        
        if (self.subset_type == "training" and self.balanced):
            s = 0
            for label in self.namelist.keys():
                s += len(self.namelist[label])
            return s
        else:
            return len(self.namelist)

def extract_xmls(target_folder=None):

    images = glob.glob(target_folder+"/*"+self.file_type)
    for image in images:
        image_base = os.path.basename(image)
        codes = image_base.split("_")
        start = 1

        mytype = "normal"
        if "scoreboard" in image.split("_")[0]:
            start += 1
            mytype = "scoreboard"

        game_code = codes[start]
        frame = codes[start+1]
        label = codes[start+2].split(".")[0]
        root = ET.Element("video_annotation")

        ET.SubElement(root, "type").text = mytype
        ET.SubElement(root, "game_code").text = game_code
        ET.SubElement(root, "frame").text = frame
        ET.SubElement(root, "label").text = label

        tree = ET.ElementTree(root)
        initial_name = "/images_"
        if mytype == "scoreboard":
            initial_name = "/scoreboard_images_"
        
        tree.write(target_folder+initial_name+codes[start]+"_"+codes[start+1]+"_"+codes[start+2].split(".")[0]+".xml")

class CustomDatasetPlays(data.Dataset):
    
    namelist = None
    path = None
    balanced = False
    subset_type = None
    random_crop = None
    center_crop = None
    scale = None
    transform_type = None
    dataset_index = -1

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.3886, 0.4391, 0.3203],
            std=[0.2294, 0.2396, 0.2175]
        )

        compute_mstd_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor() # To test the compute_mean_std method, normalize the dataset to see if the mean and stds are 0 and 1s respectively
        ])

        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.transform_type == "mstd":
            return compute_mstd_transform
        elif self.transform_type == "train":
            return train_transform
        elif self.transform_type == "test":
            return test_transform       

    def mapping(self,entire_dataset_dict=0, data_split_mode = 0):

        if data_split_mode == 0:

            trainable = 0
            true_index = 0
            for game_code in entire_dataset_dict.keys():
                for play_index in entire_dataset_dict[game_code].keys():

                    # Decide if trainable based on true_index, game_code or play_index or other
                    # -1 testing, +1 training, 0 nothing
                    #playtype = entire_dataset_dict[game_code][play_index][0]
                    #if playtype in ["PASS","RUSH"]:
                    #if game_code == "2016091100": # 90-10% split
                    #    trainable = -1
                    #else:
                    #    trainable = 1
                    play = entire_dataset_dict[game_code][play_index][0]
                    trainable = 0

                    # randomize to obtain random playtypes from all videos
                    untrainable_indices = [10,20,30,40,50,60,70,80,90]

                    if play == "PASS" or play == "RUSH":
                        if true_index % 100 in untrainable_indices:
                            trainable = -1
                        else:
                            trainable = 1

                    entire_dataset_dict[game_code][play_index] = trainable
                    true_index += 1

        return entire_dataset_dict
    
    def custom_subset(self,dataset_index=0):

        all_images = []
        for root, dirnames, filenames in os.walk(self.path):
            for filename in fnmatch.filter(filenames, '*.jpg'):
                all_images.append(os.path.join(root, filename))

        play_annotations = {}

        if not os.path.isfile("./annotations/precomputed_csv_plays/play_annotations.pkl"): 
            
            self.proper_csv_2016 = {}
            with open('./annotations/csv_files/proper_2016.csv','rb') as f:
                spamreader = csv.reader(f, delimiter=',', quotechar='"')
                prev_game_id = -1
                for row in spamreader:
                    game_id = row[0]
                    if game_id != prev_game_id:
                        prev_game_id = game_id
                        play_index = 1

                    active = (row[5] != "")
                    defense_team = row[6]
                    # offense has to be non-empty
                    play_type = row[17]

                    if active:
                        if game_id not in self.proper_csv_2016.keys():
                            self.proper_csv_2016[game_id] = {}
                        if play_index not in self.proper_csv_2016[game_id]:
                            self.proper_csv_2016[game_id][play_index] = None

                        self.proper_csv_2016[game_id][play_index] = play_type
                        play_index += 1

            play_annotations_dir = "./annotations/full_viewpoint_annotations_2016"
            annotated_files = glob.glob(play_annotations_dir+"/*_c.txt")

            for file in annotated_files:
                play_index = 1
                with open(file,"rb") as f:
                    game_id = os.path.basename(file).split("_c.txt")[0]

                    lines = f.readlines()
                    for line in lines:
                        line = line.strip("\n")
                        s = line.split(" ")

                        if s[1] == '0':
                            a = int(s[0])
                        elif s[1] == '1':
                            b = int(s[0])

                            if game_id in self.proper_csv_2016.keys() and play_index in self.proper_csv_2016[game_id].keys():

                                play_type = self.proper_csv_2016[game_id][play_index]

                                for d in range(a,b): # CHECK LIMITS
                                    d_frame = str(d).zfill(8)

                                    if game_id not in play_annotations.keys():
                                        play_annotations[game_id] = {}
                                    
                                    play_annotations[game_id][d_frame] = [play_index, play_type]

                            play_index += 1

            pickle.dump(play_annotations, open("./annotations/precomputed_csv_plays/play_annotations.pkl","wb"))
        else:
            play_annotations = pickle.load(open("./annotations/precomputed_csv_plays/play_annotations.pkl","rb"))

        self.play_types = []
        for game_id in play_annotations.keys():
            for d_frame in play_annotations[game_id].keys():
                if play_annotations[game_id][d_frame][1] not in self.play_types:
                    self.play_types.append(play_annotations[game_id][d_frame][1])

        # Evaluate how many plays exist in the entire dataset

        entire_dataset_dict = {}

        for img_name in all_images:

            img_number = os.path.basename(img_name).split(".")[0]
            parent_directory = os.path.abspath(os.path.join(img_name, os.pardir))
            game_code = parent_directory.split("/")[-1]
            grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir)).split("/")[-1]
            type_code = grandparent_directory.split("/")[-1]
            try:
                p = play_annotations[game_code][img_number]
                play_index, play_type = p[0], p[1]
            except:
                continue
            try:
                _ = entire_dataset_dict[game_code]
            except:
                entire_dataset_dict[game_code] = {}

            try:
                _ = entire_dataset_dict[game_code][play_index]
            except:
                entire_dataset_dict[game_code][play_index] = [play_type, []]

            entire_dataset_dict[game_code][play_index][1].append(img_name)

        self.trainable = self.mapping(entire_dataset_dict)
        #self.evaluation_trainer = [0 for i in range(len(entire_dataset_list))]

        for img_name in all_images:
            
            # extract information from filename, to determine subset inclusion based on the <dataset_index>
            img_number = os.path.basename(img_name).split(".")[0]
            parent_directory = os.path.abspath(os.path.join(img_name, os.pardir))
            game_code = parent_directory.split("/")[-1]
            grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir)).split("/")[-1]
            type_code = grandparent_directory.split("/")[-1]


            '''if type_code == "scoreboards":
                game_indices = {"training": scoreboards, "testing":[], "mstd": scoreboards}
            else: 
                game_indices = {"training":training_games, "testing":testing_games,"mstd":training_games}'''
                
            '''game_index = game_indices[self.subset_type]
            if game_index == []:
                continue'''

            try:
                p = play_annotations[game_code][img_number]
                play_index, play_type = p[0], p[1]
            except:
                continue

            indataset = (self.trainable[game_code][play_index] == 1 and (self.subset_type == 'training' or self.subset_type == 'mstd')) or (self.trainable[game_code][play_index] == -1 and (self.subset_type == 'testing'))

            if indataset:

                try:
                    _ = self.namelist[game_code]
                except:
                    self.namelist[game_code] = {}

                try:
                    _ = self.namelist[game_code][play_index]
                except:
                    self.namelist[game_code][play_index] = [play_type, []]

                self.namelist[game_code][play_index][1].append(img_name)

        # Map indices to all games and plays that we have obtained to be able to use them in the classifier
        index = 0
        for game_code in self.namelist.keys():
            for play_index in self.namelist[game_code].keys():
                self.indexed_namelist[index] = [game_code, play_index]
                index += 1
                 
    def __init__(self, path=None, scale=256, random_crop=[224,224], center_crop=[224,224], subset_type=None, dataset_index=0, categories="chunked", retrieve_images=True, frame_select = 12, dist = 0):
        
        self.path = path
        self.subset_type = subset_type
        self.dataset_index = dataset_index

        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.categories = categories
        self.retrieve_images = retrieve_images
        self.frame_select = frame_select

        self.indexed_namelist = {}
        self.namelist = {}
        self.dist = dist

        if subset_type == "training":
            self.transform_type = "train"
        elif subset_type == "testing" or None:
            self.transform_type = "test"
        elif subset_type == "mstd":
            self.transform_type = "mstd"            

        self.custom_subset(dataset_index)

        self.maxlength  = 0
        for game_id in self.namelist.keys():
            for play_index in self.namelist[game_id].keys():
                self.maxlength += 1

    def interval_distribution(self,window=None,full=None):
        fs = self.frame_select

        mean = (window[1]+window[0])/2.0
        std = (window[1]-window[0])/4.0 # 2 sigmas

        k = []
        for i in range(fs):
            ro = -1
            while ro <= 0 or ro >= full or ro in k:
                ro = int(np.random.normal(mean,std,1)[0])

            k.append(ro)

        #print k 
        return k

    def __getitem__(self, index):

        # Without frame_skip, there are 60.87 frames per play
        game_code, play_index = self.indexed_namelist[index]
        play_type, imgs_in_play = self.namelist[game_code][play_index]

        imgs = None

        # gaussian
        # window or entire

        full = len(imgs_in_play)
        if self.dist == 0:
            mydist = self.interval_distribution(window=[0,full], full=full)
        elif self.dist == 1:
            mydist = self.interval_distribution(window=[0,full/2], full=full)
        elif self.dist == 2:
            mydist = self.interval_distribution(window=[full/2,full], full=full)
        elif self.dist == 3:
            mydist = self.interval_distribution(window=[full/4,3*full/4], full=full)
        elif self.dist == 4:
            mydist = self.interval_distribution(window=[0,3*full/4], full=full)
        elif self.dist == 5:
            mydist = self.interval_distribution(window=[full/4,full], full=full)
        if self.dist != 6:
            mydist.sort()

        if self.retrieve_images:
            imgs_raw = []
            t = self.proper_transform()

            for idx, img in enumerate(imgs_in_play):
                if self.dist != 6:
                    if (idx not in mydist):
                        continue
                else:
                    if idx % (full/self.frame_select) != 0:
                        continue
                    if len(imgs_raw) == self.frame_select:
                        break 

                img_tensor = t(PIL.Image.open(img))
                img_tensor.unsqueeze_(0)
                imgs_raw.append(img_tensor)

            imgs = torch.cat(imgs_raw,0)
        else:
            imgs = torch.FloatTensor([len(imgs_in_play)])

        # mapping for 0 -> "other", 1 -> "pass", 2 -> "rush"
        class_number = self.play_types.index(play_type)

        play_classes = ["2","3","-"]
        if self.categories == "chunked":
            if class_number != 2 and class_number != 3:
                class_number = 0 # other
            elif class_number == 2:
                class_number = 1 # 2
            else:
                class_number = 2 # 3     

        if self.categories == "two":
            if class_number == 2:
                class_number = 0
            elif class_number == 3:
                class_number = 1     

        return imgs, np.array([class_number])

    def __len__(self):
        
        s = 0
        for game_id in self.namelist.keys():
            for play_index in self.namelist[game_id].keys():
                s += 1
        return s

class CustomDatasetViewpoint(data.Dataset):
    
    namelist = None
    path = None
    balanced = False
    subset_type = None
    random_crop = None
    center_crop = None
    scale = None
    transform_type = None

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset
    def compute_mean_std(self,loader): 

        c_sums = np.array([0,0,0])
        batches = 0
        for batch, (data,label) in enumerate(loader):
            if batch % 25 == 0:
                print batch 
            if batch > 250:
                break
            batches += 1
            img_np = data.numpy()
            b,c,h,w = img_np.shape
            c_sums = np.add(np.multiply(np.sum(img_np,(0,2,3)),1.0/(b*h*w)),c_sums)

        c_sums = np.multiply(c_sums,1.0/batches)
        print 'means: ',c_sums

        # broadcasting image means
        c_sums = np.expand_dims(c_sums,axis=1)
        c_sums = np.expand_dims(c_sums,axis=2)
        c_sums = np.expand_dims(c_sums,axis=3)

        d_sums = np.array([0,0,0])
        batches = 0
        for batch, (data,label) in enumerate(loader):
            if batch % 25 == 0:
                print batch
            if batch > 75:
                break

            batches += 1
            img_np = data.numpy()
            img_np = np.swapaxes(img_np,0,1)
            c,b,h,w = img_np.shape
            adds = np.add(img_np,np.multiply(c_sums,-1))
            sums = np.sum(np.square(adds),(1,2,3))
            d_sums = np.add(np.multiply(sums,1.0/(b*h*w)),d_sums)
            d_sums = np.add( np.multiply(np.sum(np.square(np.add(img_np,np.multiply(c_sums,-1))),(1,2,3)),1.0/(b*h*w))  ,d_sums)

        d_sums = np.multiply(d_sums,1.0/batches)
        d_sums = np.sqrt(d_sums)
        print 'standard deviations: ',d_sums

        # Update these last, so they do not affect the trainloading transforms
        means = c_sums
        sds = d_sums

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.411, 0.493, 0.309],
            std=[0.170, 0.178, 0.171]
        )

        compute_mstd_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor(), # To test the compute_mean_std method, normalize the dataset to see if the mean and std are 0s and 1s respectively
        ])

        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.transform_type == "mstd":
            return compute_mstd_transform
        elif self.transform_type == "train":
            return train_transform
        elif self.transform_type == "test":
            return test_transform       
    
    def custom_subset(self,dataset_index=0, video_target = None):

        if dataset_index == 0:
            training_games =["2014091408","2014092112","2014091410","2014101912","2014091411","2014122107","2014112305","2014100600","2014090400","2016090800","2016100202","2016112710","2016091801","2016100201","2016091808"]
            testing_games = ["2016091100","2016092500"]

        all_images = glob.glob(self.path+"/**/*.*")
    
        skip_counter = 0
        subset_counter = 0
        subset_activate = False
        for img_name in all_images:
            basename = os.path.basename(img_name)
            name = basename.split(".")[0]
            game_id, frame, label = name.split("_")
            label = int(label)
            
            # mstd should never see testing data
            in_dataset = ((self.subset_type == "training" or self.subset_type == "mstd") and (game_id in training_games)) or \
            (self.subset_type == "testing" and (game_id in testing_games))

            '''
            if in_dataset and "2016" in game_id: # balancing the dataset between 2016 and 2014
                if skip_counter % 5 != 0:
                    in_dataset = False
                skip_counter += 1
            '''

            if subset_activate:
                if in_dataset:
                    if subset_counter % 30 != 0:
                        in_dataset = False
                    subset_counter += 1
             
            if (in_dataset and (label != 2 or self.viewpoints == 3)):
                self.namelist.append((img_name,label))


    def __init__(self, path=None, scale=256, random_crop=[224,224], center_crop=[224,224], subset_type=None, dataset_index=0, file_type=".jpg",retrieve_images=True, viewpoints=3):
        
        self.viewpoints = viewpoints
        self.path = path
        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.subset_type = subset_type
        self.file_type = file_type
        self.retrieve_images = retrieve_images


        if subset_type == "training":
            self.transform_type = "train"
        elif subset_type == "testing" or None:
            self.transform_type = "test"
        elif subset_type == "mstd":
            self.transform_type = "mstd"

        self.t = self.proper_transform()          
        
        self.namelist = []
        self.custom_subset(dataset_index)
        self.namelist.sort(key=lambda x:natural_keys(x[0]))
    
    def __getitem__(self, index):

        if self.retrieve_images:
            img_name = self.namelist[index][0]
            img = PIL.Image.open(img_name)
            img_tensor = self.t(img)
        else:
            img_tensor = torch.FloatTensor([1])
        img_tensor
        label = np.array([int(self.namelist[index][1])])

        return img_tensor, label

    def __len__(self):
        return len(self.namelist)

class CustomMasterDatasetPlays(data.Dataset):
    
    namelist = None
    path = None
    balanced = False
    subset_type = None
    random_crop = None
    center_crop = None
    scale = None
    transform_type = None
    dataset_index = -1

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.3886, 0.4391, 0.3203],
            std=[0.2294, 0.2396, 0.2175]
        )

        compute_mstd_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor() # To test the compute_mean_std method, normalize the dataset to see if the mean and stds are 0 and 1s respectively
        ])

        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.transform_type == "mstd":
            return compute_mstd_transform
        elif self.transform_type == "train" and self.excitation_type != "excite":
            return train_transform
        elif self.transform_type == "test":
            return test_transform       
        elif self.transform_type == "train" and self.excitation_type == "excite":
            return test_transform

    def mapping(self,entire_dataset_dict=0, data_split_mode = 0):

        if data_split_mode == 0:

            trainable = 0
            true_index = 0
            for game_code in entire_dataset_dict.keys():
                for play_index in entire_dataset_dict[game_code].keys():

                    # Decide if trainable based on true_index, game_code or play_index or other
                    # -1 testing, +1 training, 0 nothing
                    #playtype = entire_dataset_dict[game_code][play_index][0]
                    #if playtype in ["PASS","RUSH"]:
                    #if game_code == "2016091100": # 90-10% split
                    #    trainable = -1
                    #else:
                    #    trainable = 1
                    play = entire_dataset_dict[game_code][play_index][0]
                    trainable = 0

                    # randomize to obtain random playtypes from all videos
                    untrainable_indices = [10,20,30,40,50,60,70,80,90]

                    #if play == "PASS" or play == "RUSH":
                    if true_index % 100 in untrainable_indices:
                        trainable = -1
                    else:
                        trainable = 1#
                    #if game_code in ["2016091100"] and true_index % 100 in untrainable_indices: # changeable
                    #    trainable = -1
                    #    trainable = -1
                    #else:
                    #    trainable = 1

                    entire_dataset_dict[game_code][play_index] = trainable
                    true_index += 1

        return entire_dataset_dict
    
    def custom_subset(self,dataset_index=0):

        all_images = []
        for root, dirnames, filenames in os.walk(self.path):
            for filename in fnmatch.filter(filenames, '*.jpg'):
                all_images.append(os.path.join(root, filename))

        play_annotations = {}

        if not os.path.isfile("./annotations/precomputed_csv_plays/play_annotations_master_other.pkl"):
            self.proper_csv_2016 = {}
            with open('./annotations/csv_files/proper_2016.csv','rb') as f:
                spamreader = csv.reader(f, delimiter=',', quotechar='"')
                prev_game_id = -1
                for row in spamreader:
                    game_id = row[0]
                    if game_id != prev_game_id:
                        prev_game_id = game_id
                        play_index = 1

                    active = (row[5] != "")
                    defense_team = row[6]
                    # offense has to be non-empty
                    play_type = row[17]

                    if active:
                        if game_id not in self.proper_csv_2016.keys():
                            self.proper_csv_2016[game_id] = {}
                        if play_index not in self.proper_csv_2016[game_id]:
                            self.proper_csv_2016[game_id][play_index] = None

                        self.proper_csv_2016[game_id][play_index] = play_type
                        play_index += 1

            play_annotations_dir = "./annotations/full_viewpoint_annotations_2016"
            annotated_files = glob.glob(play_annotations_dir+"/*_c.txt")

            for file in annotated_files:
                play_index = 1
                with open(file,"rb") as f:

                    First_score = True 
                    game_id = os.path.basename(file).split("_c.txt")[0]

                    lines = f.readlines()
                    for line in lines:
                        line = line.strip("\n")
                        s = line.split(" ")

                        if s[1] == '0':
                            a = int(s[0])
                        elif s[1] == '2' and First_score:
                            First_score = False
                        elif s[1] == '2':
                            b = int(s[0])

                            if game_id in self.proper_csv_2016.keys() and play_index in self.proper_csv_2016[game_id].keys():

                                play_type = self.proper_csv_2016[game_id][play_index]

                                for d in range(a,b): # CHECK LIMITS
                                    d_frame = str(d).zfill(8)

                                    if game_id not in play_annotations.keys():
                                        play_annotations[game_id] = {}
                                    play_annotations[game_id][d_frame] = [play_index, play_type]


                            play_index += 1

            pickle.dump(play_annotations, open("./annotations/precomputed_csv_plays/play_annotations_master_other.pkl","wb"))
        else:
            play_annotations = pickle.load(open("./annotations/precomputed_csv_plays/play_annotations_master_other.pkl","rb"))

        self.play_types = []
        for game_id in play_annotations.keys():
            for d_frame in play_annotations[game_id].keys():
                if play_annotations[game_id][d_frame][1] not in self.play_types:
                    self.play_types.append(play_annotations[game_id][d_frame][1])

        # Evaluate how many plays exist in the entire dataset

        entire_dataset_dict = {}

        for img_name in all_images:

            img_number = os.path.basename(img_name).split(".")[0]
            parent_directory = os.path.abspath(os.path.join(img_name, os.pardir))
            game_code = parent_directory.split("/")[-1]
            grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir)).split("/")[-1]
            type_code = grandparent_directory.split("/")[-1]
            try:
                p = play_annotations[game_code][img_number]
                play_index, play_type = p[0], p[1]
            except:
                continue
            try:
                _ = entire_dataset_dict[game_code]
            except:
                entire_dataset_dict[game_code] = {}

            try:
                _ = entire_dataset_dict[game_code][play_index]
            except:
                entire_dataset_dict[game_code][play_index] = [play_type, []]

            entire_dataset_dict[game_code][play_index][1].append(img_name)

        self.trainable = self.mapping(entire_dataset_dict)
        #self.evaluation_trainer = [0 for i in range(len(entire_dataset_list))]

        for img_name in all_images:
            
            # extract information from filename, to determine subset inclusion based on the <dataset_index>
            img_number = os.path.basename(img_name).split(".")[0]
            parent_directory = os.path.abspath(os.path.join(img_name, os.pardir))
            game_code = parent_directory.split("/")[-1]
            grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir)).split("/")[-1]
            type_code = grandparent_directory.split("/")[-1]

            try:
                p = play_annotations[game_code][img_number]
                play_index, play_type = p[0], p[1]
            except:
                continue

            indataset = (self.trainable[game_code][play_index] == 1 and (self.subset_type == 'training' or self.subset_type == 'mstd')) or (self.trainable[game_code][play_index] == -1 and (self.subset_type == 'testing'))

            if indataset:

                try:
                    _ = self.namelist[game_code]
                except:
                    self.namelist[game_code] = {}

                try:
                    _ = self.namelist[game_code][play_index]
                except:
                    self.namelist[game_code][play_index] = [play_type, []]

                self.namelist[game_code][play_index][1].append(img_name)

        # Remove all entries in the namelist that do not contain frame_select images
        removals = []
        for game in self.namelist.keys():
            for play_index in self.namelist[game]:
                if len(self.namelist[game][play_index][1]) < self.frame_select:
                    removals.append((game, play_index))
        for removal in removals:
            del self.namelist[removal[0]][removal[1]]

        # Map indices to all games and plays that we have obtained to be able to use them in the classifier
        index = 0
        for game_code in self.namelist.keys():
            for play_index in self.namelist[game_code].keys():
                self.indexed_namelist[index] = [game_code, play_index]
                index += 1
                 
    def __init__(self, path=None, scale=256, random_crop=[224,224], center_crop=[224,224], subset_type=None, dataset_index=0, categories="two", retrieve_images=True, frame_select = 16):
        
        self.path = path
        self.subset_type = subset_type
        if self.subset_type == "exciting":
            self.subset_type = "training"
        self.dataset_index = dataset_index

        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.categories = categories
        self.retrieve_images = retrieve_images
        self.frame_select = frame_select

        self.indexed_namelist = {}
        self.namelist = {}

        self.dist = 6

        self.excitation_type = "no excite"
        if subset_type == "training" or subset_type == "exciting":
            self.transform_type = "train"
            if subset_type == "exciting":
                self.excitation_type = "excite"
        elif subset_type == "testing" or None:
            self.transform_type = "test"
        elif subset_type == "mstd":
            self.transform_type = "mstd"            

        self.custom_subset(dataset_index)

        self.maxlength  = 0
        for game_id in self.namelist.keys():
            for play_index in self.namelist[game_id].keys():
                self.maxlength += 1

    def interval_distribution(self,window=None,full=None):
        fs = self.frame_select

        mean = (window[1]+window[0])/2.0
        std = (window[1]-window[0])/4.0 # 2 sigmas

        k = []
        for i in range(fs):
            ro = -1
            while ro <= 0 or ro >= full or ro in k:
                ro = int(np.random.normal(mean,std,1)[0])

            k.append(ro)

        #print k 
        return k

    def __getitem__(self, index, dist=6):

        # Without frame_skip, there are 60.87 frames per play (without replay)
        game_code, play_index = self.indexed_namelist[index]
        play_type, imgs_in_play = self.namelist[game_code][play_index]

        imgs = None

        # gaussian
        # window or entire

        full = len(imgs_in_play)

        if self.dist == 0:
            mydist = self.interval_distribution(window=[0,full], full=full)
        elif self.dist == 1:
            mydist = self.interval_distribution(window=[0,full/2], full=full)
        elif self.dist == 2:
            mydist = self.interval_distribution(window=[full/2,full], full=full)
        elif self.dist == 3:
            mydist = self.interval_distribution(window=[full/4,3*full/4], full=full)
        elif self.dist == 4:
            mydist = self.interval_distribution(window=[0,3*full/4], full=full)
        elif self.dist == 5:
            mydist = self.interval_distribution(window=[full/4,full], full=full)
        if self.dist != 6:
            mydist.sort()

        if self.retrieve_images and full >= self.frame_select:
            imgs_raw = []
            t = self.proper_transform()

            for idx, img in enumerate(imgs_in_play):
                if self.dist != 6:
                    if (idx not in mydist):
                        continue
                else:
                    if idx % (full/self.frame_select) != 0:
                        continue
                    if len(imgs_raw) == self.frame_select:
                        break 

                img_tensor = t(PIL.Image.open(img))
                img_tensor.unsqueeze_(0)
                imgs_raw.append(img_tensor)

            imgs = torch.cat(imgs_raw,0)
        else:
            imgs = torch.FloatTensor([len(imgs_in_play)])

        # mapping for 0 -> "other", 1 -> "pass", 2 -> "rush"
        class_number = self.play_types.index(play_type)

        play_classes = ["2","3","-"]
        # these change if you recalculate the precomputed csv file
        if self.categories == "chunked":
            if class_number != 1 and class_number != 0:
                class_number = 0 # other
            elif class_number == 1:
                class_number = 2 # 2
            else:
                class_number = 1 # 3     

        if self.categories == "two":
            if class_number == 1:
                class_number = 0
            elif class_number == 3:
                class_number = 1     

        return imgs, np.array([class_number])

    def __len__(self):
        
        s = 0
        for game_id in self.namelist.keys():
            for play_index in self.namelist[game_id].keys():
                s += 1
        return s

class CustomDatasetViewpointRaw(data.Dataset):
    
    namelist = None
    path = None
    balanced = False
    subset_type = None
    random_crop = None
    center_crop = None
    scale = None
    transform_type = None

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset
    def compute_mean_std(self,loader): 

        c_sums = np.array([0,0,0])
        batches = 0
        for batch, (data,label) in enumerate(loader):
            if batch % 25 == 0:
                print batch 
            if batch > 250:
                break
            batches += 1
            img_np = data.numpy()
            b,c,h,w = img_np.shape
            c_sums = np.add(np.multiply(np.sum(img_np,(0,2,3)),1.0/(b*h*w)),c_sums)

        c_sums = np.multiply(c_sums,1.0/batches)
        print 'means: ',c_sums

        # broadcasting image means
        c_sums = np.expand_dims(c_sums,axis=1)
        c_sums = np.expand_dims(c_sums,axis=2)
        c_sums = np.expand_dims(c_sums,axis=3)

        d_sums = np.array([0,0,0])
        batches = 0
        for batch, (data,label) in enumerate(loader):
            if batch % 25 == 0:
                print batch
            if batch > 75:
                break

            batches += 1
            img_np = data.numpy()
            img_np = np.swapaxes(img_np,0,1)
            c,b,h,w = img_np.shape
            adds = np.add(img_np,np.multiply(c_sums,-1))
            sums = np.sum(np.square(adds),(1,2,3))
            d_sums = np.add(np.multiply(sums,1.0/(b*h*w)),d_sums)
            d_sums = np.add( np.multiply(np.sum(np.square(np.add(img_np,np.multiply(c_sums,-1))),(1,2,3)),1.0/(b*h*w))  ,d_sums)

        d_sums = np.multiply(d_sums,1.0/batches)
        d_sums = np.sqrt(d_sums)
        print 'standard deviations: ',d_sums

        # Update these last, so they do not affect the trainloading transforms
        means = c_sums
        sds = d_sums

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.40694026 , 0.47509631 , 0.26367791],
            std=[ 0.14004936 , 0.15746613 , 0.1310064]
        )

        compute_mstd_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor(), # To test the compute_mean_std method, normalize the dataset to see if the mean and std are 0s and 1s respectively
        ])

        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.transform_type == "mstd":
            return compute_mstd_transform
        elif self.transform_type == "train":
            return train_transform
        elif self.transform_type == "test":
            return test_transform       
    
    def custom_subset(self,dataset_index=0, subset_type="training"):

        if dataset_index == 0:
            testing_games = ["2016091100"]
        if dataset_index == -1:
            all_games = ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108","2016121110","2016122411"]
            training_games = ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108"]
            testing_games = ["2016121110","2016122411"]

        all_images = glob.glob(self.path+"/**/*.jpg")
    
        skip_counter = 0
        subset_counter = 0
        subset_activate = False

        play_annotations_dir = "./annotations/full_viewpoint_annotations_2016"
        annotated_files = glob.glob(play_annotations_dir+"/*_c.txt")

        labels_entire = {}
        games = testing_games
        if subset_type == "training":
            games = training_games

        for game in games:
            labels_entire[game] = []

        for file in annotated_files:
            old = -1
            a = 1
            with open(file,"rb") as f:

                First_score = True 
                game_id = os.path.basename(file).split("_c.txt")[0]
                if game_id in games:

                    lines = f.readlines()
                    for line in lines:
                        line = line.strip("\n")
                        s = line.split(" ")

                        if old == -1:
                            a = int(s[0])
                            old = int(s[1])
                        elif s[1] != old:
                            for i in range(a,int(s[0])):
                                labels_entire[game_id].append(int(old))
                            old = int(s[1])
                            a = int(s[0]) 

                    for i in range(a,a+100): # since we have no info for where it stops
                        labels_entire[game_id].append(s[1])

        for game in games:
            i = 0
            for img_name in all_images: # make sure the images are sorted
                if game in img_name:
                    label = int(labels_entire[game][i])
                    i += 1
                    self.namelist.append((img_name,label))

    def __init__(self, path=None, scale=256, random_crop=[224,224], center_crop=[224,224], subset_type=None, dataset_index=0, file_type=".jpg",retrieve_images=True, viewpoints=3):
        
        self.viewpoints = viewpoints
        self.path = path
        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.subset_type = subset_type
        self.file_type = file_type
        self.retrieve_images = retrieve_images


        if subset_type == "training":
            self.transform_type = "train"
        elif subset_type == "testing" or None:
            self.transform_type = "test"
        elif subset_type == "mstd":
            self.transform_type = "mstd"

        self.t = self.proper_transform()          
        
        self.namelist = []
        self.custom_subset(dataset_index, subset_type=subset_type)
        self.namelist.sort(key=lambda x:natural_keys(x[0]))
    
    def __getitem__(self, index):

        if self.retrieve_images:
            img_name = self.namelist[index][0]
            img = PIL.Image.open(img_name)
            img_tensor = self.t(img)
        else:
            img_tensor = torch.FloatTensor([1])
        img_tensor
        label = np.array([int(self.namelist[index][1])])

        return img_tensor, label

    def __len__(self):
        return len(self.namelist)

# The framewise dataset trains only on the training games, but after the first 20% of each game
# Thus it can be used with 80% of the training games
# The ED network can be trained on 80% of all games and tested on 20% of all games (per frame VGG has not seen them)
# or trained on 80% of training games and tested on 100% of testing games (per frame VGG has not seen them)

class CustomDatasetViewpointFramewise(data.Dataset):
    
    namelist = None
    path = None
    balanced = False
    subset_type = None
    random_crop = None
    center_crop = None
    scale = None
    transform_type = None

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset
    def compute_mean_std(self,loader): 

        mean = np.array([0,0,0],dtype=np.float64)
        std = np.array([0,0,0],dtype=np.float64)
        batch_whole = 0

        for batch, (data,label) in enumerate(loader):
            mean += np.mean(data.numpy(),axis=(0,2,3))*float(data.size()[0])
            batch_whole += data.size()[0]

        print mean / batch_whole

        batch_whole = 0
        for batch, (data,label) in enumerate(loader):
            std += np.std(data.numpy(),axis=(0,2,3))*math.sqrt(float(data.size()[0]))
            batch_whole += data.size()[0]

        print std / math.sqrt(batch_whole)

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.47114186 , 0.5305849 , 0.37241445],
            std=[3.1218878 ,  3.33657165 , 3.02621915]
        )

        compute_mstd_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor(), # To test the compute_mean_std method, normalize the dataset to see if the mean and std are 0s and 1s respectively
        ])

        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.transform_type == "mstd":
            return compute_mstd_transform
        elif self.transform_type == "train":
            return train_transform
        elif self.transform_type == "test":
            return test_transform       
    
    def custom_subset(self, subset_type="training"):

        #all_games = ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108","2016121110","2016122411"]
        training_games = ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108"]
        testing_games = []

        all_images = glob.glob(self.path+"/**/*.jpg")+glob.glob(self.path+"/scoreboards/**/*.jpg")
    
        skip_counter = 0
        subset_counter = 0
        subset_activate = False

        play_annotations_dir = "./annotations/full_viewpoint_annotations_2016"
        annotated_files = glob.glob(play_annotations_dir+"/*_c.txt")

        labels_entire = {}
        games = testing_games
        if subset_type == "training" or subset_type == "mstd" or subset_type == "testing":
            games = training_games

        for game in games:
            labels_entire[game] = []

        label_count = [0,0,0]
        for file in annotated_files:
            old = -1
            a = 1
            with open(file,"rb") as f:

                First_score = True 
                game_id = os.path.basename(file).split("_c.txt")[0]
                if game_id in games:

                    lines = f.readlines()
                    for line in lines:
                        line = line.strip("\n")
                        s = line.split(" ")

                        if old == -1:
                            a = int(s[0])
                            old = int(s[1])
                        elif s[1] != old:
                            for i in range(a,int(s[0])):
                                labels_entire[game_id].append(int(old))
                            old = int(s[1])
                            a = int(s[0]) 

                    for i in range(a,a+100): # since we have no info for where it stops
                        labels_entire[game_id].append(s[1])

        inclusion_counter = 0
        select_one_in = 10
        data_index = 0
        if subset_type == "testing":
            data_index = 1

        for game in games:
            i = 0
            game_list = []
            for img_name in all_images: # make sure the images are sorted
                if game in img_name and "scoreboard" not in img_name:
                    label = int(labels_entire[game][i])
                    i += 1
                    game_list.append((img_name, label))
            remove_20_percent = int(len(game_list)*0.2)
            game_list_last = game_list[remove_20_percent:]

            for duet in game_list_last:
                inclusion_counter += 1
                if inclusion_counter % select_one_in == data_index:
                    label_count[duet[1]] += 1
                    self.namelist.append(duet)
                    if inclusion_counter == select_one_in:
                        inclusion_counter = 0

        for img_name in all_images:
            if "scoreboard" in img_name and subset_type != "testing":
                label = 2
                inclusion_counter += 1
                if inclusion_counter % select_one_in == data_index:
                    label_count[label] += 1
                    self.namelist.append((img_name, label))
                    if inclusion_counter == select_one_in:
                        inclusion_counter = 0

        #print label_count

    def __init__(self, path=None, scale=256, random_crop=[224,224], center_crop=[224,224], subset_type=None, file_type=".jpg",retrieve_images=True, viewpoints=3):
        
        self.viewpoints = viewpoints
        self.path = path
        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.subset_type = subset_type
        self.file_type = file_type
        self.retrieve_images = retrieve_images


        if subset_type == "training":
            self.transform_type = "train"
        elif subset_type == "testing" or None:
            self.transform_type = "test"
        elif subset_type == "mstd":
            self.transform_type = "mstd"

        self.t = self.proper_transform()          
        
        self.namelist = []
        self.custom_subset(subset_type=subset_type)
        self.namelist.sort(key=lambda x:natural_keys(x[0]))
    
    def __getitem__(self, index):

        if self.retrieve_images:
            img_name = self.namelist[index][0]
            img = PIL.Image.open(img_name)
            img_tensor = self.t(img)
        else:
            img_tensor = torch.FloatTensor([1])
        img_tensor
        label = np.array([int(self.namelist[index][1])])

        return img_tensor, label

    def __len__(self):
        return len(self.namelist)

class CustomDatasetViewpointIntervals(data.Dataset):
    
    namelist = None
    path = None
    balanced = False
    subset_type = None
    random_crop = None
    center_crop = None
    scale = None
    transform_type = None

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.47114186 , 0.5305849 , 0.37241445],
            std=[3.1218878 ,  3.33657165 , 3.02621915]
        )


        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.transform_type == "train":
            return train_transform
        elif self.transform_type == "test":
            return test_transform       
    
    def custom_subset(self, subset_type="training",splitting="whole",overlap="consecutive", interval_samples_per_game = 1000, interval_size=32, only_extract=False):

        #all_games = ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108","2016121110","2016122411"]
        if not only_extract:
            training_games = ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108"]
            testing_games = ["2016121110","2016122411"]
        else:
            training_games = []
            testing_games = ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108","2016121110","2016122411"]

        all_images = glob.glob(self.path+"/**/*.jpg")

        play_annotations_dir = "./annotations/full_viewpoint_annotations_2016"
        annotated_files = glob.glob(play_annotations_dir+"/*_c.txt")

        labels_entire = {}
        games = testing_games
        if subset_type == "training" or (subset_type == "testing" and splitting == "20"):
            games = training_games


        for game in games:
            labels_entire[game] = []


        for file in annotated_files:
            old = -1
            a = 1
            with open(file,"rb") as f:

                First_score = True 
                game_id = os.path.basename(file).split("_c.txt")[0]
                if game_id in games:

                    lines = f.readlines()
                    for line in lines:
                        line = line.strip("\n")
                        s = line.split(" ")

                        if old == -1:
                            a = int(s[0])
                            old = int(s[1])
                        elif s[1] != old:
                            for i in range(a,int(s[0])):
                                labels_entire[game_id].append(int(old))
                            old = int(s[1])
                            a = int(s[0]) 

                    for i in range(a,a+100): # since we have no info for where it stops
                        labels_entire[game_id].append(s[1])


        if subset_type == "training":
            for game in games:
                i = 0
                game_list = []
                for img_name in all_images:
                    if game in img_name:
                        label = int(labels_entire[game][i])
                        i += 1
                        game_list.append((img_name,label))
                remove_20_percent = int(len(game_list)*0.2)
                game_list_last = game_list[remove_20_percent:]

                for j in range(interval_samples_per_game):
                    x_start = random.randint(0,len(game_list_last)-interval_size)
                    x_end = x_start + interval_size
                    self.namelist.append(game_list_last[x_start:x_end])

        if subset_type == "testing":
            for game in games:
                i = 0 
                game_list = []
                for img_name in all_images:
                    if game in img_name:
                        label = int(labels_entire[game][i])
                        i += 1
                        game_list.append((img_name,label))
                if splitting == "20":
                    remove_80_percent = int(len(game_list)*0.2)
                    game_list_last = game_list[:remove_80_percent]
                elif splitting == "whole":
                    game_list_last = game_list[:]

                x_start = 0
                if overlap == "consecutive":
                    x_end = x_start + interval_size
                    while x_end < len(game_list_last):
                        self.namelist.append(game_list_last[x_start:x_end])
                        x_start += interval_size
                        x_end = x_start + interval_size

                elif overlap == "sliding":

                    step_size = interval_size / 10                
                    x_end = x_start + interval_size
                    while x_end < len(game_list_last):
                        self.namelist.append(game_list_last[x_start:x_end])
                        x_start += step_size
                        x_end = x_start + interval_size

        '''
        if subset_type == "testing":
            for interval in self.namelist:
                print(len(interval), interval[0][0], interval[255][0])
        '''

        '''
        inclusion_counter = 0
        select_one_in = 10
        data_index = 0
        if subset_type == "testing":
            data_index = 1

        for game in games:
            i = 0
            game_list = []
            for img_name in all_images: # make sure the images are sorted
                if game in img_name and "scoreboard" not in img_name:
                    label = int(labels_entire[game][i])
                    i += 1
                    game_list.append((img_name, label))
            remove_20_percent = int(len(game_list)*0.2)
            game_list_last = game_list[remove_20_percent:]

            for duet in game_list_last:
                inclusion_counter += 1
                if inclusion_counter % select_one_in == data_index:
                    self.namelist.append(duet)
                    inclusion_counter = 0
        '''

        #print label_count

    def __init__(self, path=None, scale=256, random_crop=[224,224], center_crop=[224,224], subset_type=None, file_type=".jpg",viewpoints=3,splitting="whole",overlap="consecutive", interval_samples_per_game = 1000, interval_size=32,only_extract=False):
        
        self.viewpoints = viewpoints
        self.path = path
        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.subset_type = subset_type
        self.file_type = file_type


        if subset_type == "training":
            self.transform_type = "train"
        elif subset_type == "testing" or None:
            self.transform_type = "test"
        elif subset_type == "mstd":
            self.transform_type = "mstd"

        self.t = self.proper_transform()          
        
        self.namelist = []
        self.custom_subset(subset_type=subset_type,splitting=splitting,overlap=overlap, interval_samples_per_game = interval_samples_per_game, interval_size=interval_size,only_extract=only_extract)
        self.only_extract = only_extract
    
    def __getitem__(self, index):

        # return all of the images together
        img_names = self.namelist[index][:]
        my_img_interval_list = []
        my_labels = []
        for i, img_name in enumerate(img_names):
            img = PIL.Image.open(img_name[0])
            img_tensor = self.t(img)
            img_tensor = img_tensor.unsqueeze(0)
            label = torch.LongTensor(np.array([int(img_name[1])]))
            label = label.unsqueeze(0)
            my_img_interval_list.append(img_tensor)
            my_labels.append(label)
        img_tensor = torch.cat(my_img_interval_list)
        label_tensor = torch.cat(my_labels)

        if not self.only_extract:
            for video in ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108","2016121110","2016122411"]:
                if video in self.namelist[index][0][0]:
                    video_name = video
                    break
            return img_tensor, label_tensor, video_name
        else:
            for video in ["2016090800","2016091100","2016091101","2016091808","2016091813","2016092500","2016100201","2016100202","2016112710","2016120402","2016121108","2016121110","2016122411"]:
                if video in self.namelist[index][0][0]:
                    video_name = video
                    break
            
            return img_tensor, label_tensor, video_name

    def __len__(self):
        return len(self.namelist)


class CustomClusteringDataset(data.Dataset):


    def proper_transform(self,a,b,c,d):

        normal_transform = transforms.Compose([
            transforms.Normalize([a],[b]),
            transforms.Normalize([c],[d]),
        ])
        return normal_transform



    def __init__(self, path = None, subset_type=None, load_number = 100):



        self.subset_type = subset_type
        self.path = path       

        npy_orig_images = []
        npy_orig_labels = []
        npy_images = glob.glob(self.path+"/full_frame_data_2016_npy_trinary/*.npy")#random.sample(glob.glob(self.path+"/full_frame_data_2016_npy_trinary/*.npy"),load_number)
        for i in range(len(npy_images)):
            batch_id = os.path.basename(npy_images[i]).split(".npy")[0]
            npy_orig_images.append(self.path+"/full_frame_data_2016_orig_trinary/"+str(batch_id)+"_seq.npy")
            npy_orig_labels.append(self.path+"/full_frame_data_2016_orig_trinary/"+str(batch_id)+"_tar.npy")

        self.npy_images = {}
        for i in range(len(npy_images)):
            batch_id = os.path.basename(npy_images[i]).split(".npy")[0]
            self.npy_images[batch_id] = []
            num = np.load(npy_images[i])
            ori = np.load(npy_orig_images[i])
            tar = np.load(npy_orig_labels[i])
            for j in range(16):
                self.npy_images[batch_id].append((num[j,:,:],ori[0,j,:,:,:],tar))

    def __getitem__(self, index):
        index_batch = index / 16
        index_part = index % 16

        keys = self.npy_images.keys()
        num = self.npy_images[keys[index_batch]][index_part]
        num0 = imresize(num[0], (64,64), mode='F')


        minimum = float(np.min(num0))
        maximum = float(np.max(num0))
        if minimum + maximum != 0:
            t = self.proper_transform(0, maximum,0.03325934,0.07267264)
            mytensor = torch.from_numpy(num0)
            mytensor = mytensor.unsqueeze(0)
            mytensor = t(mytensor)
        else:
            t = self.proper_transform(0,1,0.03325934,0.07267264)
            mytensor = t(torch.from_numpy(num0).unsqueeze(0))

        num = mytensor, num[1], num[2], torch.from_numpy(num[0]).unsqueeze(0)
        return num

    def __len__(self):
        return len(self.npy_images)*16


class MasterPlaysets(data.Dataset):

    #   transform type in ["mstd","train","test"]
    #   returns appropriate transformation to apply to the dataset

    def proper_transform(self):
        
        normalize = transforms.Normalize(
            mean=[0.486, 0.57, 0.364],
            std=[0.117,  0.113,  0.121]
        )

        compute_mstd_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.ToTensor(), # To test the compute_mean_std method, normalize the dataset to see if the mean and stds are 0 and 1s respectively
        ])

        train_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.RandomCrop((self.random_crop[0],self.random_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Scale(self.scale),
            transforms.CenterCrop((self.center_crop[0],self.center_crop[1])),
            transforms.ToTensor(),
            normalize
        ])

        if self.subset_type == "mstd":
            return compute_mstd_transform
        elif self.subset_type == "testing" or self.subset_type == "exciting":
            return test_transform     
        elif self.subset_type == "training":
            if not self.excitation:
                return train_transform  
            else: # subset will be training even if we want to use excitation on the image centers
                return test_transform

    def mapping(self, entire_dataset_dict):

        trainable = 0
        true_index = 0
        null_games = ["2016121110","2016122411"]#["2016112710","2016091100","2016122411"] # MAYBE THESE ["2016121110","2016122411"]
        for game_code in entire_dataset_dict.keys():
            for play_index in entire_dataset_dict[game_code].keys():

                trainable = 0

                untrainable_indices = [10,20,30,40,50,60,70,80,90]

                if self.part == "part":
                    if true_index % 100 in untrainable_indices:
                        trainable = -1
                    else:
                        trainable = 1

                    if game_code in null_games:
                        trainable = 0
                else:
                    trainable = 1
                    if game_code in null_games and self.subset_type != "entire":
                        trainable = -1

                if self.view == "both":
                    entire_dataset_dict[game_code][play_index][0] = trainable
                    entire_dataset_dict[game_code][play_index][1] = trainable
                elif self.view == "0":
                    entire_dataset_dict[game_code][play_index][0] = trainable
                    entire_dataset_dict[game_code][play_index][1] = 0
                else:
                    entire_dataset_dict[game_code][play_index][0] = 0
                    entire_dataset_dict[game_code][play_index][1] = trainable

                true_index += 1

        return entire_dataset_dict
    
    def custom_subset(self):

        all_images = []
        for root, dirnames, filenames in os.walk(self.path):
            if "scoreboard" not in root and "2016112709" not in root: # 2016112709 has not been annotated
                for filename in fnmatch.filter(filenames, '*.jpg'):
                    all_images.append(os.path.join(root, filename))

        play_annotations = {}
        play_annotations_pred = {}

        if not os.path.isfile("./experiments/playtype_EB/play_annotations_gt.pkl"):
            self.proper_csv_2016 = {}
            with open('./annotations/csv_files/proper_2016.csv','rb') as f:
                spamreader = csv.reader(f, delimiter=',', quotechar='"')
                prev_game_id = -1
                for row in spamreader:
                    game_id = row[0]
                    if game_id != prev_game_id:
                        prev_game_id = game_id
                        play_index = 1

                    # offense has to be non-empty
                    active = (row[5] != "")
                    play_type = row[17]

                    if active:
                        if game_id not in self.proper_csv_2016.keys():
                            self.proper_csv_2016[game_id] = {}
                        if play_index not in self.proper_csv_2016[game_id]:
                            self.proper_csv_2016[game_id][play_index] = None

                        self.proper_csv_2016[game_id][play_index] = play_type
                        play_index += 1

            play_annotations_dir = "./annotations/full_viewpoint_annotations_2016"

            annotated_files = glob.glob(play_annotations_dir+"/*_c.txt")

            for file in annotated_files:
                play_index = 1
                with open(file,"rb") as f:

                    First_score = True # all files begin with a scoreboard view
                    game_id = os.path.basename(file).split("_c.txt")[0]

                    lines = f.readlines()

                    for line in lines:

                        line = line.strip("\n")
                        line = line.strip("\r")
                        s = line.split(" ")

                        if s[1] == '0':
                            view0 = int(s[0])
                        elif s[1] == '1':
                            view1 = int(s[0])
                        elif s[1] == '2' and First_score:
                            First_score = False
                        elif s[1] == '2':
                            view2 = int(s[0])

                            if game_id in self.proper_csv_2016.keys() and play_index in self.proper_csv_2016[game_id].keys():

                                play_type = self.proper_csv_2016[game_id][play_index]

                                if game_id not in play_annotations.keys():
                                    play_annotations[game_id] = {}

                                for d in range(view0,view1):
                                    d_frame = str(d).zfill(8)
                                    play_annotations[game_id][d_frame] = [play_index, play_type, 0]

                                for d in range(view1,view2): 
                                    d_frame = str(d).zfill(8)
                                    play_annotations[game_id][d_frame] = [play_index, play_type, 1]

                            play_index += 1

            pickle.dump(play_annotations, open("./experiments/playtype_EB/play_annotations_gt.pkl","wb"))
        else:
            play_annotations = pickle.load(open("./experiments/playtype_EB/play_annotations_gt.pkl","rb"))

        if not os.path.isfile("./experiments/playtype_EB/play_annotations_pred.pkl"):
            self.proper_csv_2016 = {}
            with open('./annotations/csv_files/proper_2016.csv','rb') as f:
                spamreader = csv.reader(f, delimiter=',', quotechar='"')
                prev_game_id = -1
                for row in spamreader:
                    game_id = row[0]
                    if game_id != prev_game_id:
                        prev_game_id = game_id
                        play_index = 1

                    # offense has to be non-empty
                    active = (row[5] != "")
                    play_type = row[17]

                    if active:
                        if game_id not in self.proper_csv_2016.keys():
                            self.proper_csv_2016[game_id] = {}
                        if play_index not in self.proper_csv_2016[game_id]:
                            self.proper_csv_2016[game_id][play_index] = None

                        self.proper_csv_2016[game_id][play_index] = play_type
                        play_index += 1

            play_annotations_dir = "./experiments/final_viewpoint_experiments/ED32_vgg_frozen/"

            annotated_files = glob.glob(play_annotations_dir+"/*_c_estimated_viterbi.txt")

            for file in annotated_files:
                play_index = 1
                with open(file,"rb") as f:

                    First_score = True 
                    game_id = os.path.basename(file).split("_c_estimated_viterbi.txt")[0]

                    lines = f.readlines()
                    
                    for line in lines:

                        line = line.strip("\n")
                        line = line.strip("\r")
                        s = line.split(" ")

                        if s[1] == '0':
                            view0 = int(s[0])
                        elif s[1] == '1':
                            view1 = int(s[0])
                        elif s[1] == '2' and First_score:
                            First_score = False
                        elif s[1] == '2':
                            view2 = int(s[0])

                            if game_id not in play_annotations_pred.keys():
                                play_annotations_pred[game_id] = {}

                            # STARTING FROM view0 and going to view2, it has overlap with proper file in gt
                            # So read that file, find the best overlap every time and assign that label here

                            temp_gt = open("./experiments/final_playtype_experiments/gt/{}.txt".format(game_id),"r")

                            z = [0,0,0]
                            for myline in temp_gt.readlines():
                                myline = myline.strip("\n")
                                from_f, to_f, label_f = myline.split(" ")
                                for fff in range(int(from_f), int(to_f)):
                                    if fff in [r for r in range(view0, view2)]:
                                        #print(view0, view2, from_f, to_f)
                                        z[int(label_f)] += 1
                                #print(game_id, from_f, to_f, label_f)

                            playtype = z.index(max(z))
                            print(playtype, z)
                            pred_types = ["CLOCK STOP","PASS","RUSH"]
                            playtype = pred_types[playtype]

                            for d in range(view0,view1): 
                                d_frame = str(d).zfill(8)
                                play_annotations_pred[game_id][d_frame] = [play_index,playtype, 0]
                            for d in range(view1,view2): 
                                d_frame = str(d).zfill(8)
                                play_annotations_pred[game_id][d_frame] = [play_index, playtype, 1]

                            play_index += 1

            pickle.dump(play_annotations_pred, open("./experiments/playtype_EB/play_annotations_pred.pkl","wb"))
        else:
            play_annotations_pred = pickle.load(open("./experiments/playtype_EB/play_annotations_pred.pkl","rb")) # These should not obtain labels, they are only used for prediction


        if self.from_file=="pred":
            play_annotations = play_annotations_pred

        self.play_types = ['CLOCK STOP', 'EXTRA POINT', 'FIELD GOAL', 'FUMBLES', 'KICK OFF', 'NO PLAY', 'PASS', 'PUNT', 'QB KNEEL', 'RUSH', 'SACK', 'SCRAMBLE', 'TWO-POINT CONVERSION'] # [6] PASS [9] RUSH [-] OTHER

        # the dataset dict will decide how to train based on indicators (trainable) per game, play and view
        entire_dataset_dict = {}

        for img_name in all_images:

            img_number = os.path.basename(img_name).split(".")[0]
            parent_directory = os.path.abspath(os.path.join(img_name, os.pardir))
            game_code = parent_directory.split("/")[-1]
            grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir)).split("/")[-1]
            type_code = grandparent_directory.split("/")[-1]
            try:
                p = play_annotations[game_code][img_number]
                play_index, play_type, play_view = p[0], p[1], p[2]
            except:
                continue
            try:
                _ = entire_dataset_dict[game_code]
            except:
                print(game_code)
                entire_dataset_dict[game_code] = {}

            try:
                _ = entire_dataset_dict[game_code][play_index]
            except:
                entire_dataset_dict[game_code][play_index] = {}

            entire_dataset_dict[game_code][play_index][play_view] = 0 # whether it is trainable

        '''
        mina = 0
        minb = 0
        minc = 0
        d = 0
        avg_c = 0
        for game in entire_dataset_dict.keys():
            for play_index in entire_dataset_dict[game].keys():
                a ,b = entire_dataset_dict[game][play_index][1], entire_dataset_dict[game][play_index][2]
                c = a+b
                avg_c += len(a)
                if len(a) > mina:
                    mina = len(a)
                if len(b) > minb:
                    minb = len(b)
                if len(c) > minc:
                    minc = len(c)
                    print("in game "+str(game)+" for play "+str(play_index), minc, len(a), len(b), len(c))
                d += 1

        print(avg_c / (1.0*d))

        print(d/13.0)
        print(mina, minb, minc)
        print("Done")
        '''

        trainable_0 = [0,0,0]
        trainable_1 = [0,0,0]
        for game in entire_dataset_dict.keys():
            for play_index in entire_dataset_dict[game].keys():
                trainable_0[1+entire_dataset_dict[game][play_index][0]] += 1
                trainable_1[1+entire_dataset_dict[game][play_index][1]] += 1
        print trainable_0, trainable_1

        self.trainable = self.mapping(entire_dataset_dict)

        trainable_0 = [0,0,0]
        trainable_1 = [0,0,0]
        for game in self.trainable.keys():
            for play_index in self.trainable[game].keys():
                trainable_0[1+self.trainable[game][play_index][0]] += 1
                trainable_1[1+self.trainable[game][play_index][1]] += 1
        print trainable_0, trainable_1



        for img_name in all_images:

            img_number = os.path.basename(img_name).split(".")[0]
            parent_directory = os.path.abspath(os.path.join(img_name, os.pardir))
            game_code = parent_directory.split("/")[-1]
            grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir)).split("/")[-1]
            type_code = grandparent_directory.split("/")[-1]

            try:
                p = play_annotations[game_code][img_number]
                play_index, play_type, play_view = p[0], p[1], p[2]
            except: # image not in annotations
                continue

            #print(self.trainable[game_code][play_index][play_view], self.subset_type)

            indataset = (self.trainable[game_code][play_index][play_view] == 1 and self.subset_type != 'testing') or (self.trainable[game_code][play_index][play_view] == -1 and self.subset_type == 'testing') # based on image trainability

            if self.subset_type == "entire":
                indataset = True
            if self.subset_type == "exciting":
                indataset = True

            if indataset:
                try:
                    _ = self.namelist[game_code]
                except:
                    self.namelist[game_code] = {}

                try:
                    _ = self.namelist[game_code][play_index]
                except:
                    self.namelist[game_code][play_index] = [play_type, []]

                self.namelist[game_code][play_index][1].append(img_name)

        '''
        plays_in_dataset = 0
        avg_length = 0
        min_length = 1000
        max_length = 0
        d = 0
        for game in self.namelist.keys():
            for play_index in self.namelist[game]:
                plays_in_dataset += len(self.namelist[game][play_index][1])
                if min_length > len(self.namelist[game][play_index][1]):
                    min_length = len(self.namelist[game][play_index][1])
                if max_length < len(self.namelist[game][play_index][1]):
                    max_length = len(self.namelist[game][play_index][1])
                avg_length += 1
                if len(self.namelist[game][play_index][1]) < 32:
                    d += 1
        print(d)
        '''

        #print("avg length is ", plays_in_dataset*1.0/avg_length, avg_length," are the games", min_length, max_length)

        '''
        # Remove all entries in the namelist that do not contain frame_select images
        removals = []
        for game in self.namelist.keys():
            for play_index in self.namelist[game]:
                if len(self.namelist[game][play_index][1]) < self.frame_select:
                    removals.append((game, play_index))
        for removal in removals:
            del self.namelist[removal[0]][removal[1]]
        '''

        # Map indices to all games and plays that we have obtained to be able to use them in the classifier

        # Go through the entire namelist and if a game code and play_index contain less than the limit of images (proper length), remove them completely

        if self.subset_type != "entire":
            removals = []
            for game_code in self.namelist.keys():
                for play_index in self.namelist[game_code].keys():
                    if len(self.namelist[game_code][play_index][1]) < self.proper_length:

                        removals.append((game_code, play_index))
                        print("removing, ", game_code, play_index, len(self.namelist[game_code][play_index][1]), self.proper_length)


            for game,index in removals:
                del self.namelist[game][index]

        else:
            # go through entire dataset and produce intervals for each game code
            for game_code in self.namelist.keys():
                f = open("./experiments/final_playtype_experiments/gt/{}.txt".format(game_code),"w")

                for play_index in self.namelist[game_code].keys():
                    class_number = self.play_types.index(self.namelist[game_code][play_index][0])
                    if class_number != 6 and class_number != 9:
                        class_number = 0 # other
                    elif class_number == 6:
                        class_number = 1 # pass
                    else:
                        class_number = 2 # rush   

                    base1=os.path.basename(self.namelist[game_code][play_index][1][0])
                    base2=os.path.basename(self.namelist[game_code][play_index][1][len(self.namelist[game_code][play_index][1])-1])
                    interval = (int(os.path.splitext(base1)[0]), int(os.path.splitext(base2)[0]))
                    print("game {} play {} class {} and interval {} - {}".format(game_code,play_index,class_number,interval[0], interval[1]))
                    f.write("{} {} {}\n".format(interval[0],interval[1],class_number))
                f.close()

        

        index = 0
        self.maxlength = 0
        for game_code in self.namelist.keys():
            my_play_indices = self.namelist[game_code].keys()
            my_play_indices.sort()
            for play_index in my_play_indices:
                print(index, play_index)

                self.indexed_namelist[index] = [game_code, play_index]
                index += 1
                self.maxlength += 1

                 
    def __init__(self, path=None, scale=[256,256], random_crop=[224,224], center_crop=[224,224], subset_type="training", retrieve_images=False,from_file="gt", part="part", view="both", excitation=False, proper_length=128):
        
        self.path = path

        self.scale = scale
        self.random_crop = random_crop
        self.center_crop = center_crop

        self.subset_type = subset_type
        self.proper_length = proper_length

        self.from_file = from_file
        self.part = part
        self.retrieve_images = retrieve_images
        self.excitation = excitation
        self.view = view

        self.indexed_namelist = {}
        self.namelist = {}

        self.custom_subset()

    def __getitem__(self, index):

        game_code, play_index = self.indexed_namelist[index]
        play_type, imgs_in_play = self.namelist[game_code][play_index]

        imgs = None

        if self.retrieve_images:
            imgs_raw = []
            t = self.proper_transform()

            #print(len(imgs_in_play), self.subsampling, int(len(imgs_in_play)/self.subsampling_rate))

            
            if self.subset_type == "training":
                t_start = random.randint(0,len(imgs_in_play)-self.proper_length)
                t_end = t_start + self.proper_length
            else:
                t_start = 0
                t_end = len(imgs_in_play)

            for index in range(t_start,t_end):
                img = imgs_in_play[int(round(index))]
                #print("obtained", int(round(true_index)))
                img_tensor = t(PIL.Image.open(img))
                img_tensor.unsqueeze_(0)
                imgs_raw.append(img_tensor)
        
            imgs = torch.cat(imgs_raw,0)
                #print(imgs.size(), self.proper_length)
        else:
            imgs = torch.FloatTensor(len(imgs_in_play))

        # mapping for - -> "other", 6 -> "pass", 9 -> "rush"
        class_number = self.play_types.index(play_type)

        if class_number != 6 and class_number != 9:
            class_number = 0 # other
        elif class_number == 6:
            class_number = 1 # pass
        else:
            class_number = 2 # rush     

        #print("play type {} and length {} / {}".format(play_type, len(imgs_in_play), torch.LongTensor([class_number])))

        return imgs, torch.LongTensor([class_number]), imgs_in_play

    def __len__(self):
        return self.maxlength

    def compute_mean_std(self, loader): 

        mean = np.array([0,0,0],dtype=np.float64)
        std = np.array([0,0,0],dtype=np.float64)
        batch_whole = 0

        for batch, (data, label) in enumerate(loader):
            if batch >= 100:
                break
            data = data[0]
            mean += np.mean(data.numpy(),axis=(0,2,3))*float(data.size()[0])
            batch_whole += data.size()[0]

        #print mean / batch_whole

        batch_whole = 0
        for batch, (data,label) in enumerate(loader):
            if batch >= 100:
                break
            data = data[0]
            sample_std = np.std(data.numpy(),axis=(0,2,3))
            sample_variance = [sample_std[0]**2,sample_std[1]**2,sample_std[2]**2]
            std += [sample_variance[0]*(float(data.size()[0])-1),sample_variance[1]*(float(data.size()[0])-1),sample_variance[2]*(float(data.size()[0])-1)] # sample bias, so square and multiply with N-1 to get the sum over the values
            batch_whole += data.size()[0]

        std = [math.sqrt(std[0])/math.sqrt(batch_whole),math.sqrt(std[1])/math.sqrt(batch_whole),math.sqrt(std[2])/math.sqrt(batch_whole)]
        #print std