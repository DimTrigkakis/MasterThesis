import sys

sys.path.insert(0, './utils/')

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from datasets import CustomClusteringDataset
from models import VAE, CVAE, W2V
from magnificentoracle import MagnificentOracle
import scipy.misc
import chamber
import os.path
import pickle
import random
from sklearn.metrics import pairwise_distances_argmin_min

torch.multiprocessing.set_sharing_strategy('file_system')
ngpu = 1
nz = 1000
ngf = 64
ndf = 64
nc = 1
batch_size = 64

def proper_untransform(c=0.0124364, d=0.0442814):
    normal_transform = transforms.Compose([
        transforms.Normalize([-c * 1.0 / d], [1.0 / d]),
    ])
    return normal_transform

def show_tensor(t):
    plt.figure(1)
    for i in range(1):
        img = t.numpy()[i,:,:]
        plt.subplot(1, 1, i + 1)
        plt.imshow(img)
        plt.draw()
        plt.axis('off')
    plt.show()

class clustering():

    def __init__(self, sizes=(128,128), batch_size=32, save_models=True):

        self.sizes = sizes
        self.save_models = save_models
        self.batch_size = batch_size

        self.trainset = CustomClusteringDataset(sizes=self.sizes, path="/scratch/datasets/NFLsegment/experiments/vpEB_image_dataset/all_images/")
        self.trainloader = torch.utils.data.DataLoader(dataset=self.trainset, batch_size=batch_size, shuffle=True, num_workers=16)
        mylogger.log("Loaded dataset with {} sequences of attention maps".format(len(self.trainloader)))

        self.VAE = VAE(sizes=self.sizes).cuda()
        
        self.oracle = chamber.Oracle()
        self.epochs = 2000

    def run(self):
        lr_mod = 1.0
        lr_change = 0.99#0.995
        lr_base = 1e-3
        wd = 0
        for epoch in range(1,self.epochs):

            self.optimizer = optim.Adam(self.VAE.parameters(), weight_decay=wd, lr=lr_base*lr_mod)
            lr_mod *= lr_change
            for i, datum in enumerate(self.trainloader):
                #if i != 5*16/self.batch_size:
                #    continue

                orig_datum = Variable(datum[0])
                in_datum = Variable(datum[1].cuda())

                Z, z_mu, z_var, out_datum = self.VAE(in_datum)
                loss = self.loss_function(out_datum, in_datum,  z_mu, z_var)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(epoch, "/", self.epochs, " for batch ", i, " out of ", len(self.trainloader), "loss: ",loss.data.cpu().numpy()[0], ", lr=",lr_mod*lr_base)

                if epoch % 50 == 0:
                    if i < 256/self.batch_size:
                        #print(orig_datum.squeeze(0).size(), in_datum.squeeze(0).size(), out_datum.size())
                        if self.batch_size != 1:
                            self.oracle.visualize_tensors([orig_datum.squeeze(0).data.cpu(), in_datum.squeeze(0).data.cpu(), out_datum.data.cpu()],file="./small_epochs/viewpoint_result_e"+str(epoch)+"_b"+str(i))
                        else:
                            self.oracle.visualize_tensors([orig_datum.data.cpu(), in_datum.data.cpu(), out_datum.data.cpu()],file="./small_epochs/viewpoint_result_e"+str(epoch)+"_b"+str(i))
                if epoch % 50 == 0:
                    if i ==0 and self.save_models:
                        torch.save(self.VAE.state_dict(), "./model_"+str(epoch)+".model")
                        self.VAE.load_state_dict(torch.load("./model_"+str(epoch)+".model"))

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1,self.sizes[0]*self.sizes[1]), x.view(-1, self.sizes[0]*self.sizes[1]))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.size()[0] * self.sizes[0]*self.sizes[1]

        return BCE + KLD

def one_hot(codeword, n_clusters):
    oh = torch.FloatTensor(n_clusters).zero_()
    oh[codeword] = 1.0
    return oh

def zero_hot(codeword, n_clusters):
    oh = torch.LongTensor([int(codeword)])
    return oh

def word2vec(mode="raw",sizes=(64,64), n_clusters=8, J_clusters = 2, word_features=256, lr=0.001):

    batch_size = 32
    oracle = chamber.Oracle()
    trainset = CustomClusteringDataset(sizes=sizes, path="/scratch/datasets/NFLsegment/experiments/vpEB_image_dataset/all_images/")
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=4)
    trainloader_single = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=False, num_workers=4)

    pre_load = False
    pre_train = True

    if mode == "raw":
        directory = "./clustering/raw_representation_c"+str(n_clusters)+"/"

    mylogger.set_log(logfile=directory+"/log.txt")
    estimator = pickle.load(open(directory+"/cluster_estimator.pkl","rb"))

    if not os.path.isfile(directory+"/cluster_estimator_closest.pkl"):
        vector_aggregator = []
        whole_data = []

        for batch_idx, (data) in enumerate(trainloader_single):
            print(batch_idx, len(trainloader))
            vector_datum = data[1].numpy().reshape(sizes[0] * sizes[1])
            vector_aggregator.append(vector_datum)
            whole_data.append((data[1], data[0]))

        vector_aggregator = np.asarray(vector_aggregator)
        closest, _ = pairwise_distances_argmin_min(estimator.cluster_centers_, vector_aggregator)
        pickle.dump(closest, open(directory+"/cluster_estimator_closest.pkl","wb"))



    if (not os.path.isfile(directory+"/word.pkl")) or pre_load:
        word_pair_trainer = []
        pair_dist = batch_size/8
        counter1, counter0 = 0, 0
        for i, datum in enumerate(trainloader):
            vector_datum = datum[1].numpy().reshape(-1, sizes[0] * sizes[1])
            prediction_c = estimator.predict(vector_datum)
            for j in range(batch_size):
                j_start = max(0, j-pair_dist)
                j_end = min(j+pair_dist, batch_size)
                for k in range(j_start,j_end):
                    if k != j:
                        word_pair_trainer.append((one_hot(prediction_c[j],n_clusters), zero_hot(prediction_c[k],n_clusters)))
                        if (int(prediction_c[k]) == 1):
                            counter1 += 1
                        else:
                            counter0 += 1
        print(counter1,counter0)

        pickle.dump(word_pair_trainer, open(directory+"/word.pkl","wb"))


    if (not os.path.isfile(directory+"/model_29.model")) or pre_train:

        w2v_model = W2V(dict_size=n_clusters, hidden_features=word_features).cuda()
        word_pair_trainer = pickle.load(open(directory+"/word.pkl","rb"))

        epochs = 30
        m = nn.LogSoftmax()
        loss = nn.NLLLoss().cuda()
        optimizer = optim.Adam(w2v_model.parameters(), weight_decay=0, lr=lr)
        mylogger.log("training W2V model")
        for epoch in range(epochs):
            examples = 0
            total_loss = 0
            random.shuffle(word_pair_trainer)
            for pair in word_pair_trainer:
                #print(examples)
                input_word = Variable(pair[0].cuda()).unsqueeze(0)
                output_word = Variable(pair[1].cuda())
                #print(input_word, output_word)
                word = w2v_model(input_word)[1]
                myloss = loss(m(word), output_word)
                examples += 1
                total_loss += myloss.data.cpu().numpy()[0]

                optimizer.zero_grad()
                myloss.backward()
                optimizer.step()
                if (random.random()>0.9999):
                    mylogger.log(str(myloss.data.cpu().numpy()[0]))
                #    print(str(myloss.data.cpu().numpy()[0]), word.data.cpu().numpy(), pair[1].numpy()[0])

            mylogger.log("Average loss is {}".format(total_loss*1.0/examples))
            print("Loss is ",total_loss*1.0/examples)

            torch.save(w2v_model.state_dict(), directory+"/model_"+str(epoch)+".model")

    if (not os.path.isfile(directory+"/cluster_temporal.pkl")) or pre_train:
        w2v_model.load_state_dict(torch.load(directory+"/model_29.model"))

        temporal_features = []
        for i, datum in enumerate(trainloader):
            vector_datum = datum[1].numpy().reshape(-1, sizes[0] * sizes[1])
            prediction_c = estimator.predict(vector_datum)

            for j in range(batch_size):
                inputk = one_hot(prediction_c[j],n_clusters)
                input_word = Variable(inputk.cuda()).unsqueeze(0)
                f, z = w2v_model(input_word)
                temporal_features.append(f)

        pickle.dump(temporal_features,open(directory+"/cluster_temporal.pkl","wb"))

    if (not os.path.isfile(directory+"/cluster_estimator_temporal.pkl") or (not os.path.isfile(directory+"/cluster_estimator_temporal_closest.pkl"))) or pre_train:
        temporal_features = pickle.load(open(directory+"/cluster_temporal.pkl","rb"))
        temporal_vectors = []
        for i in range(len(temporal_features)):
            vector_datum = temporal_features[i].data.cpu().numpy().reshape(word_features)
            temporal_vectors.append(vector_datum)
        temporal_vectors = np.asarray(temporal_vectors)
        print(temporal_vectors)

        estimator_temporal = KMeans(init='random', n_clusters=J_clusters, n_init=10)
        estimator_temporal.fit(temporal_vectors)

        pickle.dump(estimator_temporal, open(directory+"/cluster_estimator_temporal.pkl","wb"))
        closest, _ = pairwise_distances_argmin_min(estimator_temporal.cluster_centers_, temporal_vectors)
        pickle.dump(closest, open(directory+"/cluster_estimator_temporal_closest.pkl","wb"))

    estimator_temporal = pickle.load(open(directory+"/cluster_estimator_temporal.pkl","rb"))
    closest_temporal = pickle.load(open(directory+"/cluster_estimator_temporal_closest.pkl","rb"))
    closest = pickle.load(open(directory+"/cluster_estimator_closest.pkl","rb"))

    w2v_model = W2V(dict_size=n_clusters, hidden_features=word_features).cuda()
    w2v_model.load_state_dict(torch.load(directory+"/model_29.model"))


    print(closest, closest_temporal)
    spatial_dict = [None for i in range(n_clusters)]
    temporal_dict = [None for i in range(J_clusters)]

    for i, datum in enumerate(trainloader_single):
        for j in range(n_clusters):
            if closest[j] == i:
                spatial_dict[j] = datum
        for j in range(J_clusters):
            if closest_temporal[j] == i:
                temporal_dict[j] = datum


    for i, datum in enumerate(trainloader):
        vector_datum = datum[1].numpy().reshape(-1, sizes[0] * sizes[1])
        prediction_c = estimator.predict(vector_datum)

        bos = []
        bot = []
        bas = []
        bat = []
        for j in range(batch_size):
            inputk = one_hot(prediction_c[j],n_clusters)
            input_word = Variable(inputk.cuda()).unsqueeze(0)
            f, z = w2v_model(input_word)
            prediction_f = estimator_temporal.predict(f.data.cpu().numpy().reshape(1,word_features))
            bos.append(spatial_dict[prediction_c[j]][0])
            bot.append(temporal_dict[prediction_f[0]][0])
            bas.append(spatial_dict[prediction_c[j]][1])
            bat.append(temporal_dict[prediction_f[0]][1])

        bos = torch.cat(bos,0)
        bot = torch.cat(bot,0)
        bas = torch.cat(bas,0)
        bat = torch.cat(bat,0)

        oracle.visualize_tensors([bos, bas, bot, bat, datum[0], datum[1]],file=directory+"/explanations/mysample_explanation_"+str(i)+".png")


    '''
    for i, datum in enumerate(trainloader):
        vector_datum = datum[1].numpy().reshape(-1, sizes[0] * sizes[1])
        prediction_c = estimator.predict(vector_datum)

        for j in range(batch_size):
            inputk = one_hot(prediction_c[j],n_clusters)
            input_word = Variable(inputk.cuda()).unsqueeze(0)
            f, z = w2v_model(input_word)
            prediction_f = estimator_temporal.predict(f.data.cpu().numpy().reshape(1,32))
            print(prediction_c[j], prediction_f[0])
    '''


def cluster_raw(mode="raw", n_clusters=32, sizes=(64,64)):

    oracle = chamber.Oracle()
    trainset = CustomClusteringDataset(sizes=sizes, path="/scratch/datasets/NFLsegment/experiments/vpEB_image_dataset/all_images/")
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=False, num_workers=4)

    if mode == "raw":
        directory = "./clustering/raw_representation_c"+str(n_clusters)+"/"
    
    if not os.path.exists(directory):
        os.makedirs(directory)


    vector_aggregator = []
    whole_data = []

    for batch_idx, (data) in enumerate(trainloader):
        print(batch_idx, len(trainloader))
        vector_datum = data[1].numpy().reshape(sizes[0] * sizes[1])
        vector_aggregator.append(vector_datum)
        whole_data.append((data[1], data[0]))

    vector_aggregator = np.asarray(vector_aggregator)

    estimator = KMeans(init='random', n_clusters=n_clusters, n_init=10)
    estimator.fit(vector_aggregator)
    for i, k in enumerate(estimator.labels_):
        print(i)

        curr_dir = directory + str(k) + "/"
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        I = whole_data[i][0]
        I_orig = whole_data[i][1]

        oracle.visualize_tensors([I_orig,I],file=curr_dir + str(i) + ".png")

    pickle.dump(estimator, open(directory+"/cluster_estimator.pkl","wb"))
                

def main():
    global  mylogger

    mylogger = MagnificentOracle()
    mylogger.set_log(logfile=None)
    mylogger.log("-dotted-line")

    kmeans_init = True
    word2vec_init = True
    if word2vec_init:
        word2vec()
    elif kmeans_init:
        cluster_raw()
    else:
        c = clustering()
        c.run()

if __name__ == "__main__":
    main()


