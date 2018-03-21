
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch NFL Viewpoint Training')
parser.add_argument("--log",metavar="LOG", default=None, help="Log txt file to read logger information")
parser.add_argument("--test_interval",metavar="TEST_INTERVAL", default=None, help="Log txt file to read logger information")
args = parser.parse_args() 



loss_info = []
loss_test_info = []
acc_info = []
acc_test_info = []
training_agg_info = []
aggregate = []
training_agg = 0
training_agg_length = 0
with open(args.log,"rb") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        if "Train Epoch" in line:
            a = line.find("Train Epoch: ")
            a_index = a+len("Train Epoch: ")
            b = line.find("Loss: ")
            b_index = b+len("Loss: ")
            c = line.find("[",len("LOGGER [INFO]"))
            c_index = c+len("[")
            d = line.find("(")+len("(")
            e = line.find(")")-len("%")
            per = line[d:e]
            training_agg += float(line[b_index:])
            training_agg_length += 1
            #print training_agg_length
            epoch , loss = int(line[a_index:c_index-1]), float(line[b_index:])
            epoch += float(per)/100.0
            #plot_info.append((epoch,loss))
            #print "epoch {}, loss {}".format(epoch,loss)
            #print "THERE"
        if "Training set" in line:
            training_agg /= float(training_agg_length)
            aggregate.append(training_agg)
            training_agg_info.append([epoch, training_agg])
            training_agg = 0
            training_agg_length = 0

    epochs = 0
    for line in lines:
        line = line.strip("\n")

        if "Training set" in line:
            '''print "HERE", training_agg_length
            print line 
            training_agg /= float(training_agg_length)
            aggregate.append(training_agg)'''
            
            d = line.find("(")+len("(")
            e = line.find(")")-len("%")
            
            b = line.find("Average loss: ")
            b_index = b+len("Average loss: ")
            c = line.find("Accuracy")
            c_index = c-1
            #print line[b_index:c_index], "loss"

            per = line[d:e]
            acc = float(per)
            epochs += int(args.test_interval)
            acc_info.append([epochs,acc])
            #acc_info.append([epochs, float(line[d:e]))
            loss_info.append([epochs, line[b_index:c_index]])
            '''training_agg_info.append([epochs, training_agg])
            training_agg=  0
            training_agg_length = 0'''
    epochs = 0
    for line in lines:
        line = line.strip("\n")
        if "Testing set" in line:
            '''d = line.find("(")+len("(")
            e = line.find(")")-len("%")
            per = line[d:e]
            acc = float(per)
            epochs += int(args.test_interval)
            eval_test_info.append((epochs,acc))'''
            d = line.find("(")+len("(")
            e = line.find(")")-len("%")

            b = line.find("Average loss: ")
            b_index = b+len("Average loss: ")
            c = line.find("Accuracy")
            c_index = c-1
            #print line[b_index:c_index], "loss"

            per = line[d:e]
            acc = float(per)
            epochs += int(args.test_interval)
            acc_test_info.append([epochs,acc])
            #acc_test_info.append([epochs, float(line[d:e]))
            loss_test_info.append([epochs, float(line[b_index:c_index])])



#loss = [f[1] for f in plot_info]
#epochs = [f[0] for f in plot_info]
acc_train = [s[1] for s in acc_info]
loss_train = [s[1] for s in loss_info]
epochs_train = [s[0] for s in acc_info]
acc_test = [s[1] for s in acc_test_info]
loss_test = [s[1] for s in loss_test_info]
epochs_test = [s[0] for s in acc_test_info]
loss_agg = [f[1] for f in training_agg_info]
epochs_agg = [f[0] for f in training_agg_info]


plt.figure()
plt.plot(epochs_agg, loss_agg)
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.title('Training loss during training phase')
plt.savefig('./figures/'+str(os.path.basename(args.log).split(".")[0])+'_traing_phase_loss.png')

plt.figure()
for mu in [90,80,70,60,50,40,30,20,10]:
    plt.plot((0, 60), (mu, mu), 'k:')
plt.plot(epochs_test, acc_test)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.title('Testing accuracy, evaluation')
plt.ylim([0,100])
plt.savefig('./figures/'+str(os.path.basename(args.log).split(".")[0])+'_test_acc.png')


plt.figure()
plt.plot(epochs_test, loss_test)
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.title('Testing loss, evaluation')
plt.savefig('./figures/'+str(os.path.basename(args.log).split(".")[0])+'_test_loss.png')

plt.figure()
for mu in [90,80,70,60,50,40,30,20,10]:
    plt.plot((0, 60), (mu, mu), 'k:')
plt.plot(epochs_train, acc_train)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.title('Training accuracy, evaluation')
plt.ylim([0,100])
plt.savefig('./figures/'+str(os.path.basename(args.log).split(".")[0])+'_train_acc.png')


plt.figure()
plt.plot(epochs_train, loss_train)
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.title('Training loss, evaluation')
plt.savefig('./figures/'+str(os.path.basename(args.log).split(".")[0])+'_train_loss.png')
'''


plt.figure()
plt.plot(epochs_train, acc_train)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.title('Training accuracy, evaluation')
plt.savefig('./figures/'+str(os.path.basename(args.log).split(".")[0])+'_trainacc.png')

plt.figure()
plt.plot(epochs_agg, loss_agg)
plt.ylabel('Loss during training')
plt.xlabel('epochs')
plt.title('Training loss during training')

plt.figure()
plt.plot(epochs_test, acc_test)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.title('Testing accuracy, evaluation')
plt.savefig('./figures/'+str(os.path.basename(args.log).split(".")[0])+'_testacc.png')
'''
