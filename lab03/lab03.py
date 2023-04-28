#code from: https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb

import dataloader 
# from draw import draw_picture
from model import EEGNet,DepConvNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def draw_picture(accuracy_train , accuracy_test,name):
    plt.figure(figsize=(12,6))
    activation_function_name = ["ELU", "ReLU"  , "LeakyReLU"]
    for index , line in enumerate(accuracy_train):
        epoch = [i for i in range(1,len(line)+1)]
        plt.plot( epoch , line , label = activation_function_name[index]+"_train")
        
    for index , line in enumerate(accuracy_test):
        epoch = [i for i in range(1,len(line)+1)]
        plt.plot( epoch , line , label = activation_function_name[index]+"_test")

    plt.legend(loc="lower right")
    plt.title(f"Activation Function Comparision({name})") # title
    plt.ylabel("Accuracy(%)") # y label
    plt.xlabel("Epoch") # x label
    # print(len(mean_score))
    plt.savefig(f"./results/{name}.jpg")
    plt.show()

def Accuracy(net , test_load):
    net.eval()
    for index , (inputs,labels) in enumerate(test_load):
        inputs, labels = inputs.to(device , dtype = torch.float), labels.to(device , dtype = torch.long)

        outputs = net(inputs)
        pred = outputs.argmax(dim = 1)
        same = 0
        for i in range(pred.size(dim=0)):
            if pred[i] == labels[i]:
                same+=1
    # print('accuracy is {:.2f}'.format(count/len))
    return (same/pred.size(dim=0))*100
    # print(roc_auc_score(labels, predicted))

def train(net , train_load , test_load, name , act_name):
    accuracy_Epochs_train = []
    accuracy_Epochs_test = []
    learning_rate=0.0007
    Epochs = 1000
    lose_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters() , lr=learning_rate, weight_decay=0.01)

    best_accuracy = 0
    best_state_dict = net.state_dict()

    net.train()
    for i in range(Epochs): 
        same = 0
        for _,(inputs,labels) in enumerate(train_load):

            inputs, labels = inputs.to(device , dtype = torch.float), labels.to(device , dtype = torch.long)
            # forward + backward + optimize
            outputs = net(inputs)
            loss = lose_function(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            #back-propagation
            loss.backward()
            #update weights
            optimizer.step()

            #test in training with trainind data
            pred = outputs.argmax(dim = 1)
            for i in range(pred.size(dim=0)):
                if pred[i] == labels[i]:
                    same+=1
        acc_train = (same/float(len(train_load.dataset)))*100
        accuracy_Epochs_train.append(acc_train)  
        # print(acc_train)
        #Test 
        acc_test = test(net  , test_load)
        accuracy_Epochs_test.append(acc_test)
        if acc_test > best_accuracy:
            best_accuracy = acc_test
            best_state_dict = net.state_dict()
    torch.save({
        'model_name' : name,
        'activation_function' : act_name,
        'model_state_dict' : best_state_dict
    }, "./models/{}_{}.pt".format(name,act_name))
    print(name+"_"+act_name+" test accuracy is : "+str(best_accuracy)) 
    return net,accuracy_Epochs_train , accuracy_Epochs_test


def test(net  , test_load):
    same = 0
    for _,(inputs,labels) in enumerate(test_load):
        inputs, labels = inputs.to(device , dtype = torch.float), labels.to(device , dtype = torch.long)
        # forward
        outputs = net(inputs)
        #test in training
        pred = outputs.argmax(dim = 1)
        for i in range(pred.size(dim=0)):
            if pred[i] == labels[i]:
                same+=1
    acc = (same/float(len(test_load.dataset)))*100
    return acc

def demo(net, test_load, name , act_name):
    checkpoint = torch.load("./models/{}_{}.pt".format(name,act_name))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    same = 0
    for _,(inputs,labels) in enumerate(test_load):
        inputs, labels = inputs.to(device , dtype = torch.float), labels.to(device , dtype = torch.long)
        # forward
        outputs = net(inputs)
        #test in training
        pred = outputs.argmax(dim = 1)
        for i in range(pred.size(dim=0)):
            if pred[i] == labels[i]:
                same+=1
    acc = (same/float(len(test_load.dataset)))*100
    print('{}_{} accuracy is {:.2f}'.format(name,act_name,acc))


if __name__ == '__main__':
    device = torch.device("cuda")
    #load Dataset
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_set = TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
    train_load = DataLoader(train_set , batch_size=1024 , shuffle=True )
    test_set = TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
    test_load = DataLoader(test_set , batch_size=1024 , shuffle=True )

    # network = EEGNet().to(device)
    activation_function = {"ELU" : nn.ELU(alpha=1.0) , "ReLU" :  nn.ReLU() , "LeakyReLU" : nn.LeakyReLU()}

    DeepConvNet_accuracy_train = []
    DeepConvNet_accuracy_test = []
    EEGNet_accuracy_train = []
    EEGNet_accuracy_test = []


    # Train
    for index , act_name in enumerate(activation_function.keys()):
        EEGNetwork = EEGNet(activation_function[act_name]).to(device)
        EEGNetwork,accuracy_Epochs_EEG_train , accuracy_Epochs_EEG_test = train(EEGNetwork,train_load,test_load , "EEGNet" , act_name )
        # print("Train EEGNet_{} End!".format(act_name))
        DeepConvnetwork = DepConvNet(activation_function[act_name]).to(device)
        DeepConvnetwork,accuracy_Epochs_Deep_train , accuracy_Epochs_Deep_test = train(DeepConvnetwork,train_load,test_load , "DepConvNet" , act_name)
        # print("Train DepConvNet_{} End!".format(act_name))
        
        
        # save train
        DeepConvNet_accuracy_train.append(accuracy_Epochs_Deep_train)
        EEGNet_accuracy_train.append(accuracy_Epochs_EEG_train)
        # save test
        DeepConvNet_accuracy_test.append(accuracy_Epochs_Deep_test)
        EEGNet_accuracy_test.append(accuracy_Epochs_EEG_test)
    print("End!")

    # Draw the pictrue
    draw_picture(EEGNet_accuracy_train , EEGNet_accuracy_test , "EEGNet")
    draw_picture(DeepConvNet_accuracy_train , DeepConvNet_accuracy_test , "DeepConvNet")

    # demo
    # for act_name in activation_function.keys():
    #     DeepConvnetwork = DepConvNet(activation_function[act_name]).to(device)
    #     demo(DeepConvnetwork , test_load ,"DepConvNet" , act_name )
    #     EEGNetwork = EEGNet(activation_function[act_name]).to(device)
    #     demo(EEGNetwork , test_load ,"EEGNet" , act_name )


