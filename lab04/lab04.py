from ResNet50 import ResNet50
from ResNet18 import ResNet18
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(net , EPOCH  , name , Pre_Train = "No Pretraining" ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_Epochs_train = []
    accuracy_Epochs_test = []
    learning_rate=0.005
    Epochs = EPOCH
    lose_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params= net.parameters() ,  lr=learning_rate, momentum=0.9 , weight_decay=5e-4)
    Train_Load_IMG = RetinopathyLoader("./new_train/" , 'train')


    best_accuracy = 0
    best_state_dict = net.state_dict()
    net.train()
    for i in range(Epochs): 
        same = 0
        count = 0
        train_load = tqdm(DataLoader(Train_Load_IMG , batch_size=12 , num_workers=10))

        total_loss = 0
        for _,(inputs,labels) in enumerate(train_load):
            inputs, labels = inputs.to(device , dtype = torch.float), labels.to(device , dtype = torch.long)
            # forward + backward + optimize
            outputs = net(inputs)
            loss = lose_function(outputs, labels)
            total_loss+=loss
            # zero the parameter gradients
            optimizer.zero_grad()
            #back-propagation
            loss.backward()
            #update weights
            optimizer.step()

            #test in training with trainind data
            pred = outputs.argmax(dim = 1)
            for j in range(pred.size(dim=0)):
                count+=1
                if pred[j] == labels[j]:
                    same+=1
            # print(name+"_"+Pre_Train+" train loss is : "+str(loss)) 
        print("This Epoch loss is :"+str(total_loss))
        acc_train = (same/float(count))*100
        print("This Epoch "+str(i)+" "+name+"_"+Pre_Train+" train accuracy is : "+str(acc_train)) 
        accuracy_Epochs_train.append(acc_train)  
        # print(acc_train)
        #Test 
        acc_test = test(net )
        accuracy_Epochs_test.append(acc_test)
        if acc_test > best_accuracy:
            best_accuracy = acc_test
            best_state_dict = net.state_dict()
        print("This Epoch "+str(i)+" "+name+"_"+Pre_Train+" test accuracy is : "+str(best_accuracy)) 
    torch.save({
        'model_name' : name,
        'Pre_Train' : Pre_Train,
        'model_state_dict' : best_state_dict
    }, "./models/{}_{}_{:.2f}.pt".format(name , Pre_Train , best_accuracy))
    print(name+"_"+Pre_Train+" test accuracy is : "+str(best_accuracy)) 
    return net,accuracy_Epochs_train , accuracy_Epochs_test

def test(net ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Test_Load_IMG = RetinopathyLoader("./new_test/" , 'test')
    test_load = DataLoader(Test_Load_IMG , batch_size=8  , num_workers=8)
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

def demo(net, name, highest_acc , Pre_Train = "No Pretraining"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("./models/{}_{}_{}.pt".format(name,Pre_Train, highest_acc))
    net.load_state_dict(checkpoint['model_state_dict']) 
    net.eval()

    Test_Load_IMG = RetinopathyLoader("./new_test/" , 'test')
    test_load = DataLoader(Test_Load_IMG , batch_size=8  , num_workers=8)
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
    print('{}_{} accuracy is {:.2f}'.format(name,Pre_Train,acc))

def draw_picture(accuracy_train , accuracy_test , epochs ,name):
    plt.figure(figsize=(12,6))
    activation_function_name = ["w/o pretraining", "with pretraining"  ]
    for index , line in enumerate(accuracy_train):
        epoch = [i for i in range(1,epochs+1)]
        plt.plot( epoch , line , label = "Train" + activation_function_name[index])
        
    for index , line in enumerate(accuracy_test):
        epoch = [i for i in range(1,epochs+1)]
        plt.plot( epoch , line , label = "Test"+activation_function_name[index])

    plt.legend(loc="upper left")
    plt.title(f"Result Comparision({name})") # title
    plt.ylabel("Accuracy(%)") # y label
    plt.xlabel("Epochs") # x label
    # print(len(mean_score))
    plt.savefig(f"./results/{name}.jpg")
    plt.show()


def cofusion_matrix(net, name , Pre_Train = "No Pretraining"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("./models/{}_{}.pt".format(name,Pre_Train))
    net.load_state_dict(checkpoint['model_state_dict']) 
    net.eval()
    Test_Load_IMG = RetinopathyLoader("./new_test/" , 'test')
    test_load = DataLoader(Test_Load_IMG , batch_size=4 ,num_workers=8)
    prediction_list = []
    label_list = []

    for _,(inputs,labels) in enumerate(test_load):
        inputs, labels = inputs.to(device , dtype = torch.float), labels.to(device , dtype = torch.long)
        outputs = net(inputs)
        pred = outputs.argmax(dim = 1)
        pred= pred.to('cpu')
        labels=labels.to('cpu')
        for i in pred:
            prediction_list.append(i)
        for j in labels:
            label_list.append(j)
    disp = ConfusionMatrixDisplay.from_predictions(label_list , prediction_list , labels=[0,1,2,3,4] , normalize='true')
    # disp.plot()
    plt.title(f"{name}_{Pre_Train}_Normalized confusion matrix")
    plt.savefig(f"./results/{name}_{Pre_Train}_confusion matrix.jpg")
    plt.show()  

if __name__ == "__main__":
    #check gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set hyperparameter
    EPOCH_R18 = 10
    EPOCH_R18_PRETRAIN = 10
    EPOCH_R50 = 10
    EPOCH_R50_PRETRAIN = 10
    BATCH_SIZE = 4

    # save accuracy to draw a picture
    ResNet18_accuracy_train = []
    ResNet18_accuracy_test = []
    ResNet50_accuracy_train = []
    ResNet50_accuracy_test = []

    # training model
    
    R50_pretrain = ResNet50(3 , 5 , True).to(device)
    # R50_pretrain,accuracy_Epochs_ResNet50_pretrain_train , accuracy_Epochs_ResNet50_pretrain_test = train(R50_pretrain , EPOCH_R50_PRETRAIN , "ResNet50" , "Yes Pretraining" )

    R18_pretrain = ResNet18(3 , 5 , True).to(device)
    # R18_pretrain,accuracy_Epochs_ResNet18_pretrain_train , accuracy_Epochs_ResNet18_pretrain_test = train(R18_pretrain , EPOCH_R18_PRETRAIN , "ResNet18" , "Yes Pretraining" )

    R50 = ResNet50(3 , 5 , False).to(device)
    # R50,accuracy_Epochs_ResNet50_train , accuracy_Epochs_ResNet50_test = train(R50 , EPOCH_R50 , "ResNet50" , "No Pretraining" )

    R18 = ResNet18(3 , 5 , False).to(device)
    # R18,accuracy_Epochs_ResNet18_train , accuracy_Epochs_ResNet18_test = train(R18 , EPOCH_R18 , "ResNet18" , "No Pretraining" )


    # save train
    # ResNet18_accuracy_train.append(accuracy_Epochs_ResNet18_train)
    # ResNet18_accuracy_train.append(accuracy_Epochs_ResNet18_pretrain_train)
    # ResNet50_accuracy_train.append(accuracy_Epochs_ResNet50_train)
    # ResNet50_accuracy_train.append(accuracy_Epochs_ResNet50_pretrain_train)
    # save test
    # ResNet18_accuracy_test.append(accuracy_Epochs_ResNet18_test)
    # ResNet18_accuracy_test.append(accuracy_Epochs_ResNet18_pretrain_test)
    # ResNet50_accuracy_test.append(accuracy_Epochs_ResNet50_test)
    # ResNet50_accuracy_test.append(accuracy_Epochs_ResNet50_pretrain_test)


    # Draw the pictrue
    # draw_picture(ResNet18_accuracy_train , ResNet18_accuracy_test , EPOCH_R18 , "ResNet18")
    # draw_picture(ResNet50_accuracy_train , ResNet50_accuracy_test , EPOCH_R50 , "ResNet50")

    # cofusion_matrix(R18 , "ResNet18" , "No Pretraining")
    # cofusion_matrix(R50 , "ResNet50" , "No Pretraining")
    # cofusion_matrix(R18_pretrain , "ResNet18" , "Yes Pretraining")
    # cofusion_matrix(R50_pretrain , "ResNet50" , "Yes Pretraining")

    demo(R18_pretrain , "ResNet18", 82.16 , "Yes Pretraining")
    demo(R18 , "ResNet18" , 73.35 , "No Pretraining")
    demo(R50_pretrain , "ResNet50", 83.16 , "Yes Pretraining")
    demo(R50 , "ResNet50", 73.35 , "No Pretraining")


