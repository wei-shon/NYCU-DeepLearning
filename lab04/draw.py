import matplotlib.pyplot as plt

def draw_picture(accuracy_train , accuracy_test,name):
    activation_function = ["ELU", "ReLU"  , "LeakyReLU"]
    for index , line in enumerate(accuracy_train):
        epoch = [i for i in range(1,len(line)+1)]
        plt.plot( epoch , line , label = activation_function[index]+"_train")
        
    for index , line in enumerate(accuracy_test):
        epoch = [i for i in range(1,len(line)+1)]
        plt.plot( epoch , line , label = activation_function[index]+"_test")

    plt.legend(loc="lower right")
    plt.title(f"Activation Function Comparision({name})") # title
    plt.ylabel("Accuracy(%)") # y label
    plt.xlabel("Epoch") # x label
    # print(len(mean_score))
    plt.show()
    plt.savefig(f"./results/{name}.jpg")

if __name__ == "__main__":
    accuracy_train =  [[1 , 2 , 3] , [2 , 3, 4] , [8 , 9 , 10] ] 
    accuracy_test =  [  [5 ,6, 7] , [3 , 4 , 5]  , [3 , 4 , 5]]  
    draw_picture(accuracy_train ,accuracy_test,"EEGNet" )
