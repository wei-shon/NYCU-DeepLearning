import numpy as np

def generate_linear(n=100):
    import numpy as np 
    pts = np.random.uniform(0 , 1 , (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0] , pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs) , np.array(labels).reshape(n,1)

def generate_XOR_easy():
    import numpy as np 
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i , 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i , 1-0.1*i])
        labels.append(1)
    return np.array(inputs) , np.array(labels).reshape(21,1)

class Model():

    def __init__(self , inputdatas , labels , learning_rate = 0.1):
        # data
        self.inputdatas = inputdatas
        self.labels = labels

        # weights
        self.w1 = np.random.uniform(0 , 1 , (2,15))
        self.w2 = np.random.uniform(0 , 1 , (15,15))
        self.w3 = np.random.uniform(0 , 1 , (15,1))

        # learning rate
        self.learning_rate = learning_rate
        

    def sigmoid(self , x):
        return 1.0/(1.0 + np.exp(-x))

    def derivative_sigmoid(self , x):
        return np.multiply(self.sigmoid(x) , 1.0 - self.sigmoid(x))

    def lossfunction(self , y , y_pred):
        return np.mean(np.power( (y - y_pred),2) )
        
    def derivative_lossfunction(self ,  y , y_pred):
        return y_pred - y

    def forwardpropagation(self):
        h1 = np.matmul(self.inputdatas,self.w1)
        a1 = self.sigmoid(h1)
        h2 = np.matmul(a1,self.w2)
        a2 = self.sigmoid(h2)
        h3 = np.matmul(a2,self.w3)
        output = self.sigmoid(h3)
        return  h1,a1,h2,a2,h3,output

    def backpropagation(self,h1,a1,h2,a2,h3,outputdata):

        # w3 calculate
        der_loss = self.derivative_lossfunction(self.labels , outputdata)
        loss_w3 =  der_loss * self.derivative_sigmoid(h3)
        loss_w3 = np.matmul(a2.T ,  loss_w3)

        #w2 calculate
        loss_w2 = der_loss * self.derivative_sigmoid(h3)
        loss_w2 = np.matmul(loss_w2 ,  self.w3.T)
        loss_w2 = loss_w2 * self.derivative_sigmoid(h2)
        loss_w2 = np.matmul(a1.T ,  loss_w2)

        #w1 calculate
        loss_w1 = der_loss * self.derivative_sigmoid(h3)
        loss_w1 = np.matmul(loss_w1 ,  self.w3.T)
        loss_w1 = loss_w1 * self.derivative_sigmoid(h2)
        loss_w1 = np.matmul(loss_w1 ,  self.w2)
        loss_w1 = loss_w1 * self.derivative_sigmoid(h1)
        loss_w1 = np.matmul(self.inputdatas.T ,  loss_w1)
        # print(loss_w1)


        return loss_w3,loss_w2,loss_w1


    def train(self , epoch=10000):
        for i in range(epoch):
            h1,a1,h2,a2,h3,outputdata = self.forwardpropagation()
            loss = self.lossfunction(self.labels ,outputdata )
            loss_w3,loss_w2,loss_w1 = self.backpropagation(h1,a1,h2,a2,h3,outputdata)
            self.w1 = self.w1 - self.learning_rate*loss_w1
            self.w2 = self.w2 - self.learning_rate*loss_w2
            self.w3 = self.w3 - self.learning_rate*loss_w3
            # print(self.w1)
            if i%500==0:
                print("epoch {:5d}  loss : {:.10f}".format(i, loss))
                # print("loss w1:" ,loss_w1)
                # print("loss w2:" ,loss_w2)
                # print("loss w3:" ,loss_w3)

    def predict(self):
        h1,a1,h2,a2,h3,outputdata = self.forwardpropagation()
        same_count = 0
        output_to_integer = []
        for i in range(self.labels.shape[0]):
            if(outputdata[i,0] > 0.5):
                output_to_integer.append(1)
            else:
                output_to_integer.append(0)
            if self.labels[i,0] == output_to_integer[i]:
                same_count+=1
        for i in range(self.labels.shape[0]):
            print("Iter{:2d} |    Ground truth: {} |    prediction: {:.5f} |".format(i,self.labels[i,0],outputdata[i,0]))
        print("loss={:.8f} accuracy={}%".format(self.lossfunction(outputdata , self.labels) , same_count/len(self.labels)*100 ) )   
        return output_to_integer
    
    def show_result(self,x,y,pred_y):
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.title('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0] , x[i][1] , 'ro')
            else:
                plt.plot(x[i][0] , x[i][1] , 'bo')
        
        plt.subplot(1,2,2)
        plt.title('Predict result', fontsize = 18)
        for i in range(x.shape[0]):
            if pred_y[i] == 0:
                plt.plot(x[i][0] , x[i][1] , 'ro')
            else:
                plt.plot(x[i][0] , x[i][1] , 'bo')
        plt.show()

if __name__ == '__main__':

    epoch = 100000
    inputdatas , labels = generate_linear(n=100)
    model = Model(inputdatas,labels)
    model.train(epoch)
    # testing 
    predict = model.predict()
    model.show_result(inputdatas , labels , predict)


    inputdatas_xor , labels_xor = generate_XOR_easy()
    model = Model(inputdatas_xor,labels_xor)
    model.train(epoch)
    # testing 
    predict_xor = model.predict()
    model.show_result(inputdatas_xor , labels_xor , predict_xor)