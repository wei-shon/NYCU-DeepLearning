import pandas as pd
from torch.utils import data
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        # print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        #step 1
        path = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(path)
        # img.save("test.jpg")

        #step2
        label = self.label[index]

        #step3
        # print(img.size)
        img = self.handleImage(img)
        # print(img.size)
        # img.save("test1.jpg")

        #transfer image to tensor
        resize = transforms.Resize([512,512])
        totensor = transforms.ToTensor()
        img = resize(img)
        # print (img.size)
        # print()
        # img.save("test3.jpg")
        img = totensor(img)
        return img, label
    
    def handleImage(self,img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size

        left = self.boundaryFinder(img, 0, width/2, self.vCheck)
        right = self.boundaryFinder(img, width-1, width/2, self.vCheck)
        top = self.boundaryFinder(img, 0, height/2, self.hCheck)
        bottom = self.boundaryFinder(img, height-1, width/2, self.hCheck)

        rect = (left,top,right,bottom)
        # print(rect)
        region = img.crop(rect)
        return region

    def hCheck(self,img, y, step = 50):
        count = 0
        width = img.size[0]
        for x in range(0, width, step):
            if self.isCrust(img.getpixel((x, y))):
                count += 1
            if count > width / step :
                return True
        return False

    def vCheck(self,img, x, step = 50):
        count = 0
        height = img.size[1]
        for y in range(0, height, step):
            if self.isCrust(img.getpixel((x, y))):
                count += 1
            if count > height / step :
                return True
        return False

    def isCrust(self,pix):
        return sum(pix) < 25


    def boundaryFinder(self,img,crust_side,core_side,checker):
        if not checker(img,crust_side):
            return crust_side
        if checker(img,core_side):
            return core_side

        mid = (crust_side + core_side) / 2
        while  mid != core_side and mid != crust_side:
            if checker(img,mid):
                crust_side = mid
            else:
                core_side = mid
            mid = (crust_side + core_side) / 2
        return core_side




if __name__ == "__main__":
    Train_Load_IMG = RetinopathyLoader("./new_train/" , 'train')
    # Test_Load_IMG = RetinopathyLoader("./new_test/" , 'test')
    # for i in range(Train_Load_IMG.__len__()):
    train_img , train_label = Train_Load_IMG.__getitem__(1)
    # test_img , test_label = Test_Load_IMG.__getitem__(0)