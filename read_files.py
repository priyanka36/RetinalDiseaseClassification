import os
import numpy as np 
import cv2 
from glob import glob 
from tqdm import tqdm 
import imageio

from albumentations import HorizontalFlip,VerticalFlip,ElasticTransform,GridDistortion,OpticalDistortion,CoarseDropout


def create_dir(path):
    if not os.path.exists(path):
        
        os.makedirs(path)
        


def load_data(path):

    """ X = images and Y = masks"""

    train_X = sorted(glob(os.path.join(path,"training","images","*.tif")))
   
    train_Y = sorted(glob(os.path.join(path,"training","1st_manual","*.gif")))

    test_x = sorted(glob(os.path.join(path,"test","images","*.tif")))
    test_y = sorted(glob(os.path.join(path,"test","1st_manual","*.gif")))
    
  
    return(train_X,train_Y),(test_x,test_y)

def augment_data(images,masks,save_path,augment=False):

    H= 512
    W= 512

    for idx,(x,y) in tqdm(enumerate(zip(images,masks)),total = len(images)):

        """ Extracting names"""
        print(x,y)
        names = str(x.split("/")[-1].split(".")[0])
        extension = ".jpg"
        x = cv2.imread(x,cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        print(type(y))
        y = np.asarray(y)
       

        print(x.shape,y.shape)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x,mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
          
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x,mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x,mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug = OpticalDistortion(p=1.0)
            augmented = aug(image=x,mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            X = [x,x1,x2,x3,x4]
            Y = [y,y1,y2,y3,y4]
            print(len(X))
            print(len(Y))
            
        else:
            X=[x]
            Y=[y] 
        
        index = 0
        
        
        for i,m in zip(X,Y):
            

            i = cv2.resize(i,(W,H))
            m = cv2.resize(m,(W,H))
            
            
            if len(X) == 1:
                
                tmp_image_name = f"{names}.jpg"
                tmp_mask_name = f"{names}.jpg"
            
            else:
                tmp_image_name= f"{names}_{index}.jpg"
                tmp_mask_name= f"{names}_{index}.jpg"
            
            
            
            
            image_path = os.path.join(save_path,"images",tmp_image_name)
            test_or_train = str(image_path.split("/")[1])
            
            
            mask_path = os.path.join(save_path,"mask",tmp_mask_name)
            if test_or_train == "test":
                cv2.imwrite(f"newdata/{test_or_train}/images/{tmp_image_name}",i)
                cv2.imwrite(f"newdata/{test_or_train}/mask/{tmp_mask_name}",m)
            elif test_or_train == "train" :
                cv2.imwrite(f"newdata/{test_or_train}/images/{tmp_image_name}",i)
                cv2.imwrite(f"newdata/{test_or_train}/mask/{tmp_mask_name}",m)
                
            
            
            index += 1





if __name__ == "__main__":
    """Seeding"""
    np.random.seed(42)

    """Load the data"""
    data_path="/home/priyanka/RetinaBloodVessel/dataset"
    (train_X,train_Y),(test_x,test_y)=load_data(data_path)
    create_dir("newdata/train/images")
    create_dir("newdata/train/mask")
    create_dir("newdata/test/images")
    create_dir("newdata/test/mask")

    
    augment_data(test_x, test_y, f"new_data/test/", augment=False)
    augment_data(train_X,train_Y,f"new_data/train/",augment=True)
    

