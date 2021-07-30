import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2 
from glob import glob 
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import Recall,Precision
from model_copy import build_unet 
from metrics import dice_loss,dice_coef,iou 


H = 512 
W = 512 

def create_dir(path):
    if not os.path.exists(path):
        os.makdirs(path)

def load_data(path):
    x = sorted(glob(os.path.join(path,"image","*.jpg")))
    y = sorted(glob(os.path.join(path,"mask","*.jpg")))
    return x,y

def shuffling(x,y):
    x,y = shuffle(x,y,random_state = 42)
    return x,y





if __name__ == "__main__":
    '''Seeding'''
    np.random.seed(42)
    tf.random.set_seed(42)
    ''' Directory to save files '''
    create_dir("files")
    """Hyperparameters"""
    batch_size = 2
    lr = 1*e-4
    num_epochs = 100
    model_path = os.path.join("files","model.h5")
    csv_path = os.path.join("files","data.csv")

    """ Dataset """
    dataset_path = "new_data"
    train_path = os.path.join(dataset_path,"train")
    valid_path = os.path.join(dataset_path,"valid")

    train_x,train_y = load_data(train_path)
    train_x,train_y = shuffling(train_x,train_y)
    valid_x,valid_y = load_data(valid_path)
    print(f"Train:{len(train_x)}-{len(train_y)}")
    print(f"Valid:{len(valid_x)}-{len(valid_y)}")








