import tensorflow
print(tensorflow.__version__)

import pandas as pd
import numpy as np
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%matplotlib inline

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import model_from_json

CLASS_NAME_CSV_PATH = "/opt/model_data/names.csv"
TRAIN_ANNOTATION ="/opt/model_data/anno_train.csv"
TEST_ANNOTATION ="/opt/model_data/anno_test.csv"

CLASSIFY_MOBILE_JSON = "/opt/model_data/mobile_class.json"
CLASSIFY_MOBILE_WEIGHT = "/opt/model_data/mobile_class.h5"

BB_MOBILE_JSON = "/opt/model_data/mobile_bb.json"
BB_MOBILE_WEIGHT = "/opt/model_data/mobile_bb.h5"

# Initialize the random number generator
import random
random.seed(0)
# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

def get_class_bb_maping(img_name):
    class_names_data = pd.read_csv(CLASS_NAME_CSV_PATH, header=None, names=["class_name"])
    train_annotate = pd.read_csv(TRAIN_ANNOTATION, header=None, names=["image_file","x0","y0","x1","y1","class"])
    test_annotate = pd.read_csv(TEST_ANNOTATION, header=None, names=["image_file","x0","y0","x1","y1","class"])
    train_annotate['class_name'] = train_annotate.apply(lambda x: class_names_data['class_name'][(x['class']-1)], axis=1)
    test_annotate['class_name'] = test_annotate.apply(lambda x: class_names_data['class_name'][(x['class']-1)], axis=1)
    row = test_annotate.loc[test_annotate['image_file'] == img_name]

    if row is None:
        row = train_annotate.loc[train_annotate['image_file'] == img_name]
        test_check = False
    else:
        test_check = True

    if row is not None:
       if test_check:
         row = test_annotate.iloc[row.index[0]]
       else:
         row = train_annotate.iloc[row.index[0]]

    return row
    
def show_image_with_bb(df,img_path):
    img = cv2.imread(img_path,1)
    x0=df.x0
    x1=df.x1
    y0=df.y0
    y1=df.y1

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((x0,y0), (x1-x0),(y1-y0),
                          linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title(df.class_name)
    return fig

def show_predicted_image_with_bb(img_path, bb_data):
    img = cv2.imread(img_path,1)
    x0 = bb_data[0][0]
    y0 = bb_data[0][1]
    x1 = bb_data[0][2]/224*img.shape[0]
    y1 = bb_data[0][3]/224*img.shape[1]
    

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((x0,y0), (x1-x0), (y1-y0),
                          linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    return fig


def process_image(file_path):
    img = cv2.UMat(cv2.imread(file_path, 1)).get()
    img_resized = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
    processed_img = preprocess_input(np.array(img_resized, dtype=np.float32)).reshape(1,224,224,3)

    return processed_img

def get_classify_mobile():
    return get_model_defination(CLASSIFY_MOBILE_JSON, CLASSIFY_MOBILE_WEIGHT)

def get_bb_mobile():
    return get_model_defination(BB_MOBILE_JSON, BB_MOBILE_WEIGHT)

def get_model_defination(json_path, weight_path):
    json_file_model = open(json_path,'r').read()
    model = model_from_json(json_file_model)
    model.load_weights(weight_path)
    return model

def retrieve_class_name(predict_data):
    sorted_class = np.argsort(-predict_data)[0][:5]
    class_names_data = pd.read_csv(CLASS_NAME_CSV_PATH, header=None, names=["class_name"])
    class_names_percentage = []
    for iter in sorted_class:
        data = []
        data.append(class_names_data.iloc[iter][0])
        data.append(predict_data[0][iter])
        class_names_percentage.append(data)

    return class_names_percentage
