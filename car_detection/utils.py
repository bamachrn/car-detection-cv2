import tensorflow
print(tensorflow.__version__)

import pandas as pd
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%matplotlib inline

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import model_from_json

CLASS_NAME_CSV_PATH = "/opt/model_data/names.csv"

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
