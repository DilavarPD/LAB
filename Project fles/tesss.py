
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image




# Model saved with Keras model.save()
MODEL_PATH = 'model_inception_new.h5'

# Load your trained model
model = load_model(MODEL_PATH)




img = image.load_img("D:\\Main_Project\\print_predict\\print_predict\\3_o_1_n_6.jpg",target_size=(224, 224))

# Preprocessing the image
x = image.img_to_array(img)
# x = np.true_divide(x, 255)
## Scaling
x = x / 255
x = np.expand_dims(x, axis=0)

# Be careful how your trained model deals with the input
# otherwise, it won't make correct prediction!
# x = preprocess_input(x)

preds = model.predict(x)
print(preds)

maxElement = np.amax(preds)
print(maxElement)

if (maxElement >= .96):
    preds = np.argmax(preds, axis=1)
    if preds == 0 :
        print("ID1")
    elif preds == 1 :
        print("ID2")
    elif preds == 2 :
        print("ID3")

else:
    print("false ID")












