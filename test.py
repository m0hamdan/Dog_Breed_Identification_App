from sklearn.datasets import load_files 
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
from glob import glob
from helper import DogModel


model = DogModel()
print(model.detect_dog_human('./static/images/Afghan_hound_3.jpeg'))
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
   
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    
    return dog_files, dog_targets
