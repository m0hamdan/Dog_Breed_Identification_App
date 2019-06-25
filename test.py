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
    #print(data['target'][:5]) #[94 56 87  7  7]
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    #print(dog_targets[:5])
    return dog_files, dog_targets
   # load train, test, and validation datasets
#train_files, train_targets = load_dataset('./data/dog_images/train')
#valid_files, valid_targets = load_dataset('./data/dog_images/valid')
#test_files, test_targets = load_dataset('./data/dog_images/test')

# load list of dog names
##dog_names = [item[27:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
#print('There are %d total dog categories.' % len(dog_names))
#print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
#print('There are %d training dog images.' % len(train_files))
#print('There are %d validation dog images.' % len(valid_files))
#print('There are %d test dog images.'% len(test_files))