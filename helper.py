import os
from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
from extract_bottleneck_features import *
from keras.applications.resnet50 import preprocess_input, decode_predictions
from glob import glob
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.resnet50 import ResNet50, preprocess_input
'''
Flask uses multiple threads. The problem you are running into is because the tensorflow model is not loaded and used in the same thread. One workaround is to force tensorflow to use the gloabl default graph .
'''
class DogModel:

  model = None
  graph = None
  def keras_resource(self):
        num_cores = 4

        if os.getenv('TENSORFLOW_VERSION') == 'GPU':
            num_gpu = 1
            num_cpu = 1
        elif os.getenv('TENSORFLOW_VERSION') == 'CPU':
            num_gpu = 0
            num_cpu = 1
        else:
          num_gpu = 0
          num_cpu = 1
            #raise Exception()#NonResourceException

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                                device_count={'CPU': num_cpu, 'GPU': num_gpu})
        config.gpu_options.allow_growth = True
        
        return config
  def __init__(self):
    #config = self.keras_resource()
    #session = tf.Session(config=config)
    #self.graph = session.graph
    #set_session(session)
    self.model = load_model('./model/model_Resnet50_final.h5')
    self.resnet50_model =ResNet50(weights='imagenet', include_top=False)
    self.graph = tf.get_default_graph()
    
    #self.graph = tf.get_default_graph()
    #self.graph = tf.Graph()
    self.dog_names=['Affenpinscher',
                    'Afghan_hound',
                    'Airedale_terrier',
                    'Akita',
                    'Alaskan_malamute',
                    'American_eskimo_dog',
                    'American_foxhound',
                    'American_staffordshire_terrier',
                    'American_water_spaniel',
                    'Anatolian_shepherd_dog',
                    'Australian_cattle_dog',
                    'Australian_shepherd',
                    'Australian_terrier',
                    'Basenji',
                    'Basset_hound',
                    'Beagle',
                    'Bearded_collie',
                    'Beauceron',
                    'Bedlington_terrier',
                    'Belgian_malinois',
                    'Belgian_sheepdog',
                    'Belgian_tervuren',
                    'Bernese_mountain_dog',
                    'Bichon_frise',
                    'Black_and_tan_coonhound',
                    'Black_russian_terrier',
                    'Bloodhound',
                    'Bluetick_coonhound',
                    'Border_collie',
                    'Border_terrier',
                    'Borzoi',
                    'Boston_terrier',
                    'Bouvier_des_flandres',
                    'Boxer',
                    'Boykin_spaniel',
                    'Briard',
                    'Brittany',
                    'Brussels_griffon',
                    'Bull_terrier',
                    'Bulldog',
                    'Bullmastiff',
                    'Cairn_terrier',
                    'Canaan_dog',
                    'Cane_corso',
                    'Cardigan_welsh_corgi',
                    'Cavalier_king_charles_spaniel',
                    'Chesapeake_bay_retriever',
                    'Chihuahua',
                    'Chinese_crested',
                    'Chinese_shar-pei',
                    'Chow_chow',
                    'Clumber_spaniel',
                    'Cocker_spaniel',
                    'Collie',
                    'Curly-coated_retriever',
                    'Dachshund',
                    'Dalmatian',
                    'Dandie_dinmont_terrier',
                    'Doberman_pinscher',
                    'Dogue_de_bordeaux',
                    'English_cocker_spaniel',
                    'English_setter',
                    'English_springer_spaniel',
                    'English_toy_spaniel',
                    'Entlebucher_mountain_dog',
                    'Field_spaniel',
                    'Finnish_spitz',
                    'Flat-coated_retriever',
                    'French_bulldog',
                    'German_pinscher',
                    'German_shepherd_dog',
                    'German_shorthaired_pointer',
                    'German_wirehaired_pointer',
                    'Giant_schnauzer',
                    'Glen_of_imaal_terrier',
                    'Golden_retriever',
                    'Gordon_setter',
                    'Great_dane',
                    'Great_pyrenees',
                    'Greater_swiss_mountain_dog',
                    'Greyhound',
                    'Havanese',
                    'Ibizan_hound',
                    'Icelandic_sheepdog',
                    'Irish_red_and_white_setter',
                    'Irish_setter',
                    'Irish_terrier',
                    'Irish_water_spaniel',
                    'Irish_wolfhound',
                    'Italian_greyhound',
                    'Japanese_chin',
                    'Keeshond',
                    'Kerry_blue_terrier',
                    'Komondor',
                    'Kuvasz',
                    'Labrador_retriever',
                    'Lakeland_terrier',
                    'Leonberger',
                    'Lhasa_apso',
                    'Lowchen',
                    'Maltese',
                    'Manchester_terrier',
                    'Mastiff',
                    'Miniature_schnauzer',
                    'Neapolitan_mastiff',
                    'Newfoundland',
                    'Norfolk_terrier',
                    'Norwegian_buhund',
                    'Norwegian_elkhound',
                    'Norwegian_lundehund',
                    'Norwich_terrier',
                    'Nova_scotia_duck_tolling_retriever',
                    'Old_english_sheepdog',
                    'Otterhound',
                    'Papillon',
                    'Parson_russell_terrier',
                    'Pekingese',
                    'Pembroke_welsh_corgi',
                    'Petit_basset_griffon_vendeen',
                    'Pharaoh_hound',
                    'Plott',
                    'Pointer',
                    'Pomeranian',
                    'Poodle',
                    'Portuguese_water_dog',
                    'Saint_bernard',
                    'Silky_terrier',
                    'Smooth_fox_terrier',
                    'Tibetan_mastiff',
                    'Welsh_springer_spaniel',
                    'Wirehaired_pointing_griffon',
                    'Xoloitzcuintli',
                    'Yorkshire_terrier']
  def predict(self,img_path):
    img = self.path_to_tensor(img_path)
    
    #return model_Resnet50.predict(preprocess_input(path_to_tensor(img_path)))
    # extract bottleneck features
    #bottleneck_feature = extract_Resnet50(img,self.graph)
    # obtain predicted vector
    #with graph.as_default():
    with self.graph.as_default():
      bottleneck_feature = self.resnet50_model.predict(preprocess_input(img))
      predicted_vector = self.model.predict(bottleneck_feature)
      return self.dog_names[np.argmax(predicted_vector)]
    #K.clear_session()
    result = np.where(predicted_vector == np.amax(predicted_vector))
    retVal = []
    for x in result:
        retVal.append(self.dog_names[x[0]])

    return retVal
    #print(np.argmax(predicted_vector))
    # return dog breed that is predicted by the model
    #return dog_names[np.argmax(predicted_vector)]
  def path_to_tensor(self,img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

  

