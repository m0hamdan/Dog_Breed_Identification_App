# Dog Breed Classifier

<br/>

### Table of Contents

1. [Project Motivation](#motivation)
2. [Dependencies](#depend)
3. [Datasets](#data)
4. [Content](#files)
5. [Analysis](#analysis)
5. [Instructions](#instructions)
6. [Licensing](#licensing)
7. [Acknowledgements](#ack)
8. [Demo](#demo)

### Project Motivation:<a name="motivation"></a>

This project is part of Udacity Data Scientist Nanodegree program and I'll develop an CNN algorithm that will accept any dog image and return an estimate of the dog's breed. If human is detected, it will provide an estimate of the most resembling dog breed. If neither dog or human face is detected the algorithm will return an error.

The project consitutes of 2 parts:
1. The Jupyter notebook which includes all of the implemented algorithms and approaches
2. Flask we app which uses the saved model created through the jupyter notebook to perform the dog breed identification on uploaded photos


The following libraries were used in building the dog breed identification algorithm:
1. Keras (Tensorflow high level API)
2. OpenCV (Computer vision library)
3. face_detection library


### Dependencies <a name="depend"></a>
refer to requirements.txt

### Datasets <a name="data"></a>
1. Dog dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
2. Human dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip


### Content <a name="files"></a>
1. dog_app.ipynb
        - A jupyter notebook file that contains the god breed identification algorithm, here will find the source code of the algorithms
2. model directory:
        - model_Resnet50_final.h5: The saved model that contains the model architecture and weight, created using the code in the jupyter notebook
3. static
        - contains static images used by the Flask web app
4. templates
        - the html templates used for the Flask web application(master.html and predict.html)
5. helper.py
        - A python file contains the dog or human identification algorithm, it loads the saved model from the model folder
6. app.py
        - this is the main flask python file that is responsible for routing and rendering html templates
7. requirements.txt
        - lists all the dependecies
8. test.py
        - a test file to test the algorithm locally that will instantiate the model and try to predict the breed of a supplied image

### Analysis <a name="analysis"></a>
1. Face detection:

    To detect whether the supplied image is a human face I've compared OpenCV's implementation of Haar feature-based cascade classifiers accuracy with that of [face_recognition](https://pypi.org/project/face_recognition/). On a sample data of 100 human face and 100 dog,  OpenCV was able to dectect 100% of the human faces however 11% of the dog images were classified as human faces. on the other hand, face_recognition library  provides 2 methods for face dectection, HOG-based and using deep learning.

    The HOG-based model identified 100 faces of the supplied human faces sample and 10 faces of the dog sample dataset. 1% better than OpenCV.
    The deep learning model of the face_recognition library was able to detect 100 faces of the supplied human sample dataset and 1 face of the dog data set which is a remarkable accuracy compared to the other methods.

    Eventually we would use the latter model on a GPU enabled system, and OPenCV or HOG-based model where GPU is not available for the sake of performance.
    In the notebook I'm going to use the HOG-based model.

2. Dog detection:

    To detect whether a dog is present in the supplied image ReseNet-50 pre-trained model was used which can identify 1000 categories.
    Trying his model on a sample dog dataset gave an excellent result, it was able to detect dogs in all of the sample set and also didn't mistakenly identify dogs in any of the human sample dataset.

3. Dog breed classification:

    In this part, after a dog is detected in an image we want to be able to identify it's breed.

    The following models were applied to find out the best approach:

    - Create a CNN to classify dog breed from scratch 
        Training a CNN model that's built from scratch gave us a test accuracy of 8.9% which is better than a random guess but still no an acceptable result.
        ![model architecture](/screenshots/scratch_arch.PNG)

    - Use a CNN to classify Dog Breeds
        Here we are going to train a CNN using transfer learning using the pre-trained VGG-16 model where the last convolutional output of the model is fed as input to our model.
        So we only need to train the fully connected layer that we added to the VGG-16 model and this resulted in reduced training time without sacrificing accuracy.

        The achieved test accuracy was ~43%, a big mprovement compared to the model we built from scratch.

    ![model architecture](/screenshots/VGG16_arch.PNG)

    - Create a CNN to classify dog breeds using Transfer Learning
        Similar to the above step we will use transfer learning to a CNN model but we'll try to achieve at least 60% accuracy on the test data.
        For this purpose we are going to use Resnet50 pre-trained model and modify it's architecture by adding our custom fully connected layer.

        After several test I was able to reach accuracy of ~81% which is a huge improvement and acceptable metric for his project.

        ![model architecture](/screenshots/Resnet50_arch.PNG)

4. Algorithm

    This section will build on the algorithm built previously to determine whether the supplied image contains a human, dog, or neither. If a dog is detected, return the predicted breed, if a human is detected, return the resembling dog and if neither is detected, provide output that indicates an error.

5. Testing

    Testing the algorthim seems to work as expected, not all dog breeds were detected, also it can't detect multiple breeds in a single image.

    ![Image description](/screenshots/detection.PNG)

    To improve our results, the below points can be considered:
    - Increase the training set size
    - Apply data augmentation
    - Use Regional-CNN or YOLO algorithms to detect multiple objects in a single image





### Instructions <a name="instructions"></a>

1. Clone the repository: git clone https://github.com/m0hamdan/Dog_Breed_Identification_App.git
2. Uncomment lines 54 and 55 in app.py, this is only required if running locally
3. Run the following command in the repo root directory: python -m flask run
4. Go to http://localhost:8000


### Licensing <a name="licensing"></a>
None

### Acknowledgements <a name="ack"></a>
1. Udacity for providing the dogs images and jupyter notebook walkthrough

### Demo <a name="demo"></a>
https://dog-breed.azurewebsites.net

