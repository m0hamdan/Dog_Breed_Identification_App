# Dog Breed Classifier

<br/>

### Table of Contents

1. [Dependencies](#depend)
2. [Project Motivation](#motivation)
3. [Content](#files)
4. [Instructions](#instructions)
5. [Licensing](#licensing)
6. [Acknowledgements](#ack)

### Project Motivation:<a name="motivation"></a>

Write an CNN Algorithm for a Dog Identification App, given an image of a dog, the algorithm will identify an estimate of the canine’s breed

### Dependencies <a name="depend"></a>
refer to requirements.txt

### Content: <a name="files"></a>
1. ** model directory:**
        *model_Resnet50_final.h5: A saved model that contains the model architecture and weight, created using the code in the jupyter notebook
2. ** static
        * containes static images
3. ** templates
        * the html templates (master.html and predict.html)
4. ** dog_app.ipynb
        * A jupyter notebook file that contains the god breed identification algorithm

### Instructions:<a name="instructions"></a>

1. Clone the repository: git clone https://github.com/m0hamdan/Dog_Breed_Identification_App.git
2. Uncomment lines 54 and 55 in app.py, this is only required if running locally
3. Run the following command in the repo root directory: python -m flask run
4. Go to http://localhost:8000


### Licensing <a name="licensing"></a>
None

### Acknowledgements <a name="ack"></a>
1. Udacity for providing the dogs images

