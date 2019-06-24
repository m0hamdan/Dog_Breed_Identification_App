import os
from flask import Flask
from flask import url_for, redirect, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired,FileAllowed
from datetime import datetime
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from helper import DogModel
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

###

app = Flask(__name__)
#app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'I have a dream222 333'
#app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()

#photos = UploadSet('photos', IMAGES)
#configure_uploads(app, photos)
#patch_request_class(app)  # set maximum file size, default is 16MB

model  = DogModel()

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(['jpg', 'png'], u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = UploadForm()
    file_path = './static/images/Airedale_terrier_00163.jpg'
   
    if form.validate_on_submit():

        #filename = photos.save(form.photo.data)
        #file_url = photos.url(filename)
        if(True):
            f = form.photo.data
            filename = secure_filename(f.filename)
            #name = hashlib.md5('admin' + str(time.time())).hexdigest()[:15]
            file_path = os.path.join(
                'static', 'images', filename
            )
            f.save(file_path)
        
        #model  = DogModel()
        #full_filename = os.path.join('uploads', 'Airedale_terrier_00164.jpg')
        prediction = model.predict('./static/images/Anatolian_shepherd_dog_00665.jpg')#'./static/images/Airedale_terrier_00163.jpg'
        #return redirect(url_for('prediction'))
        return render_template('predict.html', form=form,file_path=file_path,prediction=prediction)

    return render_template('predict.html', form=form)

@app.route("/")
@app.route("/hello/<name>")
def home(name = None):
    #return redirect(url_for('upload'))

 
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")

if __name__ == "__main__":
  app.run( port=8000, debug=False, host='localhost')


