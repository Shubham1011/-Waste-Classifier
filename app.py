
# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template , redirect ,url_for
from tensorflow.keras.preprocessing import image
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads') # you'll need to create a folder named uploads
model=load_model('wasteclassifier.h5')
graph = tf.get_default_graph()
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB
image_shape = (196,262,3)
result=""
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        fileaddress=basedir+'\\uploads\\'+filename
        result=pred(fileaddress)
        if(result==0):
            return redirect(url_for('organic'))
        else:
            return redirect(url_for('recyclable'))
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url,)

def pred(fileurl):
    global graph
    with graph.as_default():
        session = keras.backend.get_session()
        init = tf.global_variables_initializer()
        session.run(init)
        my_image = image.load_img(fileurl,target_size=image_shape)
        my_image = image.img_to_array(my_image)
        my_image = np.expand_dims(my_image, axis=0)
        pred=model.predict_classes(my_image)
        return pred
        
@app.route('/organic',methods=['GET'])    
def organic():
    return render_template('organic.html')

@app.route('/recyclable',methods=['GET'])
def recyclable():
    return render_template('recyclable.html')
    

if __name__ == '__main__':
    app.run()