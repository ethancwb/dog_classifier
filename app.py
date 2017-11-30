from flask import Flask
from flask import request
from flask import render_template
import dog_app
import os, time
app = Flask(__name__)

UPLOAD_FOLDER = ('static/upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('homepage.html')

@app.route('/result', methods=['POST'])
def get_image():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    name = '../static/upload/' + file.filename
    dogImage = 'static/upload/' + file.filename
    breed = dog_app.classifier(dogImage)
    return render_template('result.html', filename=name, breed=breed)