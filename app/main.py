import os
import json
from flask import Flask
from flask import render_template
from app.monet import MonetDataset

app_folder = os.path.dirname(__file__).replace('\\', '/')
template_folder = os.path.join(app_folder, 'templates')
static_folder = os.path.join(app_folder, 'static')

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
m = MonetDataset()

@app.route('/')
def hello_world():
    context = m.get_pair('next')
    return render_template('index.html', **context)

@app.route('/images')
def get_images():
    context = m.get_pair('next')
    return json.dumps(context)

if __name__ == '__main__':
    app.run()