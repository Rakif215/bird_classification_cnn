from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
from src.model import build_model  # This should import your model-building function
from src.helper_functions import load_and_prep_image, pred_and_plot

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'path_to_your_model.h5'
DEVICE = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'

# Load the model
model = build_model(num_classes=525)  # Adjust number of classes as needed
model.load_weights(MODEL_PATH)
model.to(DEVICE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = load_and_prep_image(file_path)  # Pass the path to the function
        prediction = pred_and_plot(model, img, class_names=['List', 'Of', 'Class', 'Names'])

        return render_template('result.html', prediction=prediction, image_url=url_for('static', filename='uploads/' + filename))

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
