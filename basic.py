from flask import Flask, render_template, request
import os
import cv2
from keras.models import load_model
import keras.models
from keras.models import Model
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import get_default_graph
from PIL import Image
import matplotlib.pyplot as plt
import scipy

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_model():
    global model, graph
    model = load_model("xray_model_final.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = get_default_graph()
    print("model_loaded")


def get_mobileNet(model):
    # get AMP layer weights
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # extract wanted output
    Plot_model = Model(inputs=model.input,
                       outputs=(model.layers[-4].output, model.layers[-1].output))
    return Plot_model, all_amp_layer_weights


def mobilenet_CAM(image, model, all_amp_layer_weights):
    # get filtered images from convolutional output + model prediction vector
    with graph.as_default():
        last_conv_output, pred_vec = model.predict(image)
    # change dimensions of last convolutional outpu tto 7 x 7 x 1024
    last_conv_output = np.squeeze(last_conv_output)
    # get model's prediction (number between 0 and 999, inclusive)
    pred = np.argmax(pred_vec, axis=1)
    # bilinear upsampling to resize each filtered image to size of original image
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1)  # dim: 224 x 224 x 1024
    # get AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, pred]  # dim: (1024,)
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224 * 224, 1024)), amp_layer_weights).reshape(224, 224)  # dim: 224 x 224
    # return class activation map
    return final_output


def plot_mobilenet_CAM(image, image1, ax, model, all_amp_layer_weights):
    # load image, convert BGR --> RGB, resize image to 224 x 224,

    # plot image
    ax.imshow(image, alpha=0.5)
    # get class activation map
    CAM = mobilenet_CAM(image1, model, all_amp_layer_weights)
    # plot class activation map
    ax.imshow(CAM, cmap="jet", alpha=0.5)
    return CAM


get_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/xray/')
    if not os.path.isdir(target):
        os.mkdir(target)
    filename = ""
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)

    image = cv2.resize(cv2.cvtColor(cv2.imread(destination), cv2.COLOR_BGR2RGB), (224, 224))
    image1 = preprocess_input(image)
    image1 = np.expand_dims(image1, axis=0)
    with graph.as_default():
        pred = model.predict(image1)
    prediction = int(np.argmax(pred, axis=1)[0])
    if prediction:
        plot_dest = "/".join([target, "result.png"])
        mob_model, all_amp_layer_weights = get_mobileNet(model)
        fig, ax = plt.subplots()
        plot_mobilenet_CAM(image, image1, ax, mob_model, all_amp_layer_weights)
        plt.savefig(plot_dest, bbox_inches='tight')

    return render_template("result.html", prediction=prediction, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
