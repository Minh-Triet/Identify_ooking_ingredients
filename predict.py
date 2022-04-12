
import tensorflow as tf

from flask import Flask
from flask import request,jsonify
from flask_cors import CORS,cross_origin

import numpy as np
from keras.models import load_model
import cv2
import sys

#INIT

config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
session = tf.compat.v1.Session(config=config)

graph = tf.compat.v1.get_default_graph()

#load model

class_name=['Bắp', 'Bắp cải', 'Bí ngô', 'Bí xanh', 'Bông cải xanh', 'Bơ', 'Cà chua', 'Cà rốt', 'Cà tím', 'Cá hồi', 'Cá ngừ', 'Cải bó xôi', 'Cải trắng', 'Chanh', 'Chuối', 'Củ cải trắng', 'Củ hành', 'Dâu', 'Dưa leo', 'Dứa', 'Gừng', 'Hành tím', 'Hàu', 'Hạnh nhân', 'Khế', 'Khoai lang', 'Khoai tây', 'Khổ Qua', 'Kiwi', 'Lựu', 'Óc chó', 'Ớt', 'Ớt chuông', 'Su Hào', 'Súp lơ', 'Táo', 'Thanh long', 'Thịt bò', 'Thịt heo', 'Tôm', 'Tỏi', 'Xà lách', 'Xoài', 'Đậu xanh', 'Đu đủ']


with session.as_default():
    with graph.as_default():
        my_model=load_model("fruit_n_veg_model.h5")

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS']='Content-Type'

@app.route('/upload',methods=['GET', 'POST'])
@cross_origin(origins='*')
def index():
    return "Server running"

@app.route('/upload',methods=['GET', 'POST'])
@cross_origin(origins='*')
def upload():
    global session,graph, my_model
    f= request.files['file']
    image= cv2.imdecode(np.fromstring(f.read(),np.uint8),cv2.IMREAD_COLOR)

    image = cv2.resize(image,dsize=(224,224))

    image=np.expand_dims(image,axis=0)

    with session.as_default():
        with graph.as_default():
            predict=my_model.predict(image)

    print("This is",class_name[np.argmax(predict)])
    
    return class_name[np.argmax(predict)]

if __name__ == "__main__":
    app.run()