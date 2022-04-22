import tensorflow as tf

from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

import numpy as np
from keras.models import load_model
import cv2

# INIT
#
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
session = tf.compat.v1.Session(config=config)

graph = tf.compat.v1.get_default_graph()

# load model


class_name = ['Bắp', 'Bắp cải', 'Bí ngô', 'Bí xanh', 'Bông cải xanh', 'Bơ', 'Cà chua', 'Cà rốt',
               'Cà tím', 'Cá hồi', 'Cá ngừ', 'Cải bó xôi',
               'Cải trắng', 'Chanh', 'Chuối', 'Củ cải trắng', 'Củ hành', 'Dâu', 'Dưa leo', 'Dứa', 'Gạo',
               'Giá đỗ', 'Gừng', 'Hành tím', 'Hàu',
               'Hạnh nhân', 'Hạt sen', 'Khế', 'Khoai lang', 'Khoai môn', 'Khoai tây', 'Khổ qua', 'Kiwi',
               'Lựu', 'Măng', 'Măng tây', 'Mướp', 'Mực',
               'Óc chó', 'Ớt', 'Ớt chuông', 'Rau muống', 'Rau mùi tây', 'Su hào', 'Su su', 'Súp lơ',
               'Tàu hủ', 'Táo', 'Thanh long', 'Thịt gà',
               'Thịt bò', 'Thịt dê', 'Thịt heo', 'Tôm', 'Tỏi', 'Trứng', 'Xà lách', 'Xoài', 'Đậu bắp',
               'Đậu phộng', 'Đậu xanh', 'Đu đủ']

with session.as_default():
    with graph.as_default():
        my_model = load_model("fruit_and_veg_model_new2.h5",compile=False)



app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['GET', 'POST'])
@cross_origin(origins='*')
def index():
    return "Server running"


@app.route('/upload', methods=['GET', 'POST'])
@cross_origin(origins='*')
def upload():
    global session, graph, my_model
    f = request.files['file']

    if f is not None:
        image = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)

        image = cv2.resize(image, dsize=(256, 256))

        image = np.asarray(image, dtype=np.float32)
        # normalizing the image
        image = image / 255
        # reshaping the image in to a 4D array
        image = image.reshape(-1, 256, 256, 3)

        # making prediction of the model

        with session.as_default():
            with graph.as_default():
                prediction = []
                predict = my_model.predict(image)
                #getting the index corresponding to the highest value in the prediction
                predict = np.argmax(predict)
                prediction.append(class_name[predict])

        print("This is", prediction[0])

        return prediction[0]
    else:
        return "No image found"


if __name__ == "__main__":
    app.run(debug=True, host='192.168.1.67', port=8080)
