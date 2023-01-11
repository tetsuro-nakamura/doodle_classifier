# -*- coding: utf-8 -*-

import argparse
import cv2
import json
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import load_model
from keras.utils.np_utils import to_categorical

# path of the directory in which the datasets are present
DATA_PATH = './quickdraw_data/'
# path where the model and label map will be saved
MODEL_PATH = './model/doodle_classifier.h5'
LABEL_PATH = './model/label.json'

# global parameters for drawing app
drawing = False
last_x, last_y = None, None
draw_img = np.zeros((200, 200, 3), np.uint8)


def create_cnn_model(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_test_train_data(data_path):
    # get dataset files (.npy)
    filenames = os.listdir(data_path)

    xtotal = []
    ytotal = []
    # setting value to be 200000 for the final dataset
    slice_train = int(200000 / len(filenames))
    seed = np.random.randint(1, 10e7)
    # maps integer values of labels to their corresponding strings labels
    label_map = {}

    i = 0
    for fname in filenames:
        if '.npy' not in fname:
            continue
        x = np.load(data_path + fname)
        x = x.astype('float32') / 255  # scale from 0-255 to 0-1
        label_name = fname.split('.npy')[0]
        y = [str(label_name)] * len(x)

        label_map[i] = label_name

        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        x = x[:slice_train]
        y = y[:slice_train]

        if i == 0:
            xtotal = x
            ytotal = y
        else:
            xtotal = np.concatenate([x, xtotal], axis=0)
            ytotal = np.concatenate([y, ytotal], axis=0)

        i += 1

    x_train, x_test, y_train, y_test = train_test_split(
        xtotal, ytotal, test_size=0.2, random_state=7)

    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)

    return x_train, x_test, y_train, y_test, label_map


def get_test_train_features_and_labels(x_train, x_test, y_train, y_test, input_shape):
    # reshape the features according to the format:
    #     train_features - final features to train the model on
    #     train_labels   - final labels for training data
    #     test_features - final features for evaluation of the model
    #     test_labels - final labels for test data
    train_features = x_train.reshape(
        x_train.shape[0], input_shape[0], input_shape[1], input_shape[2]
    ).astype('float32')
    test_features = x_test.reshape(
        x_test.shape[0], input_shape[0], input_shape[1], input_shape[2]
    ).astype('float32')
    # One-hot encode labels
    encoder = LabelEncoder()
    encoded_y_train = encoder.fit_transform(y_train)
    encoded_y_test = encoder.fit_transform(y_test)
    y_train_categorical = to_categorical(encoded_y_train)
    y_test_categorical = to_categorical(encoded_y_test)
    train_labels = y_train_categorical
    test_labels = y_test_categorical
    return train_features, train_labels, test_features, test_labels


def train_and_evaluate(data_path, model_path, label_path):
    x_train, x_test, y_train, y_test, label_map = get_test_train_data(data_path)
    # width, height of the final images for training, testing and prediction
    width = 28
    height = 28
    # number of channels in the images, here the images are grayscale,
    # so depth is 1
    depth = 1
    # K.image_data_format is the format of the input data for the model.
    # There are two formats, 'channels_first' and 'channels_last'.
    # The input data will be shaped according to these formats.
    # 'channels_first' corresponds to (num_samples, depth, width, height)
    # 'channels_last' corresponds to (num_samples, width, height, depth)
    if K.image_data_format == 'channels_first':
        input_shape = (depth, width, height)
    else:
        input_shape = (width, height, depth)
    train_features, train_labels, test_features, test_labels = get_test_train_features_and_labels(
        x_train, x_test, y_train, y_test, input_shape)
    print("Creating a new model...")
    model = create_cnn_model(num_classes=len(label_map), input_shape=input_shape)
    model.fit(x=train_features, y=train_labels, batch_size=100, epochs=3)
    # save model and label map
    model.save(model_path)
    label_file = open(label_path, mode="w")
    json.dump(label_map, label_file)
    label_file.close()
    print("Evaluating the model...")
    eval_results = model.evaluate(test_features, test_labels)
    print('Loss: ', eval_results[0], 'Accuracy: ', eval_results[1])


# mouse callback to draw on the 'drawing' window
def draw(event, x, y, flags, param):
    global drawing, last_x, last_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if last_x is None or last_y is None:
                last_x = x
                last_y = y
            cv2.line(draw_img, (last_x, last_y), (x, y), (255, 255, 255), 5)
            last_x = x
            last_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_x = None
        last_y = None


def reshape_for_prediction(arr, w, h, d):
    # reshape to (num_of_samples, width, height, depth (or channels)) or
    # (num_of_samples, depth (or channels) width, height)
    # according to the configuration
    # and convert from 0-255 range to 0-1 range
    if K.image_data_format == 'channels_first':
        return arr.reshape((1, d, w, h)).astype('float32') / 255
    else:
        return arr.reshape((1, w, h, d)).astype('float32') / 255


def run_drawing_app(model_path, label_path):
    global drawing, draw_img, last_x, last_y
    print('Loading model...')
    model = load_model(model_path)
    label_file = open(label_path, 'r')
    label_map = json.load(label_file)
    label_file.close()

    pred_img = np.zeros((100, 300, 3), np.uint8)

    cv2.namedWindow('drawing')
    cv2.namedWindow('prediction')
    cv2.setMouseCallback('drawing', draw)
    font = cv2.FONT_HERSHEY_SIMPLEX

    width = 28
    height = 28
    depth = 1

    while(True):
        # show images on their windows
        cv2.imshow('drawing', draw_img)
        cv2.imshow('prediction', pred_img)
        # copy the image on the 'drawing' window and use it to predict
        img_to_pred = draw_img.copy()
        # convert BGR to grayscale image
        img_to_pred = cv2.cvtColor(img_to_pred, cv2.COLOR_BGR2GRAY)
        # resize to a width x height image
        img_to_pred = cv2.resize(img_to_pred, (width, height))
        # reshape to a suitable shape for the model and convert values from 0-255 range to 0-1 range
        img_to_pred = reshape_for_prediction(img_to_pred, width, height, depth)
        # predict the image
        prediction = model.predict_classes(img_to_pred)
        # refill the prediction image with zeros (with black color)
        pred_img.fill(0)
        # if nothing is drawn yet, don't predict anything
        if draw_img.any():
            cv2.putText(
                pred_img, str(label_map[str(int(prediction))]), (10, 70),
                font, 1.5, (255, 255, 255), 2, cv2.LINE_AA
            )

        k = cv2.waitKey(20)  # capture the pressed key
        if k == 27:  # 27 - escape key
            cv2.destroyAllWindows()
            break
        elif k == ord('c'):  # clear the 'drawing' image if c key is pressed
            draw_img.fill(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()
    if args.train:
        train_and_evaluate(DATA_PATH, MODEL_PATH, LABEL_PATH)
    else:
        run_drawing_app(MODEL_PATH, LABEL_PATH)
