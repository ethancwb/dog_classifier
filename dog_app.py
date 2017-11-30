import os
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from extract_bottleneck_features import *
from tqdm import tqdm
from PIL import ImageFile
import random

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
ImageFile.LOAD_TRUNCATED_IMAGES = True

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
# print('There are %d total dog categories.' % len(dog_names))
# print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
# print('There are %d training dog images.' % len(train_files))
# print('There are %d validation dog images.' % len(valid_files))
# print('There are %d test dog images.'% len(test_files))

random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
#print('There are %d total human images.' % len(human_files))

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = train_files[:100]

# human_init_count = 0
# dog_init_count = 0

# for human in human_files_short:
#     if face_detector(human):
#         human_init_count += 1
# print('human image detection success rate: ' + str(human_init_count *100 / len(human_files_short)) + '%')
#
# for dog in dog_files_short:
#     if face_detector(dog):
#         dog_init_count += 1
# print('dog image detected as human rate: ' + str(dog_init_count *100 / len(dog_files_short)) + '%')

eye_haarcascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
nose_haarcascade = cv2.CascadeClassifier('haarcascades/Nariz.xml')
mouth_haarcascade = cv2.CascadeClassifier('haarcascades/Mouth.xml')

def clear_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye = eye_haarcascade.detectMultiScale(gray)
    nost = nose_haarcascade.detectMultiScale(gray)
    mouth = mouth_haarcascade.detectMultiScale(gray)
    result = len(eye) != 0 and len(mouth) != 0 and len(nost) != 0
    if face_detector(img_path) == False and result == True:
        result = False
    return result
#
# clear_face_count = 0
#
# for face in human_files_short:
#     if clear_face(face):
#         clear_face_count += 1

# print('clear face image detection success rate: ' + str(clear_face_count * 100 / human_init_count) + '%')

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# human_count = 0
# for human_img in human_files_short:
#     if dog_detector(human_img):
#         human_count = human_count + 1
#
# dog_count = 0
# for dog_img in dog_files_short:
#     if dog_detector(dog_img):
#         dog_count = dog_count + 1
#
# print(str(human_count) + "% of the humans were detected as dogs")
# print(str(dog_count) + "% of the dogs were detected as dogs!")

# pre-process the data for Keras

try:
    Resnet50_model = load_model('saved_models/dogIdentifier.h5')
except OSError:
    train_tensors = paths_to_tensor(train_files).astype('float32') / 255
    valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
    test_tensors = paths_to_tensor(test_files).astype('float32') / 255

    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    train_DogResnet50 = bottleneck_features['train']
    valid_DogResnet50 = bottleneck_features['valid']
    test_DogResnet50 = bottleneck_features['test']

    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_DogResnet50.shape[1:]))
    Resnet50_model.add(Dense(1024, kernel_initializer='he_normal'))
    Resnet50_model.add(BatchNormalization())
    Resnet50_model.add(Activation('relu'))
    Resnet50_model.add(Dense(133, activation='softmax'))

    Resnet50_model.summary()

    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.ResNet50.hdf5',
                                   verbose=1, save_best_only=True)

    Resnet50_model.fit(train_DogResnet50, train_targets,
              validation_data=(valid_DogResnet50, valid_targets),
              epochs=25, batch_size=20, callbacks=[checkpointer], verbose=1)

    Resnet50_model.load_weights('saved_models/weights.best.ResNet50.hdf5')
    Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_DogResnet50]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    Resnet50_model.save('saved_models/dogIdentifier.h5')

def dog_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)

    ind = np.argsort(predicted_vector[0])[::-1][:3]
    # return dog breed that is predicted by the model
    result = ''
    for i in ind:
        result += '(' + dog_names[i] + "  {0:.2f}%".format(predicted_vector[0][i] * 100) + " )" + '\n'
    return result

def classifier(img_path):
    breed = dog_breed(img_path)
    if dog_detector(img_path):
        #print('This cutie is ...', '\n', breed)
        return breed
    elif clear_face(img_path):
        #print('You look like ...', '\n', breed)
        return breed
    else:
        print('No human or dog!')
        return 'No human or dog!'

# print(train_files[0])
# a = classifier("sample_pictures/trump.jpg")
# print(a)
#
# ## Load the cell
# sample_files = np.array(glob("sample_pictures/*"))
# print(sample_files)
#
# for path in sample_files:
#     classifier(path)