# Required imports
# import numpy for mathematical operations
import numpy
import os.path
import sys, getopt
import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
K.set_image_dim_ordering('th')

def load_data():
    # Load dataset from keras 
    return mnist.load_data()

def build_model(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    return model

def pre_process():
    (X_train, y_train), (X_test, y_test) = load_data()

    # reshape to be samples, pixels, width, height
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)

def load_image(img_path):
    img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor.reshape(-1, 1, 28, 28).astype('float32')
    # img_tensor = numpy.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255
    return img_tensor

def display_image(pred, img_path):
    # Adding a title to the Plot 
    plt.title(pred)
    # Using the plt.imshow() to add the image plot to 
    # the matplotlib figure 
    img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
    plt.imshow(img, cmap='gray')
    # This just hides x and y tick values by passing in 
    # empty lists to make the output a little cleaner 
    plt.xticks([]), plt.yticks([]) 
    plt.show()

def main(argv):
    try:
      opts, args = getopt.getopt(argv,"hbt:")
    except getopt.GetoptError:
      print("digitrec.py -b (Build neural network)")
      print("digitrec.py -t <image>")
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('digitrec.py -b (build model)')
         print('digitrec.py -t (image as png)')
         sys.exit()
      elif opt in ("-t"):
         if(os.path.exists("digitrec_model.h5")):
            print("Loading model from file...")
            model = load_model("digitrec_model.h5")
            image_path = arg
             # load a single image
            new_image = load_image(image_path)
            # check prediction
            pred = model.predict_classes(new_image)
            print("Prediction: ", pred[0])
            display_image(pred, image_path)
      elif opt in ("-b"):
         print("Building new model...")
         (X_train, y_train), (X_test, y_test) = pre_process()
         num_classes = y_test.shape[1]
         model = build_model(X_train, y_train, X_test, y_test, num_classes)
         # Final evaluation of the model
         scores = model.evaluate(X_test, y_test, verbose=0)
         print("Error: %.2f%%" % (100-scores[1]*100))
         model.save('digitrec_model.h5')        

if __name__ == '__main__':
   main(sys.argv[1:])