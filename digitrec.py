# Required imports
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

# Load data function, calls the mnist.load_data() function from keras
def load_data():
    # Load dataset from keras 
    return mnist.load_data()

# Build model function, pass in training values and number of classes
def build_model(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential()
    # Add a convolutional layer with 32 input filters/channels, input shape is (samples, width, height)
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # Add a Max Pooling layer with size 2 * 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout layer to exclude 20% of the nuerons in the model to prevent over fitting
    model.add(Dropout(0.2))
    # Flatten the input layers to be read by classifcation laye
    model.add(Flatten())
    # Add a fully connected layer with 128 neurons
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model with categorical crossentropy as loss function
    # Adam as the optimizer and set the metrics to accuracy to try and achieve the best accuarcy possible
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model with our data
    # Set number of epochs to 10, batch size to 200
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    return model

# Pre process function loads MNIST dataset and shapes inputs and outputs for our model
def pre_process():
    # Load MNIST dataset
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

# Load image function, takes image path as a parameter
def load_image(img_path):
    # load image with load_image() function from keras
    # set color mode to grayscale
    # resize the image to 28 * 28
    img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
    # convert image to an array
    img_tensor = image.img_to_array(img)
    # reshape image array to (samples, width, height)
    img_tensor = img_tensor.reshape(img_tensor.shape[0], 1, 28, 28).astype('float32')
    # normalize input from 0-255 to 0-1
    img_tensor = img_tensor / 255

    return img_tensor

def display_image(pred, img_path):
    # Adding a title to the Plot 
    plt.title("Prediction: " + str(pred))
    # Using the plt.imshow() to add the image plot to 
    # the matplotlib figure 
    img = image.load_img(img_path, target_size=(28, 28))
    plt.imshow(img, cmap='gray')
    # This just hides x and y tick values by passing in 
    # empty lists to make the output a little cleaner 
    plt.xticks([]), plt.yticks([]) 
    plt.show()

def main(argv):
    # try and read in System arguments
    try:
        # get options (h for help, b for build, t for test)
      opts, args = getopt.getopt(argv,"hbt:")
    except getopt.GetoptError:
      print("digitrec.py -b (Build neural network)")
      print("digitrec.py -t <image>")
      sys.exit(2)
    # loop through options and arguments provided
    for opt, arg in opts:
      # if option is -h print out available parameters
      if opt == '-h':
         print('digitrec.py -b (build model)')
         print('digitrec.py -t (image as png)')
         sys.exit()
      # Else if option is -t 
      elif opt in ("-t"):
         # if the model exists run prediction on input image
         if(os.path.exists("digitrec_model.h5")):
            print("Loading model from file...")
            model = load_model("digitrec_model.h5")
            image_path = arg
             # load a single image
            new_image = load_image(image_path)
            # check prediction
            pred = model.predict_classes(new_image)
            print("Prediction: ", pred)
            display_image(pred, image_path)
         # else tell user to build model first before running prediction
         else:
            print("Cannot load model as it does not exist!")
            print("Try running digitrec.py -b (to build the model first before testing!)")
      # If option is -b build the model
      elif opt in ("-b"):
         print("Building new model...")
         # pre process inputs
         (X_train, y_train), (X_test, y_test) = pre_process()
         num_classes = y_test.shape[1]
         # build the model
         model = build_model(X_train, y_train, X_test, y_test, num_classes)
         # Final evaluation of the model
         scores = model.evaluate(X_test, y_test, verbose=0)
         print("Error: %.2f%%" % (100-scores[1]*100))
         # save the model to file
         model.save('digitrec_model.h5')        

if __name__ == '__main__':
   main(sys.argv[1:])