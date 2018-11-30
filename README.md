# Emerging Technologies Project
Emerging technologies assignment repository consisting of jupyter notebooks on various topics in data science and machine learning.

## Installation & Setup
To run these notebooks you will first need to make sure that you have installed Anaconda, tensorflow, keras and numpy. You can download Anaconda [here](https://www.anaconda.com/download/).

Make sure you clone the repository by typing the following into a terminal window:
```
git clone https://github.com/lanodburke/Emerging-Technologies-Project/
```

After cloning the repository move into the Emerging-Technologies-Project folder by typing the following:
```
cd Emerging-Technologies-Project
```

After installing the required packages, open a terminal window and type the following command:
```
jupyter notebook
```

## Running digit recognition script
This script takes in command line arguments which are outlined below: 
- -h (help): will print some instructions describing the different parameters
- -b (build): this will build the keras model that will be used to run the digit recognition
- -t (test): this will take in an input file as an argument such as a PNG, JPG or any other image format

### Build the model 
Builds the model and saves it with the .h5 file format
```
python digitrec.py -b
```

### Test the model
Run the saved model and make a prediction on the image passed in as an argument
```
python digitrec.py -t image.png
```

### Contributors 
- [Donal Burke](https://github.com/lanodburke/) (G00337729)

