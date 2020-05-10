from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st

# The data will be downloaded only once and cached for future use by using "st.cache"
# get_model() loads the model into the cache
@st.cache(allow_output_mutation=True)
def get_model():
	model = load_model('Traffic_Sign_Classifier_CNN.hdf5')
        print('Model Loaded')
        return model

        
def predict(image):
        loaded_model = get_model()
        
        # loads the image user has uploaded
        # Resizes it to 32x32 and converts it into Grayscale
        image = load_img(image, target_size=(32, 32), color_mode = "grayscale")
        
        # Convert the image pixels to a numpy array
        image = img_to_array(image)
        
        # Normalize the images
        image = image/255.0
        
        # reshape the array to feed it into our model
        image = np.reshape(image,[1,32,32,1])
        
        # Make Predictions
        classes = loaded_model.predict_classes(image)

        return classes
