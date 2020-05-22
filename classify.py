from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st

@st.cache(allow_output_mutation=True)
def get_model():
        model = load_model('Model/Traffic_Sign_Classifier_CNN.hdf5')
        print('Model Loaded')
        return model 

        
def predict(image):
        loaded_model = get_model()
        image = load_img(image, target_size=(32, 32), color_mode = "grayscale")
        image = img_to_array(image)
        image = image/255.0
        image = np.reshape(image,[1,32,32,1])

        classes = loaded_model.predict_classes(image)

        return classes
