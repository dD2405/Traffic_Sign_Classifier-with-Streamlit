from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

def predict(image):
        model = load_model('Traffic_Sign_Classifier_CNN.hdf5')
        print('Model Loaded')
        
        image = load_img(image, target_size=(32, 32), color_mode = "grayscale")
        image = img_to_array(image)
        image = np.reshape(image,[1,32,32,1])

        classes = model.predict_classes(image)

        return classes

