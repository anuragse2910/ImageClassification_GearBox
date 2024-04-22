import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf

# Load your trained gearbox classification model
MODEL = tf.keras.models.load_model('./Saved_Models/model1.keras')
CLASS_NAMES = [ " Attention Required!:heavy_multiplication_x:","All Systems Good!:thumbsup:"]

def preprocess_image(image):
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, 0)
    image = tf.image.resize(image, (300, 300))
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

def predict_gearbox_type(image):
    input = preprocess_image(image)
    prediction = MODEL.predict(input)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    return predicted_class

def main():
    st.title("Gearbox Diagnosis:gear:")
    st.subheader("Choose an image")

    uploaded_file = st.file_uploader("", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=False)
        st.write("")
        image = np.array(image)    
        prediction = predict_gearbox_type(image)
        st.info(prediction)

if __name__ == '__main__':
    main()

# To run the web app
### streamlit run app.py --server.enableXsrfProtection false