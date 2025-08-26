import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from cv2 import resize
import tensorflow as tf 


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Conv_model_mnist.keras")

model = load_model()

if "user_input" not in st.session_state:
    st.session_state.user_input = None 

def predict(model,img):
    prediction = np.argmax(model.predict(img))
    return prediction

st.title("ML Handwriting number detection")


canvas_result = st_canvas(
    stroke_width= 25 ,
    stroke_color= "#000000" ,
    background_color= "#ffffff",
    height=400,
    width=400,
    drawing_mode="freedraw", 
    key="canvas",  
)

if st.button("Predict!!"):
    image = canvas_result.image_data
    image = image[:,:,:3]
    image = np.all(image == 0, axis=-1).astype(np.float32)
    image = resize(image,(28,28))
    image = np.expand_dims(image, axis = -1)
    image = np.expand_dims(image, axis = 0)
    st.session_state.user_input = image


    prediction = predict(model,st.session_state.user_input)
    st.success(prediction)
