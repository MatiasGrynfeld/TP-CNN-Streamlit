import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import os

st.title('Image Clasification CIFAR-10')

st.write('Choose any image and the AI will tell what it is:')

imagenes = {}
for index, file in enumerate(os.listdir('./Imagenes')):
    imagen = np.load(f'./Imagenes/{file}')
    imagenes[index] = imagen
    st.image(imagen, caption=f'{index}. {file.rstrip(".npy")}', use_container_width =True)

number = st.number_input('Choose an image:', min_value=0, max_value=9, value=0)

number_class = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

st.button('Predict', on_click=predict)

result_placeholder = st.empty()
def predict():
    model = tf.keras.models.load_model('./model.keras')
    imagen = imagenes[number].reshape(1, 32, 32, 3)
    prediction = model.predict(imagen)
    result_placeholder.write(f'The image is a {number_class[np.argmax(prediction)]}')