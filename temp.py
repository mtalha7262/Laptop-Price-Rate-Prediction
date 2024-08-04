import streamlit as st
import numpy as np
import pickle
import pandas as pd


# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
# lpp = pickle.load(open('lpp.pkl', 'rb'))
# model.lpp = pickle.load(open('lpp.pkl', 'rb'))

lpp = pd.read_pickle('lpp.pkl')
lpp.to_pickle('lpp.pkl')
# with open('lpp.pkl', 'wb') as f:
#    lpp.to_pickle(f)

st.title(" Laptop Predictor ")

# brand
company = st.selectbox('Brand', list(lpp['Company'].unique()))

# type of laptop
ty_pe = st.selectbox('Type', lpp['TypeName'].unique())

# Ram
ram = st.selectbox('RAM (in GB)', ['2', '4', '6', '8', '12', '16', '24', '32', '64'])


# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', lpp['Cpu Brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', lpp['Gpu Brand'].unique())

os = st.selectbox('OS', lpp['OS'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, ty_pe, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title((int(np.exp(pipe.predict(query)[0]))))
    st.write(r"The predicted price is: {Prediction}")
