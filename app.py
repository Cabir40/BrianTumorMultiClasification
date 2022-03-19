import streamlit as st
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

st.sidebar.image('https://eski.ahievran.edu.tr/images/haberler/basin/logomuz/ahievran_logo_210518.png', width=256)

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.5rem; padding: 0.5rem; margin-bottom: 0.5rem">{}</div>"""

st.title("Ahi Hub Prediction Playground")



modelMulti = load_model('model_best.h5')
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


PIPELINES = ['Brain Tumor Detection DL',
             'EKG Apne Detection DL',
             'Stock Price Prection DL',
             'NLP Social Media Sentiment Analysis',
             'other']

BrainTumorList = ['BinaryDetection',
                  'MultiClassification',
                  'ner_jsl',
                  'ner_bionlp']

  

st.sidebar.header('Choose Prediction Model')
st.sidebar.write('')

pipe_type = st.sidebar.selectbox('Select Prediction Type',PIPELINES)



if pipe_type == 'Brain Tumor Detection DL':
  ModelType = st.sidebar.selectbox("Select MedicalNerModel", (BrainTumorList))
  
  uploaded_file = st.file_uploader("Choose an image...", type="jpg")

  if uploaded_file is not None:
      image = Image.open(uploaded_file)


      col1, col2 = st.columns(2)

      with col1:
          st.header("Picture")
          st.image(image, caption='Uploaded Image.',use_column_width=True )
          st.write("")

      image = image.resize((128,128))
      image = np.array(image)/255.0
      images = image.reshape(1,128,128,3)
      predicted = modelMulti.predict(images)[0]
      predicted = np.argmax(predicted)
      predicted = labels[predicted]
      prob = modelMulti.predict(images)[0].round(5)

      with col2:
          st.header("Prediction")
          st.write('Predicted:', predicted)
          df = pd.DataFrame(index = labels, data= prob, columns=["probablity"])
          st.write(df)
      
      
      #predict(image, model, labels)


elif pipe_type == 'EKG Apne Detection DL':

  nerModelType = st.sidebar.selectbox("Select MedicalNerModel",(BrainTumorList))
  st.sidebar.write('')       

elif pipe_type == 'Stock Price Prection DL':
  nerModelType = st.sidebar.selectbox("Select MedicalNerModel",(BrainTumorList))
  st.sidebar.write('')

elif pipe_type == 'NLP Social Media Sentiment Analysis':
  nerModelType = st.sidebar.selectbox("Select MedicalNerModel",(BrainTumorList))
  st.sidebar.write('')

  TEXT = ['A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting. The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day .',
        'The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day.',
        'I experienced fatigue, muscle cramps, anxiety, agression and sadness after taking Lipitor but no more adverse after passing Zocor.',
        'CustomText']

  textType = st.selectbox("Select TEXT",(TEXT))

  if textType != "CustomText":
    ner_text = textType
    st.write('NER Input Text',textType)
  else:
    ner_text = st.text_area('NER Input Text', 'Please type here your text.')

else:
  st.sidebar.write('We are working new Pipeline Types')
  st.sidebar.write('')
  pipeline = None