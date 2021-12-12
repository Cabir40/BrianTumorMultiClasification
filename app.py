

import streamlit as st
import pandas as pd
import base64
import functools  
import numpy as np
from scipy import spatial

import warnings
warnings.filterwarnings('ignore')

st.sidebar.image('https://nlp.johnsnowlabs.com/assets/images/logo.png', use_column_width=True)

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.5rem; padding: 0.5rem; margin-bottom: 0.5rem">{}</div>"""

st.title("Spark NLP Pipeline Playground")
