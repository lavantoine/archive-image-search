import streamlit as st
import pandas as pd
import numpy as np
import time
from main import get_images_path, query_image
from pathlib import Path


st.title('M2RS 0.1')

images_path = get_images_path()

for col in st.columns(3):
    with col:
        st.image(str(images_path[0]))

image_to_query = [Path(__file__).parent / 'data/test/mona-lisa-test.jpg']
result = query_image(image_to_query)

st.write(result)