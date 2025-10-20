import streamlit as st
import pandas as pd
import numpy as np
import time
from utils import get_images_path
from pathlib import Path
from chroma_client import ChromaBase
import torch
import uuid
from pprint import pprint


def main():
    st.title('M2RS 0.1')
    
    uploaded_photo = st.file_uploader(label="Merci de choisir une image :", type=["jpg", "jpeg", "png"])
    if uploaded_photo is not None:
        _, center, _ = st.columns((1,2,1))
        center.image(
            image=uploaded_photo.getvalue(),
            caption=uploaded_photo.name,
            use_column_width=True
        )
    
    st.subheader('Images similaires :')

    # images_path = get_images_path()

    # for col in st.columns(3):
    #     with col:
    #         st.image(str(images_path[0]))

    # image_to_query = [Path(__file__).parent / 'data/test/mona-lisa-test.jpg']
    # result = query_image(image_to_query)

    # st.write(result)
    
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    chroma_base = ChromaBase()
    
    images_path = get_images_path()
    ids = [str(uuid.uuid4()) for _ in images_path]
    metadatas = [{"path": str(i_path), "filename": i_path.name} for i_path in images_path]
    
    embeddings = chroma_base.compute_embeddings(filespath=images_path)
    
    chroma_base.add_to_collection(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
        )
    
    image_to_query = [Path(__file__).parent / 'data/test/mona-lisa-test.jpg']

    results = chroma_base.query_image(image_to_query=image_to_query)
    
    print()
    pprint(results)

if __name__ == '__main__':
    main()