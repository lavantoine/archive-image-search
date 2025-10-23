import streamlit as st
import pandas as pd
import numpy as np
import time
from utils import get_all_images_path, generate_id
from pathlib import Path
from chroma_client import ChromaBase
import torch
import uuid
from pprint import pprint
import tempfile
from PIL import Image
from io import BytesIO
from s3 import S3

def unsure_uploaded_images(paths):
    s3 = S3()
    

@st.cache_resource
def initialize_s3(images_path):
    global s3
    s3 = S3()
    # s3.upload_files(images_path)

@st.cache_resource
def initialize_chroma() -> ChromaBase:
    chroma_base = ChromaBase()
    
    all_images_path = get_all_images_path()
    all_ids = [generate_id(_) for _ in all_images_path]
    
    # initialize_s3(images_path=all_images_path)
    
    new_images_path, new_ids = chroma_base.keep_new_only(all_images_path, ids=all_ids)
    metadatas = [{"path": str(i_path), "filename": i_path.name} for i_path in new_images_path]
    
    with st.spinner('Vérification de la base vectorielle, merci de patienter...'):
        if new_ids:
            embeddings = chroma_base.compute_embeddings(filespath=new_images_path)

            chroma_base.add_to_collection(
                ids=new_ids,
                embeddings=embeddings,
                metadatas=metadatas
                )
        return chroma_base

def main():
    st.set_page_config(
        page_title='M2RS 0.1'
    )
    st.title('M2RS 0.1')
    st.write('Texte explicatif...')
    
    with st.sidebar:
        st.subheader('Accueil')
    
    s3 = S3()
    
    chroma_base = initialize_chroma()
    
    uploaded_image = st.file_uploader(label="Merci de choisir une image :", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        _, center, _ = st.columns((1,2,1))
        center.image(
            image=image_bytes,
            caption=uploaded_image.name,
            use_column_width=True
        )
        
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_path = Path(tmp.name)
            image_pil.save(image_path)
        
        st.subheader('Images similaires :')

        image_to_query = [image_path]
        results = chroma_base.query_image(image_to_query=image_to_query, n_results=12)
        
        cols = st.columns(3)
        for i, metadata in enumerate(results['metadatas'][0]):
            # img_path = metadata['path']
            filename = metadata['filename']
            
            img_bytes = s3.download_file(filename=filename)
            
            with cols[i % 3]:
                st.image(
                    image=img_bytes,
                    use_column_width=True,
                    caption=filename
                    )

        st.subheader('Debug :')
        st.write(dict(results))

if __name__ == '__main__':
    main()