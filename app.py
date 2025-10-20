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
import tempfile
from PIL import Image
from io import BytesIO


def main():
    st.set_page_config(
        page_title='M2RS 0.1'
    )
    st.title('M2RS 0.1')
    
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

        chroma_base = ChromaBase()
        
        images_path = get_images_path()
        ids = [str(uuid.uuid4()) for _ in images_path]
        metadatas = [{"path": str(i_path), "filename": i_path.name} for i_path in images_path]
        
        with st.spinner('Création de la base, merci de patienter...'):
            embeddings = chroma_base.compute_embeddings(filespath=images_path)
        
            chroma_base.add_to_collection(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
                )
    
        image_to_query = [image_path]
        results = chroma_base.query_image(image_to_query=image_to_query)
        
        cols = st.columns(3)
        for i, metadata in enumerate(results['metadatas'][0]):
            img_path = metadata['path']
            filename = metadata['filename']
            
            with cols[i % 3]:
                st.image(
                    image=img_path,
                    use_column_width=True,
                    caption=filename
                    )

        st.subheader('Debug :')
        st.write(dict(results))

if __name__ == '__main__':
    main()