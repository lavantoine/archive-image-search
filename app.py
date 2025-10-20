import streamlit as st
import pandas as pd
import numpy as np
import time
from embeddings import get_images_path, query_image
from pathlib import Path

def main():
    st.title('M2RS 0.1')

    images_path = get_images_path()

    for col in st.columns(3):
        with col:
            st.image(str(images_path[0]))

    image_to_query = [Path(__file__).parent / 'data/test/mona-lisa-test.jpg']
    result = query_image(image_to_query)

    st.write(result)
    
    
    if __name__ == '__main__':
    # For TPU acceleration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = 'cpu'
    model_name = 'google/efficientnet-b7'
    
    images_path = get_images_path()
    
    # embed_function = EfficientNetImageEmbedding(device='cpu')

    # chroma_client = chromadb.Client()
    # collection = chroma_client.create_collection(
    #     name='my_collection',
    #     embedding_function=embed_function
    # )
    
    ids = [str(uuid.uuid4()) for _ in files]
    
    # embeddings = embed_function(files)
    
    metadatas = [{"path": str(f), "filename": f.name} for f in files]
    
    # collection.upsert(
    #     ids=ids,
    #     embeddings=embeddings,
    #     metadatas=metadata)
    
    # print(f"{len(embeddings)} embeddings ajoutés à la collection.")
    
    image_to_query = [Path(__file__).parent / 'data/test/mona-lisa-test.jpg']

    results = query_image(image_to_query)
    
    print()
    pprint(results)

if __name__ == '__main__':
    main()