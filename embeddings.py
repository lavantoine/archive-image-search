from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModel
from time import perf_counter
import numpy as np
import torch
from pathlib import Path
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
from io import BytesIO
from utils import get_device
from chromadb.api.types import Embeddable
ImageFile.LOAD_TRUNCATED_IMAGES = True
import streamlit as st
from utils import get_logger
logger = get_logger(__name__)
from s3 import S3

class EfficientNetImageEmbedding(EmbeddingFunction[Embeddable]):
    def __init__(self, bucket_client: S3, model_name: str = 'google/efficientnet-b0', device: str = get_device()) -> None:
        super().__init__()
        self.bucket_client = bucket_client
        self.model_name = model_name
        self.device = device
        
        # Processor: resize, convert to tensor [C,H,W] (channels × height × width),
        # normalize pixels, add batch dim [B,C,H,W]
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        # Model: transform the tensor to embeddings or prediction
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
    def load_process_image(self, img: Path | str):
        # Download image file from bucket
        try:
            bytes_img = self.bucket_client.download_file(img)
            if not bytes_img:
                pil_img = Image.open(img).convert('L').convert('RGB')
            else:
                pil_img = Image.open(bytes_img).convert('L').convert('RGB')
        except Exception as e:
            logger.error(f'Error while retrieving image from embedding function: {e}', exc_info=True)

        inputs = self.processor(images=pil_img, return_tensors='pt').to(self.device)
        return inputs
    
    def __call__(self, input: list[Path | str]) -> Embeddings:
        all_embeddings = []
        logger.info(f"▶️ Embedding generation for {len(input)} images...")
        
        n = len(input)
        text = 'Vérification de l\'encodage des photos, merci de patienter...'
        bar = st.progress(0.0, text)
        
        for i, img_path in enumerate(tqdm(iterable=input, desc="Embedding images", unit="img")):
            tensor_dict = self.load_process_image(img_path)
            if tensor_dict is not None:
                pixel_values = tensor_dict['pixel_values']
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                emb = outputs.pooler_output[0].cpu().numpy()  # vecteur numpy
                all_embeddings.append(emb.tolist())
                bar.progress(value=float((i+1)/n), text=f'{text} ({i}/{n})')
        bar.empty()

        
        print("✅ Embeddings finished.")
        return all_embeddings