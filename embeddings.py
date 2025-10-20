from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from time import perf_counter
import numpy as np
import torch
from pathlib import Path
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
from chromadb.api.types import Embeddable

class EfficientNetImageEmbedding(EmbeddingFunction[Embeddable]):
    def __init__(self, model_name: str = 'google/efficientnet-b0', device: str = 'cpu') -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Processor: resize, convert to tensor [C,H,W] (channels × height × width),
        # normalize pixels, add batch dim [B,C,H,W]
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        # Model: transform the tensor to embeddings or prediction
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
    def load_process_image(self, img_path: Path | str):
        img = Image.open(img_path).convert('L').convert('RGB')
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        return inputs
    
    def __call__(self, input: list[Path]) -> Embeddings:
        all_embeddings = []
        print(f"🔹 Embedding generation for {len(input)} images...")

        for img_path in tqdm(input, desc="Embedding images", unit="img"):
            pixel_values = self.load_process_image(img_path)['pixel_values']
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            emb = outputs.pooler_output[0].cpu().numpy()  # vecteur numpy
            all_embeddings.append(emb.tolist())
        
        print("✅ Embeddings finished.")
        return all_embeddings