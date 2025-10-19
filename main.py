from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from time import perf_counter
import numpy as np
import torch
from pathlib import Path
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pprint import pprint

class EfficientNetImageEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = 'google/efficientnet-b7', device: str = 'cpu') -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Processor:
        # - resize to the expected size
        # - convert the image to a tensor [C, H, W] (channels × height × width)
        # - normalize the pixels (often between -1 and 1 or 0 and 1)
        # - add the batch dimension: [batch, C, H, W] e.g. how many samples (images) are fed to the model
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        # Model: transform the tensor to embeddings or prediction
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
    def load_process_image(self, img_path: Path | str):
        img = Image.open(img_path).convert('L').convert('RGB')
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        return inputs
    
    def __call__(self, input: list[Path | str]) -> Embeddings:
        all_embeddings = []
        for img_path in input:
            pixel_values = self.load_process_image(img_path)['pixel_values']
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            emb = outputs.pooler_output[0].numpy()  # vecteur numpy
            all_embeddings.append(emb.tolist())
        return all_embeddings

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    img_path = script_dir / '../data/'
    
    # For TPU acceleration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = 'cpu'
    model_name = 'google/efficientnet-b7'

    images = [script_dir / '../data/base/mona-lisa-0.jpg', script_dir / '../data/mona-lisa-1.jpg']
    
    embed_function = EfficientNetImageEmbedding()

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name='my_collection', embedding_function=embed_function)
    
    embeddings = embed_function(images)
    
    collection.upsert(ids=['id0', 'id1'], embeddings=embeddings)
    
    results = collection.query(
    query_embeddings=embed_function([script_dir / '../data/base/mona-lisa-3-crop.jpeg']),
    n_results=2,
    include=["embeddings", "distances"]) # type: ignore
    
    pprint(results)

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# mona0 = embed(script_dir / '../data/base/mona-lisa-0.jpg')
# mona1 = embed(script_dir / '../data/mona-lisa-1.jpg')

# print("cosine_similarity: ", cosine_similarity(mona0, mona1))