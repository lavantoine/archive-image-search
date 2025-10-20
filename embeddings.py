from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from time import perf_counter
import numpy as np
import torch
from pathlib import Path
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pprint import pprint
from tqdm import tqdm
import uuid
from utils import get_images_path
from chromadb.api.types import Embeddable

class EfficientNetImageEmbedding(EmbeddingFunction[Embeddable]):
    def __init__(self, model_name: str = 'google/efficientnet-b0', device: str = 'mps') -> None:
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

# def get_images_path() -> list[Path]:
#     script_dir = Path(__file__).parent
#     data_dir = script_dir / 'data/portrait-0.1k'
#     images = list(data_dir.rglob('*.jpg'))
#     print(f"{len(images)} images found.")
#     return images

# def query_image(image_to_query: list[Path]):
#     return collection.query(
#     query_embeddings=embed_function(query_image),
#     n_results=5,
#     include=["distances", 'metadatas']) # type: ignore
    

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

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# mona0 = embed(script_dir / '../data/base/mona-lisa-0.jpg')
# mona1 = embed(script_dir / '../data/mona-lisa-1.jpg')

# print("cosine_similarity: ", cosine_similarity(mona0, mona1))