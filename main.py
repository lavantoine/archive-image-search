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

class EfficientNetImageEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = 'google/efficientnet-b0', device: str = 'mps') -> None:
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
        print(f"🔹 Embedding generation for {len(input)} images...")

        for img_path in tqdm(input, desc="Embedding images", unit="img"):
            pixel_values = self.load_process_image(img_path)['pixel_values']
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            emb = outputs.pooler_output[0].cpu().numpy()  # vecteur numpy
            all_embeddings.append(emb.tolist())
        
        print("✅ Embeddings finished.")
        return all_embeddings

if __name__ == '__main__':
    # For TPU acceleration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = 'cpu'
    model_name = 'google/efficientnet-b7'

    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data/portrait-0.1k'
    files = list(data_dir.rglob('*.jpg'))
    print(f"{len(files)} images trouvées.")
    
    embed_function = EfficientNetImageEmbedding(device='cpu')

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name='my_collection',
        embedding_function=embed_function
    )
    
    
    ids = [f"id_{i}" for i in range(len(files))]
    
    embeddings = embed_function(files)
    
    collection.upsert(ids=ids, embeddings=embeddings)
    
    print(f"{len(embeddings)} embeddings ajoutés à la collection.")
    
    query_image = [script_dir / 'data/test/mona-lisa-test.jpg']
    
    results = collection.query(
    query_embeddings=embed_function(query_image),
    n_results=5,
    include=["embeddings", "distances"]) # type: ignore
    
    print()
    pprint(results)

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# mona0 = embed(script_dir / '../data/base/mona-lisa-0.jpg')
# mona1 = embed(script_dir / '../data/mona-lisa-1.jpg')

# print("cosine_similarity: ", cosine_similarity(mona0, mona1))