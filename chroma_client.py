import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, QueryResult
from embeddings import EfficientNetImageEmbedding
from pathlib import Path
from utils import get_logger
logger = get_logger(__name__)

class ChromaBase():
    def __init__(self) -> None:
        self.embedding_function = EfficientNetImageEmbedding()
        self.chroma_client = chromadb.PersistentClient(path=Path(__file__).parent / '.local')
        self.collection = self.chroma_client.get_or_create_collection(
            name='my_collection',
            embedding_function=self.embedding_function
        )
    
    def keep_new_only(self, filespath, ids):
        existing_ids = set(self.collection.get()["ids"])

        new_filespath = []
        new_ids = []
        for filepath, id in zip(filespath, ids):
            if id not in existing_ids:
                new_filespath.append(filepath)
                new_ids.append(id)
        return new_filespath, new_ids
        
    def compute_embeddings(self, filespath) -> Embeddings:
        return self.embedding_function(filespath)
    
    def add_to_collection(self, ids: list[str], embeddings, metadatas):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
            )
        logger.info(msg=f"{len(embeddings)} embeddings added to collection")

    def query_image(self, image_to_query: list[Path], n_results: int = 3, include: list = ["distances", 'metadatas']) -> QueryResult:
        return self.collection.query(
            query_embeddings=self.embedding_function(input=image_to_query),
            n_results=n_results,
            include=include
            ) # type: ignore