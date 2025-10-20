import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, QueryResult
from embeddings import EfficientNetImageEmbedding
import logging
logger = logging.getLogger(__name__)
from pathlib import Path


class ChromaBase():
    def __init__(self) -> None:
        self.embedding_function = EfficientNetImageEmbedding()
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name='my_collection',
            embedding_function=self.embedding_function
        )
    
    def compute_embeddings(self, filespath) -> Embeddings:
        return self.embedding_function(filespath)
    
    def add_to_collection(self, ids: list[str], embeddings, metadatas):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
            )
        logger.info(f"{len(embeddings)} embeddings add to collection.")

    def query_image(self, image_to_query: list[Path], n_results: int = 3, include: list = ["distances", 'metadatas']) -> QueryResult:
        return self.collection.query(
            query_embeddings=self.embedding_function(image_to_query),
            n_results=n_results,
            include=include
            ) # type: ignore