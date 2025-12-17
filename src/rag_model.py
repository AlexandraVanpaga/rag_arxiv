# src/rag_model.py

import numpy as np
import pandas as pd
import torch
import faiss
from pathlib import Path
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import PATHS


class ModelManager:
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_embedder(self, model_name: str):
        if 'embedder' not in self._models:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._models['embedder'] = SentenceTransformer(model_name, device=device)
        return self._models['embedder']
    
    def get_reranker(self, model_name: str):
        if 'reranker' not in self._models:
            self._models['reranker'] = CrossEncoder(model_name)
        return self._models['reranker']
    
    def clear_models(self):
        self._models.clear()
        torch.cuda.empty_cache()


class ArxivRAG:
    def __init__(self, embedder_name: str, reranker_name: str, use_reranker: bool = True):
        self.model_manager = ModelManager()
        self.embedder_name = embedder_name
        self.reranker_name = reranker_name
        self.use_reranker = use_reranker
        
        self.embedder = None
        self.reranker = None
        self.index = None
        self.metadata_df = None
        self.embeddings = None
    
    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = self.model_manager.get_embedder(self.embedder_name)
        return self.embedder
    
    def _get_reranker(self):
        if self.reranker is None and self.use_reranker:
            self.reranker = self.model_manager.get_reranker(self.reranker_name)
        return self.reranker
    
    def load_data(self, path: Optional[str] = None):
        if path is None:
            path = Path(PATHS['processed_data']) / 'metadata.parquet'
        self.metadata_df = pd.read_parquet(path)
        print(f"Загружено: {len(self.metadata_df)} документов")
    
    def build_index(self, batch_size: int = 64):
        if self.metadata_df is None:
            raise ValueError("Загрузите данные через load_data()")
        
        embedder = self._get_embedder()
        texts = self.metadata_df['abstract'].tolist()
        
        self.embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        
        print(f"Индекс: {self.index.ntotal} документов, dim={dimension}")
    
    def search(self, query: str, k: int = 5, rerank_top_k: Optional[int] = None):
        if self.index is None:
            raise ValueError("Постройте индекс через build_index()")
        
        if self.use_reranker and rerank_top_k is None:
            rerank_top_k = k * 10
        
        embedder = self._get_embedder()
        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        search_k = rerank_top_k if rerank_top_k else k
        distances, indices = self.index.search(query_emb, search_k)
        
        results = self.metadata_df.iloc[indices[0]].copy()
        results['score'] = distances[0]
        
        if self.use_reranker and rerank_top_k:
            reranker = self._get_reranker()
            pairs = [[query, doc] for doc in results['abstract'].tolist()]
            results['rerank_score'] = reranker.predict(pairs)
            results = results.nlargest(k, 'rerank_score')
        else:
            results = results.head(k)
        
        return results['id'].tolist(), results
    
    def save_index(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / 'faiss.index'))
        np.save(path / 'embeddings.npy', self.embeddings)
        self.metadata_df.to_parquet(path / 'metadata.parquet')
        print(f"Сохранено: {path}")
    
    def load_index(self, path: str):
        path = Path(path)
        self.index = faiss.read_index(str(path / 'faiss.index'))
        self.embeddings = np.load(path / 'embeddings.npy')
        self.metadata_df = pd.read_parquet(path / 'metadata.parquet')
        print(f"Загружено: {self.index.ntotal} документов")