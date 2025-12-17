# src/evaluate_rag.py

import time
from typing import Dict, List
from pathlib import Path
import pandas as pd
from src.rag_model import ArxivRAG
from config import PATHS


def calculate_mrr(predictions: List[List[str]], ground_truth: List[str], k: int = 5) -> float:
    reciprocal_ranks = []
    for pred_ids, true_id in zip(predictions, ground_truth):
        if true_id in pred_ids[:k]:
            rank = pred_ids.index(true_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_retrieval(rag: ArxivRAG, test_df: pd.DataFrame, k: int = 5) -> Dict:
    predictions = []
    total_time = 0
    
    print(f"Оценка на {len(test_df)} запросах...")
    
    for idx, row in test_df.iterrows():
        start = time.time()
        ids, _ = rag.search(row['query'], k=k)
        predictions.append(ids)
        total_time += time.time() - start
        
        if (idx + 1) % 100 == 0:
            print(f"Обработано: {idx + 1}/{len(test_df)}")
    
    mrr = calculate_mrr(predictions, test_df['id'].tolist(), k=k)
    
    results = {
        'mrr': mrr,
        'avg_time_ms': total_time / len(test_df) * 1000,
        'total_time': total_time,
        'predictions': predictions
    }
    
    print(f"\n{'='*70}")
    print(f"MRR@{k}: {mrr:.4f}")
    print(f"Время: {results['avg_time_ms']:.2f} ms/запрос")
    print(f"Всего: {results['total_time']:.2f} s")
    print(f"{'='*70}")
    
    return results


def run_rag_pipeline(test_df: pd.DataFrame, embedder: str, reranker: str, k: int = 5):
    """Полный pipeline: загрузка → индексация → оценка"""
    
    rag = ArxivRAG(
        embedder_name=embedder,
        reranker_name=reranker,
        use_reranker=True
    )
    
    rag.load_data()
    rag.build_index(batch_size=64)
    rag.save_index('data/index')
    
    results = evaluate_retrieval(rag, test_df, k=k)
    
    return rag, results


if __name__ == "__main__":
    # Загрузка тестовых данных из правильного пути
    data_path = Path(PATHS['extracted_data'])
    test_file = list(data_path.rglob('*test_sample*.csv'))[0]
    
    print(f"Загрузка test данных из: {test_file}")
    test_df = pd.read_csv(test_file)
    
    # Запуск
    rag, results = run_rag_pipeline(
        test_df,
        embedder="all-MiniLM-L6-v2",
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        k=5
    )