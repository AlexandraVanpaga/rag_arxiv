# src/llm_generator.py

import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMGenerator:
    """Генерация ответов на основе найденных документов"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 512
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        
        print(f"Загрузка LLM: {model_name} на {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map=self.device
        )
        self.model.eval()
    
    def generate(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """Генерация ответа на основе контекста"""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer the question based on the provided context."
        
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        user_message = f"""Context:
{context_text}

Question: {query}

Answer based on the context above:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def clear_memory(self):
        """Очистка GPU памяти"""
        if self.device == 'cuda':
            self.model.cpu()
            torch.cuda.empty_cache()


def generate_answers(
    rag,
    llm: LLMGenerator,
    test_df,
    predictions: List[List[str]],
    num_examples: int = 5,
    top_k: int = 3
):
    """Генерация ответов для примеров из test_df"""
    
    print(f"\n{'='*80}")
    print(f"ГЕНЕРАЦИЯ ОТВЕТОВ (первые {num_examples} примеров)")
    print(f"{'='*80}")
    
    for i in range(min(num_examples, len(test_df))):
        query = test_df.iloc[i]['query']
        true_id = test_df.iloc[i]['id']
        found_ids = predictions[i][:top_k]
        
        # Получаем контексты
        contexts = [
            rag.metadata_df[rag.metadata_df['id'] == doc_id].iloc[0]['abstract']
            for doc_id in found_ids
        ]
        
        # Генерируем ответ
        answer = llm.generate(query, contexts)
        
        print(f"\n{'='*80}")
        print(f"ПРИМЕР {i+1}")
        print(f"{'='*80}")
        print(f"\nВопрос:\n{query}")
        print(f"\nОтвет:\n{answer}")
        print(f"\nИспользованные документы:")
        for j, doc_id in enumerate(found_ids, 1):
            doc = rag.metadata_df[rag.metadata_df['id'] == doc_id].iloc[0]
            marker = "ПРАВИЛЬНЫЙ" if doc_id == true_id else ""
            print(f"  {j}. [{doc_id}] {doc['title'][:70]} {marker}")
        print(f"\nПравильный в топ-{top_k}: {'ДА' if true_id in found_ids else 'НЕТ'}")


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    from src.rag_model import ArxivRAG
    from src.init_and_eval import evaluate_retrieval
    from config import PATHS
    
    print("="*80)
    print("ГЕНЕРАЦИЯ ОТВЕТОВ С LLM")
    print("="*80)
    
    # Загрузка test данных
    data_path = Path(PATHS['extracted_data'])
    test_file = list(data_path.rglob('*test_sample*.csv'))[0]
    test_df = pd.read_csv(test_file)
    print(f"Загружено тестовых запросов: {len(test_df)}")
    
    # Загрузка RAG
    print("\nЗагрузка RAG индекса...")
    rag = ArxivRAG("all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rag.load_index('data/index')
    
    # Оценка retrieval
    print("\nОценка retrieval...")
    results = evaluate_retrieval(rag, test_df, k=5)
    predictions = results['predictions']
    
    # Генерация ответов
    llm = LLMGenerator("Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=256)
    generate_answers(rag, llm, test_df, predictions, num_examples=5, top_k=3)
    
    llm.clear_memory()
    print("\nГенерация завершена")