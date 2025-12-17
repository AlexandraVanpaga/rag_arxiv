# RAG System for arXiv 

Retrieval-Augmented Generation (RAG) система для поиска и генерации ответов на основе научных статей arXiv.

## Результаты

| Метрика | Значение |
|---------|----------|
| **MRR@5** | **0.9398** |
| Точность @1 | 91.2% |
| Среднее время | ~50 ms/запрос |
| База документов | 98,213 статей |

## Архитектура
```
Query → Embedder → FAISS Search (топ-50) → Reranker (топ-5) → LLM → Answer
```

**Компоненты:**
- **Embedder**: `all-MiniLM-L6-v2` (80MB) - семантическое представление
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (130MB) - переранжирование
- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct` (3GB) - генерация ответов
- **Vector DB**: FAISS IndexFlatIP - косинусное сходство

## Быстрый старт

### Установка
```bash
# Клонирование репозитория
git clone https://github.com/AlexandraVanpaga/rag_arxiv.git
cd rag_arXiv

# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Подготовка данных
```bash
# 1. Скачивание данных с Яндекс.Диска
python -m src.get_raw_data

# 2. Предобработка данных
python -m src.preprocessing
```

### Запуск RAG
```bash
# 3. Построение индекса и оценка на тестовых данных
python -m src.rag_model
python -m src.init_and_eval
# 4. Генерация ответов с LLM
python -m src.generation
```


## Анализ производительности

### Распределение позиций правильных ответов

| Позиция | Количество | Процент |
|---------|------------|---------|
| 1 место | 912 | 91.2% |
| 2 место | 42 | 4.2% |
| 3 место | 14 | 1.4% |
| 4 место | 7 | 0.7% |
| 5 место | 2 | 0.2% |
| Не найдено | 23 | 2.3% |
