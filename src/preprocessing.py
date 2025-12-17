# src/preprocessing.py

import json
import pandas as pd
from pathlib import Path
from config import PATHS


def load_raw_data() -> pd.DataFrame:
    """Загрузка сырых данных"""
    data_path = Path(PATHS['extracted_data'])
    metadata_file = list(data_path.rglob('*arxiv-metadata*.json'))[0]
    
    print(f"Загрузка из: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Загружено записей: {len(df)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка и подготовка данных"""
    # Оставляем только нужные колонки
    df = df[['id', 'title', 'abstract']].copy()
    
    # Убираем пустые значения
    df['abstract'] = df['abstract'].fillna('')
    df['title'] = df['title'].fillna('')
    
    # Убираем записи без abstract (нечего индексировать)
    df = df[df['abstract'].str.len() > 50].copy()
    
    # Базовая очистка текста
    df['abstract'] = df['abstract'].str.strip()
    df['title'] = df['title'].str.strip()
    
    # Убираем дубликаты по id
    df = df.drop_duplicates(subset='id', keep='first')
    
    print(f"После очистки: {len(df)} записей")
    return df


def save_processed_data(df: pd.DataFrame, format: str = 'parquet'):
    """Сохранение обработанных данных"""
    output_path = Path(PATHS['processed_data'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        output_file = output_path / 'metadata.parquet'
        df.to_parquet(output_file, index=False)
    elif format == 'json':
        output_file = output_path / 'metadata.json'
        df.to_json(output_file, orient='records', lines=True)
    
    print(f"Сохранено в: {output_file}")
    print(f"Размер файла: {output_file.stat().st_size / 1024**2:.2f} MB")


def preprocess_pipeline():
    """Полный pipeline предобработки"""
    print("="*70)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*70)
    
    # 1. Загрузка
    df = load_raw_data()
    
    # 2. Очистка
    df = clean_data(df)
    
    # 3. Статистика
    print(f"\nСтатистика:")
    print(f"  • Записей: {len(df)}")
    print(f"  • Средняя длина abstract: {df['abstract'].str.len().mean():.0f} символов")
    print(f"  • Средняя длина title: {df['title'].str.len().mean():.0f} символов")
    
    # 4. Сохранение
    save_processed_data(df, format='parquet')
    
    return df


if __name__ == "__main__":
    df = preprocess_pipeline()
    print("\nПредобработка завершена")