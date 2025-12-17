# src/get_raw_data.py
import zipfile
import tarfile
from pathlib import Path
import requests
from config import PATHS, YANDEX_DISK_URL


def download_from_yandex_disk(public_url: str, save_path: Path) -> bool:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        response = requests.get(api_url, params={'public_key': public_url}, timeout=30)
        download_url = response.json().get('href')
        
        if not download_url:
            return False
        
        file_response = requests.get(download_url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Файл сохранён: {save_path}")
        return True
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            mode = 'r:gz' if archive_path.suffix in ['.gz', '.tgz'] else 'r'
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"Разархивировано в: {extract_to}")
        return True
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


if __name__ == "__main__":
    archive_path = Path(PATHS['raw_data'])
    extract_path = Path(PATHS['extracted_data'])
    
    if download_from_yandex_disk(YANDEX_DISK_URL, archive_path):
        extract_archive(archive_path, extract_path)