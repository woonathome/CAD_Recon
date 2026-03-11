import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 설정 (청크 번호 범위 지정)
# ==========================================
START_CHUNK = 71
END_CHUNK = 99

BASE_URL = "https://archive.nyu.edu"
COLLECTION_URL = "https://archive.nyu.edu/handle/2451/43778?rpp=100&value=1&data-order=asc"
DOWNLOAD_DIR = "abc_dataset_download"

TARGET_PATTERN = re.compile(r'(step|meta)_v\d{2}\.7z')

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_file(url, filename, pos):
    save_path = os.path.join(DOWNLOAD_DIR, filename)
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with tqdm(total=1, desc=f"[Skip] {filename}", position=pos, leave=True, bar_format='{desc}') as bar:
            pass
        return

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            position=pos,
            leave=True
        ) as bar:
            for data in response.iter_content(chunk_size=1048576): 
                size = file.write(data)
                bar.update(size)
                
    except Exception as e:
        with tqdm(total=1, desc=f"[오류] {filename}: {e}", position=pos, leave=True, bar_format='{desc}') as bar:
            pass
        if os.path.exists(save_path):
            os.remove(save_path)

def main():
    print(f"[{START_CHUNK} ~ {END_CHUNK}] 번호에 해당하는 청크(Chunk) 데이터만 탐색합니다...\n")
    
    res = requests.get(COLLECTION_URL)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    chunk_links = {}
    
    for a_tag in soup.find_all('a', href=True):
        text = a_tag.text.strip()
        if 'ABC Dataset Chunk' in text:
            try:
                chunk_num = int(text.split()[-1])
                if START_CHUNK <= chunk_num <= END_CHUNK:
                    chunk_links[chunk_num] = urljoin(BASE_URL, a_tag['href'])
            except ValueError:
                continue

    print(f"조건에 맞는 청크 페이지를 총 {len(chunk_links)}개 찾았습니다.\n")

    chunk_tasks = {}
    
    for idx, (chunk_num, chunk_url) in enumerate(sorted(chunk_links.items())):
        print(f"\r내부 파일 탐색 중... [{idx+1}/{len(chunk_links)}]", end="")
        try:
            chunk_res = requests.get(chunk_url)
            chunk_soup = BeautifulSoup(chunk_res.text, 'html.parser')
            
            # ★ 변경된 부분: 리스트 대신 딕셔너리를 사용하여 파일명 중복 제거 ★
            unique_tasks = {}
            for a_tag in chunk_soup.find_all('a', href=True):
                href = a_tag['href']
                if TARGET_PATTERN.search(href):
                    download_url = urljoin(BASE_URL, href)
                    filename = download_url.split('/')[-1].split('?')[0]
                    # 파일명이 이미 딕셔너리에 없다면 추가 (중복 원천 차단)
                    if filename not in unique_tasks:
                        unique_tasks[filename] = download_url
            
            # 딕셔너리에 모인 고유한 파일들만 리스트로 변환
            if unique_tasks:
                chunk_tasks[chunk_num] = [(url, fname) for fname, url in unique_tasks.items()]
        except:
            pass

    print("\n\n탐색 완료! 묶음 단위 다운로드를 시작합니다.\n")
    print("=" * 60)
    
    for chunk_num in sorted(chunk_tasks.keys()):
        tasks = chunk_tasks[chunk_num]
        print(f"▶ Chunk {chunk_num:04d} 다운로드 시작 ({len(tasks)}개 파일 동시 진행)")
        
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = []
            for i, (url, filename) in enumerate(tasks):
                futures.append(executor.submit(download_file, url, filename, i))
            
            for future in as_completed(futures):
                future.result() 
        
        print("\n" * (len(tasks) - 1))
        print(f"✔ Chunk {chunk_num:04d} 완료! 다음 순번으로 넘어갑니다.\n")
        print("-" * 60)

    print("모든 청크 다운로드가 완벽하게 종료되었습니다!")

if __name__ == "__main__":
    main()