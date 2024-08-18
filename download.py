import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 기본 URL 설정
base_url = "https://tomcat.apache.org/tomcat-10.1-doc/"
download_dir = "tomcat_docs"

# 다운로드 디렉토리 생성
os.makedirs(download_dir, exist_ok=True)

# 페이지 로드
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

# 모든 링크 찾기
links = soup.find_all("a")

for link in links:
    href = link.get("href")
    if href and href.endswith(".html"):  # .html 파일만 찾기
        file_url = urljoin(base_url, href)
        file_path = os.path.join(download_dir, os.path.basename(href))
        
        # HTML 파일 다운로드
        with requests.get(file_url) as file_response:
            with open(file_path, "wb") as file:
                file.write(file_response.content)
        
        print(f"Downloaded: {file_url}")