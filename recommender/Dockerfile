# 청춘일기장 추천 서비스 Docker 이미지
# 빌드:  docker build -t diary-recsys .
# 실행:  docker run -p 8000:8000 diary-recsys
# 접속:  http://localhost:8000/docs

FROM python:3.11-slim

# Kiwi 빌드에 필요할 수 있는 도구 (안전하게 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (코드만 바뀌면 이 레이어는 캐시 재사용 → 빌드 빠름)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY diary_recommender_service.py .

# 클라우드(Render 등)는 PORT 환경변수로 포트를 줌. 없으면 8000.
ENV PORT=8000
EXPOSE 8000

# 0.0.0.0 으로 떠야 외부에서 접속 가능 (localhost 만 두면 컨테이너 밖에서 못 봄)
CMD ["sh", "-c", "uvicorn diary_recommender_service:app --host 0.0.0.0 --port ${PORT}"]
