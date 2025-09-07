FROM python:3.13-slim

RUN pip install --upgrade pip && \
    apt-get update && \
    apt-get upgrade -y

WORKDIR /app

RUN apt-get update && apt-get install -y \
  cmake \
  build-essential \
  libboost-all-dev \
  libgl1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 4000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4000"]
