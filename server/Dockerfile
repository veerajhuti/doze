FROM python:3.10-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  libx11-dev \
  libatlas-base-dev \
  libgtk-3-dev \
  libboost-python-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install dlib==19.24.0

RUN pip install -r requirements.txt

COPY . .

EXPOSE 4000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4000"]