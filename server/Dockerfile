FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
  build-essential cmake libboost-all-dev libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN sed '/torch/d;/torchvision/d' requirements.txt > requirements_no_torch.txt
RUN pip install --no-cache-dir -r requirements_no_torch.txt

COPY . .

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app /app

EXPOSE 4000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4000"]
