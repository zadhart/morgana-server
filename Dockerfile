FROM python:3.8-buster

WORKDIR /app

RUN pip3 install torch torchvision torchaudio

RUN pip install transformers

RUN pip install flask

COPY . .

ENV PORT=5000

EXPOSE 5000

CMD ["python", "main.py"]
