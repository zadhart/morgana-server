FROM python:3.8-buster

WORKDIR /app

RUN pip install flask

RUN pip install parlai

COPY . .

ENV PORT=5000

EXPOSE 5000

CMD ["python", "main.py"]