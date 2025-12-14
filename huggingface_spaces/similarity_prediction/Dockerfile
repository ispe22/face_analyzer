FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix permissions to allow writing (for unzipping)
RUN chmod -R 777 /app

CMD ["python", "app.py"]
