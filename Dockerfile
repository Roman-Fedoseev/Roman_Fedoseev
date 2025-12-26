# Базовый образ Python
FROM python:3.10-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла с зависимостями
COPY requirements.txt .

# Установка всех необходимых пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всех файлов проекта в контейнер
COPY . .

# Команда запуска вашего решения
CMD ["python", "main.py"]
