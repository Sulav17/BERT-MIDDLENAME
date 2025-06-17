FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose API port
EXPOSE 8347

# Default environment variable
ENV PORT=8347

# Entrypoint
ENTRYPOINT ["sh", "./entrypoint.sh"]