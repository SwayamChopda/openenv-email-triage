FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security (common Hugging Face Space practice)
RUN useradd -m -u 1000 user
USER user

# Copy application code
COPY --chown=user:user . .

# Expose Hugging Face Space default port
EXPOSE 7860

# Command to run the application
RUN echo "busting broken Hugging Face builder cache layer v2"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
