FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml requirements.txt /app/
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY notebooks /app/notebooks

RUN python -m pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "scripts/run_predict.py"]
