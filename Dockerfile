FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 MPLBACKEND=Agg
WORKDIR /app
RUN pip install --no-cache-dir 'pandas>=2,<3' 'numpy>=1.24,<3' 'scikit-learn>=1.3,<2' 'matplotlib>=3.7,<4'
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . /app
RUN chown appuser:appuser /app
USER appuser
CMD ["python", "run_pipeline.py"]
