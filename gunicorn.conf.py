import multiprocessing
import os

bind = "127.0.0.1:8448"

# Количество воркеров: 2 * CPU + 1
workers = int(os.environ.get("GUNICORN_WORKERS") or (multiprocessing.cpu_count() * 2 + 1))

# Лёгкие тредовые воркеры для смешанной нагрузки IO/CPU
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS") or 2)

# Тайминги и устойчивость
timeout = int(os.environ.get("GUNICORN_TIMEOUT") or 30)
graceful_timeout = 30
keepalive = 15
max_requests = 500
max_requests_jitter = 50

# Быстрый tmp в памяти (меньше IO)
worker_tmp_dir = "/dev/shm"

# Логи в stdout/stderr (смотрим через journalctl)
loglevel = "info"
accesslog = "-"
errorlog = "-"


