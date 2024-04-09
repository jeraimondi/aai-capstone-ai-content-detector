# NOTE* - Code retrieved and modified from:
# https://learn.microsoft.com/en-us/azure/developer/python/tutorial-containerize-simple-web-app-for-app-service?tabs=web-app-flask

# Gunicorn configuration file
import multiprocessing

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:50505"

# commented out to disable multiprocessing
#workers = (multiprocessing.cpu_count() * 2) + 1
#threads = workers

timeout = 120