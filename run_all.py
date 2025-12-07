import subprocess
import time
import os
import signal

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(BASE_DIR, "streamlit_frontend", "app.py")

# Start Django backend
backend = subprocess.Popen(["python", "manage.py", "runserver"], cwd=BASE_DIR)

# Wait for backend to start
time.sleep(3)

# Start Streamlit frontend
frontend = subprocess.Popen(["streamlit", "run", FRONTEND_PATH], cwd=BASE_DIR)

try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("Shutting down both servers...")
    backend.send_signal(signal.SIGINT)
    frontend.send_signal(signal.SIGINT)
