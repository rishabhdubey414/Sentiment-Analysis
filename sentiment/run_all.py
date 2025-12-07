import subprocess
import time
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(BASE_DIR, "streamlit_frontend", "app.py")

# Start Django backend
backend = subprocess.Popen(["python", "manage.py", "runserver"], cwd=BASE_DIR)

# Wait for backend to start
time.sleep(3)

# Start Streamlit frontend
frontend = subprocess.Popen(["streamlit", "run", FRONTEND_PATH], cwd=BASE_DIR)

# Keep both running
backend.wait()
frontend.wait()
