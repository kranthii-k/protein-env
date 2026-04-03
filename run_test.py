import os
import time
import subprocess
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# We set explicit HF models to bypass OpenAI requirement if API_BASE_URL is missing
if not os.environ.get("API_BASE_URL"):
    os.environ["API_BASE_URL"] = "https://api-inference.huggingface.co/v1/"
if not os.environ.get("MODEL_NAME"):
    os.environ["MODEL_NAME"] = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Starting server... (Model: {os.environ.get('MODEL_NAME')})")
server_proc = subprocess.Popen(
    [os.sys.executable, "-m", "uvicorn", "server.app:app", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to initialize
time.sleep(5)

print("Starting inference script...")
try:
    result = subprocess.run([os.sys.executable, "inference.py"], text=True, capture_output=True)
    print("--- INFERENCE STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- INFERENCE STDERR ---")
        print(result.stderr)
finally:
    print("Shutting down server...")
    server_proc.terminate()
    server_proc.wait()
    print("Done test.")
