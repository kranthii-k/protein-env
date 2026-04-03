import os
import sys
import json
import openai

# 1. Mock the OpenAI API completely so we don't hit the failed HF gated models
class DummyChoice:
    class DummyMessage:
        content = json.dumps({
            "action_type": "submit_prediction", 
            "predicted_family": "Kinase", 
            "predicted_go_terms": ["GO:0003700"], 
            "predicted_pathogenicity": "Benign", 
            "reasoning": "Mocked response verifying local loop logic."
        })
    message = DummyMessage()

class DummyCompletions:
    def create(self, **kwargs):
        # Scan the messages payload exactly as it's sent to the LLM
        for msg in kwargs.get("messages", []):
            if "For GO terms, predict using format GO:XXXXXXX" in msg["content"]:
                print("\n[✔] HACKATHON HINT SUCCESSFULLY DETECTED IN MEDIUM TASK PROMPT:")
                print("    " + msg["content"].split("\n\nFor GO terms")[1].replace("\n", "\n    "))
        
        class DummyResp:
            choices = [DummyChoice()]
        return DummyResp()

class DummyChat:
    completions = DummyCompletions()

class DummyClient:
    def __init__(self, *args, **kwargs):
        self.chat = DummyChat()

openai.OpenAI = DummyClient

# 2. Set Env vars so inference.py doesn't crash
os.environ["API_BASE_URL"] = "http://dummy.url"
os.environ["MODEL_NAME"] = "mock-model"
os.environ["HF_TOKEN"] = "mock_hf_token"

# 3. Call the inference script main function
print("Initiating full mocked end-to-end local test...")
from inference import main
try:
    main()
except SystemExit:
    pass
