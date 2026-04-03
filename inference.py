"""
inference.py — ProteinEnv baseline inference script.

Hackathon requirement: must be at project root, named exactly inference.py.
Reads credentials from environment variables:
  API_BASE_URL  — LLM API base URL (OpenAI-compatible)
  MODEL_NAME    — Model identifier String
  HF_TOKEN      — Hugging Face token (for ESM2 model download)

Uses OpenAI client for ALL LLM calls (no direct HF calls from this script).
Connects to the running ProteinEnv server and runs one episode per task.
Must complete all 3 tasks in < 20 minutes on 2vCPU / 8GB RAM.
Produces a reproducible baseline score on all 3 tasks.
"""

import os
import sys
import json
import signal
import logging

import openai

# Use the client.py from Phase 1
sys.path.insert(0, os.path.dirname(__file__))
try:
    from client import ProteinEnvClient
except ImportError:
    # Handle environment where client.py is injected during test run
    pass

from models import ProteinAction

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")

SYSTEM_PROMPT = """You are a protein biology expert AI agent. You will be given a protein sequence
and a task. You have access to one tool: get_esm2_embedding, which returns a
320-dimensional embedding vector for any amino acid sequence.

For each task, respond with a JSON object in exactly this format:
{
  "action_type": "submit_prediction" | "call_tool",
  "tool_name": "get_esm2_embedding",        (only if action_type is call_tool)
  "tool_args": {"sequence": "<AA sequence>"}, (only if action_type is call_tool)
  "predicted_family": "<family name>",       (only for easy task)
  "predicted_go_terms": ["GO:XXXXXXX", ...], (only for medium task)
  "predicted_pathogenicity": "<value>",      (only for hard task)
  "predicted_diseases": ["<disease>", ...],  (only for hard task)
  "reasoning": "<your reasoning>"
}

Valid pathogenicity values: Pathogenic, Likely pathogenic,
Variant of Uncertain Significance, Likely benign, Benign"""

MAX_STEPS_PER_EPISODE = 10

def timeout_handler(signum, frame):
    logging.error("Maximum execution time of 15 minutes exceeded. Terminating.")
    sys.exit(1)

def main():
    if hasattr(signal, 'alarm'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15 * 60)

    try:
        openai_client = openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        env = ProteinEnvClient(base_url="http://localhost:8000")
        
        scores = {}
        tasks = ["easy", "medium", "hard"]
        
        for task in tasks:
            obs = env.reset(task_type=task)
            
            sys_prompt = SYSTEM_PROMPT
            if task == "medium":
                sys_prompt += "\n\nFor GO terms, predict using format GO:XXXXXXX (7 digits).\n"
                sys_prompt += "Common molecular function terms: GO:0003700, GO:0005515, GO:0003677\n"
                sys_prompt += "Common biological process terms: GO:0006915, GO:0043065, GO:0008150\n"
                sys_prompt += "Common cellular component terms: GO:0005634, GO:0005737, GO:0005829"
                
            messages = [{"role": "system", "content": sys_prompt}]
            final_reward = 0.0
            
            for _ in range(MAX_STEPS_PER_EPISODE):
                # Send observation to LLM (include task_description from observation)
                messages.append({
                    "role": "user",
                    "content": f"Observation: {obs.model_dump_json()}"
                })
                
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                
                raw_response = response.choices[0].message.content
                messages.append({"role": "assistant", "content": raw_response})
                
                # Parse LLM response into a ProteinAction
                action_dict = json.loads(raw_response)
                action = ProteinAction(**action_dict)
                
                # Call env.step(action)
                result = env.step(action)
                obs = result.observation
                
                if result.done:
                    final_reward = result.reward
                    break
                    
            # Record final reward for this task
            scores[task] = final_reward
            
        print("═══════════════════════════════════════════")
        print("ProteinEnv Baseline Inference Results")
        print("═══════════════════════════════════════════")
        for task in tasks:
            score = scores.get(task, 0.0)
            print(f"Task: {task:<6} | Score: {score:.3f} | Status: DONE")
        print("───────────────────────────────────────────")
        mean_score = sum(scores.values()) / len(scores) if scores else 0.0
        print(f"Mean Score: {mean_score:.3f}")
        print("═══════════════════════════════════════════")
        
        sys.exit(0)
    except Exception as e:
        logging.error(f"Inference failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
