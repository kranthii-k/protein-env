"""
inference.py — ProteinEnv baseline inference script.

Hackathon requirement: must be at project root, named exactly inference.py.
Reads credentials from environment variables:
  API_BASE_URL  — LLM API base URL (OpenAI-compatible)
  MODEL_NAME    — Model identifier String
  HF_TOKEN      — Hugging Face token (for ESM2 model download)

Uses OpenAI client for ALL LLM calls (no direct HF calls from this script).
Connects to the running ProteinEnv server at localhost:8000.
Must complete all 3 tasks in < 20 minutes on 2vCPU / 8GB RAM.
Emits structured [START] / [STEP] / [END] logs for the evaluator.
"""

import os
import sys
import json
import signal
import logging

import openai
from dotenv import load_dotenv

# Allow running from project root without package install
sys.path.insert(0, os.path.dirname(__file__))

try:
    from client import ProteinEnvClient
except ImportError:
    # client.py is injected by the OpenEnv orchestrator at eval time
    ProteinEnvClient = None  # type: ignore

from models import ProteinAction

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME")
HF_TOKEN     = os.environ.get("HF_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # raw — evaluator parses stdout directly
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

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

GO_HINT = (
    "\n\nFor GO terms, predict using format GO:XXXXXXX (7 digits).\n"
    "Common molecular function terms: GO:0003700, GO:0005515, GO:0003677\n"
    "Common biological process terms: GO:0006915, GO:0043065, GO:0008150\n"
    "Common cellular component terms: GO:0005634, GO:0005737, GO:0005829"
)

MAX_STEPS_PER_EPISODE = 10


def timeout_handler(signum, frame):  # noqa: ANN001
    log.error("Maximum execution time exceeded. Terminating.")
    sys.exit(1)


def main() -> None:
    # ── SIGALRM timeout (Linux / macOS only) ──────────────────────────────
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(18 * 60)  # 18-minute hard cap, well under 20-min limit

    # ── Validate env vars ─────────────────────────────────────────────────
    for var, val in [("API_BASE_URL", API_BASE_URL), ("MODEL_NAME", MODEL_NAME), ("HF_TOKEN", HF_TOKEN)]:
        if not val:
            log.error("Missing required environment variable: %s", var)
            sys.exit(1)

    if ProteinEnvClient is None:
        log.error("client.py not found — the OpenEnv orchestrator must inject it.")
        sys.exit(1)

    openai_client = openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ProteinEnvClient(base_url="http://localhost:8000")

    tasks  = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}

    for task in tasks:
        # ── [START] ───────────────────────────────────────────────────────
        print(f"[START] task={task}", flush=True)

        obs = env.reset(task_type=task)

        sys_prompt = SYSTEM_PROMPT
        if task == "medium":
            sys_prompt += GO_HINT

        messages    = [{"role": "system", "content": sys_prompt}]
        final_reward = 0.0
        step_num     = 0

        for step_num in range(1, MAX_STEPS_PER_EPISODE + 1):
            messages.append({
                "role":    "user",
                "content": f"Observation: {obs.model_dump_json()}",
            })

            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            messages.append({"role": "assistant", "content": raw})

            action_dict = json.loads(raw)
            action      = ProteinAction(**action_dict)

            # ── [STEP] ────────────────────────────────────────────────────
            print(
                f"[STEP] task={task} step={step_num} "
                f"action={action.action_type}",
                flush=True,
            )

            result = env.step(action)
            obs    = result.observation

            if result.done:
                final_reward = result.reward
                break

        scores[task] = final_reward

        # ── [END] ─────────────────────────────────────────────────────────
        print(
            f"[END] task={task} score={final_reward:.3f} "
            f"steps={step_num} status=done",
            flush=True,
        )

    # ── Summary table ─────────────────────────────────────────────────────
    print("", flush=True)
    print("=" * 45, flush=True)
    print("ProteinEnv Baseline Inference Results", flush=True)
    print("=" * 45, flush=True)
    for task in tasks:
        score = scores.get(task, 0.0)
        emoji = "🟢" if score >= 0.9 else ("🟡" if score >= 0.4 else "🔴")
        print(f"  {task:<6}  {score:.3f}  {emoji}", flush=True)
    print("-" * 45, flush=True)
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Mean    {mean:.3f}", flush=True)
    print("=" * 45, flush=True)


if __name__ == "__main__":
    main()
