# ProteinEnv 🧬

> **Meta PyTorch + Hugging Face OpenEnv Hackathon Submission**
> **Team / Author:** ESM2 Team 
> **Deadline:** April 8th 2026, 11:59 PM IST

ProteinEnv is an OpenEnv reinforcement learning environment for training AI agents to predict protein function, Gene Ontology terms, and disease-variant associations using Meta's ESM2 protein language model embeddings.

---

## 🎯 Task Tiers

The environment provides 3 tiered difficulties for evaluating agent performance:

*   **Easy:** `protein_family_classification` - Classify a well-known protein into its correct protein family.
*   **Medium:** `go_term_prediction` - Predict Gene Ontology terms for a moderately characterized protein.
*   **Hard:** `disease_variant_association` - Identify pathogenicity and disease associations for a missense variant.

Agents are given a raw amino acid sequence and must learn to use the `get_esm2_embedding` tool iteratively to gather representations of the sequence to inform their final submission.

## 📡 OpenEnv Interface

### Observation Space
The `ProteinObservation` includes:
- **`protein_id`**: UniProt accession.
- **`sequence`**: Amino-acid sequence.
- **`variant_info`**: Missense variant details (wildtype, mutant, position) for hard-tier tasks.
- **`task_description`**: tier-specific natural language instructions.
- **`available_tools`**: `["get_esm2_embedding"]`.

### Action Space
Agents interact via `ProteinAction`:
- **`CALL_TOOL`**: Invoke `get_esm2_embedding` with a sequence.
- **`SUBMIT_PREDICTION`**: Submit final values:
  - `predicted_family` (Easy)
  - `predicted_go_terms` (Medium)
  - `predicted_pathogenicity` & `predicted_diseases` (Hard)

## 🛠 Project Structure

*   **/core:** ESM2 embedding client, state manager, and reward calculators.
*   **/data:** Fixtures across the 3 task difficulties.
*   **/graders:** Hackathon-compliant graders checking for variants, GO terms, and protein families safely.
*   **/server:** FastAPI OpenEnv app orchestrator.
*   `inference.py`: Baseline zero-shot completion inference script.
*   `openenv.yaml`: Validated OpenEnv specification.
*   `models.py`: Strongly typed Pydantic v2 schemas used across the environment.

## 🚀 Setup & Installation

### 1. Requirements

*   Python 3.12+
*   Hugging Face Token (for weights download)
*   LLM API Base URL (OpenAI-compatible scaled endpoints)

### 2. Local Installation

Using `uv` (recommended) or `pip`:
```bash
git clone https://github.com/your-username/protein_env.git
cd protein_env

uv pip install -r requirements.txt
# OR
pip install pydantic fastapi uvicorn pytest pytest-cov openai
```

### 3. Environment Variables
Create a `.env` file or export the following in your terminal:
```bash
export API_BASE_URL="https://api.openai.com/v1"  # Or your evaluator endpoint
export MODEL_NAME="gpt-4o"                       # Target baseline model
export HF_TOKEN="hf_..."                         # Required for ESM2 model init
```

## 🧠 Running the Environment

### Server
Start the OpenEnv environment server locally:
```bash
cd server
docker compose up --build
# OR locally via: uvicorn server.app:app --host 0.0.0.0 --port 8000
```
This serves the environment endpoints and lazily loads `facebook/esm2_t6_8M_UR50D` on the first embedding request.

### Inference Baseline Assessment
To test the environment on all 3 difficulties and print the required hackathon benchmark formatted tables, run:
```bash
python3 inference.py
```
> Ensure your Server is running at `localhost:8000` before running inference.

## 🧪 Testing

We strictly adhere to zero side-effect design patterns with 80%+ coverage metrics.
```bash
python3 -m pytest tests/unit/ -v --tb=short
python3 -m pytest -v tests/unit/ --cov=. --cov-report=term-missing
```

## ☁️ Deployment

**Hugging Face Space:** [https://huggingface.co/spaces/vibe-paglu/protein-env](https://huggingface.co/spaces/vibe-paglu/protein-env)
> The environment is live and OpenEnv-validated at this endpoint.

## 📖 License & Acknowledgements 
*   **Model:** Powered by [Meta ESM2](https://github.com/facebookresearch/esm) (`facebook/esm2_t6_8M_UR50D`).
*   **Data:** Sequences derived from UniProt Swiss-Prot (CC BY 4.0), GO annotations from QuickGO (CC0 1.0), and ClinVar (Public Domain).
*   **Framework:** Built for the OpenEnv Hackathon spec.
