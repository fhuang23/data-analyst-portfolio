"""Runtime configuration.

Model tiering is the cost lever from the design: a cheap model does the
high-volume structuring/parsing, a stronger one does the judgment call.
Swap the defaults for whatever Gemini is current in your project.
"""
import os

APP_NAME = "sra_matcher"
DEFAULT_USER_ID = "eval-user"

# Cheap model: intake structuring (and, if you split it out, per-criterion parsing).
INTAKE_MODEL = os.getenv("SRA_INTAKE_MODEL", "gemini-2.5-flash")

# Stronger model: the eligibility judgment where a wrong call has a cost.
REASONER_MODEL = os.getenv("SRA_REASONER_MODEL", "gemini-2.5-pro")

# Coarse-retrieval page size = size of the candidate set the reasoner scores.
MAX_CANDIDATES = int(os.getenv("SRA_MAX_CANDIDATES", "25"))

# --- self-hosted contender (fine-tuned Qwen via vLLM) ------------------------
# Point at your local OpenAI-compatible vLLM server. QWEN_MODEL is the model or
# LoRA-module name you served it under (e.g. "sra" if you used --lora-modules).
VLLM_BASE_URL = os.getenv("SRA_VLLM_BASE_URL", "http://localhost:8000")
QWEN_MODEL = os.getenv("SRA_QWEN_MODEL", "qwen3:4b")
