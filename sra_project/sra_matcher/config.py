"""Runtime configuration.

Model tiering is the cost lever from the design: a cheap model does the
high-volume structuring/parsing, a stronger one does the judgment call.
Swap the defaults for whatever Gemini is current in your project.
"""
import os

APP_NAME = "sra_matcher"
DEFAULT_USER_ID = "eval-user"

# Cheap model: intake structuring (and, if you split it out, per-criterion parsing).
INTAKE_MODEL = os.getenv("SRA_INTAKE_MODEL", "gemini-3.1-flash-lite")

# Stronger model: the eligibility judgment where a wrong call has a cost.
REASONER_MODEL = os.getenv("SRA_REASONER_MODEL", "gemini-3.7-flash")

# Coarse-retrieval page size = size of the candidate set the reasoner scores.
MAX_CANDIDATES = int(os.getenv("SRA_MAX_CANDIDATES", "25"))
