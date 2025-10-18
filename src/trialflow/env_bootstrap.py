# src/trialflow/env_bootstrap.py
import os
from dotenv import load_dotenv
load_dotenv(os.getenv("ENV_FILE", ".env"))