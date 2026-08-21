"""
Entry point for the Agentic AI Interview Manager.

Usage:
    python main.py

This script:
1. Loads environment variables from .env
2. Initializes the SQLite database
3. Checks ChromaDB — if empty, loads HuggingFace datasets
4. Launches the Streamlit UI
"""

import os
import sys
import subprocess

# Ensure we run from the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv()

from database.override_log import init_db
from rag.vectorstore import collection_count


def check_and_load_datasets():
    """Load HuggingFace datasets into ChromaDB if not already done."""
    collections_to_check = ["resumes", "job_descriptions", "interview_questions"]
    needs_loading = any(collection_count(c) == 0 for c in collections_to_check)

    if needs_loading:
        print("\nChromaDB collections are empty. Loading HuggingFace datasets...")
        print("This only happens on the first run.\n")
        from data.load_datasets import load_all_datasets
        load_all_datasets()
    else:
        print("ChromaDB collections already populated. Skipping dataset loading.")


def launch_streamlit():
    """Launch the Streamlit app."""
    app_path = os.path.join(PROJECT_DIR, "ui", "app.py")
    print(f"\nLaunching Streamlit UI: {app_path}")
    print("Open your browser at http://localhost:8501\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "false"],
        cwd=PROJECT_DIR,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  AGENTIC AI INTERVIEW MANAGER")
    print("=" * 60)

    # Step 1: Initialize SQLite
    print("\n[1/3] Initializing SQLite database...")
    init_db()
    print("      SQLite ready.")

    # Step 2: Load datasets if needed
    print("\n[2/3] Checking ChromaDB collections...")
    check_and_load_datasets()

    # Step 3: Launch UI
    print("\n[3/3] Launching Streamlit UI...")
    launch_streamlit()
