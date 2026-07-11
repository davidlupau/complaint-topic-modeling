import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(file_name: str) -> pd.DataFrame | None:
    """Load Customer Complaints Sentiment and Priority CSV from the project data directory.

    Parameters:
        file_name (str): Name of the CSV file inside the project's data/ folder.

    Returns:
        pd.DataFrame | None: Loaded DataFrame, or None if the file is not found
            or an error occurs.
    """
    print("\nLoading dataset...\n")
    try:
        # Get the project root (go up from src/)
        project_root = Path(__file__).parent.parent
        data_file = project_root / "data" / file_name

        # Check if file exists first
        if not data_file.exists():
            print(f"File not found: {data_file}")
            return None

        df = pd.read_csv(data_file)
        print(f"Successfully loaded {file_name} \n")
        return df
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

