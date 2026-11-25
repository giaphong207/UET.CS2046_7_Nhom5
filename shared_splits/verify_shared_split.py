"""Repository wrapper for the canonical exact-CSV and B v9 split verifier."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diabetes_research.shared_splits import main

if __name__ == '__main__':
    main()
