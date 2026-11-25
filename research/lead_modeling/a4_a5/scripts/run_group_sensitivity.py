"""Separate exact-X sensitivity; defaults to outer diagnostics without fitting."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'src'))
from diabetes_research.group_sensitivity import main

if __name__ == '__main__':
    main()
