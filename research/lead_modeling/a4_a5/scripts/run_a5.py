"""A5 V3 entrypoint; imports the single repository-root src package.

Requires a fresh, matching V3 A4 run. It never refits or selects models.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'src'))
from diabetes_research.analysis import main
if __name__=='__main__':main()
