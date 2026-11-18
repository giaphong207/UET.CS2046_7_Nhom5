"""From research/lead_modeling/a4_a5: python scripts/run_a4.py --data data/<csv>.csv.

From repository root, prefix both script and relative data/output paths with
research/lead_modeling/a4_a5/.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diabetes_research.modeling import main
if __name__=='__main__':main()
