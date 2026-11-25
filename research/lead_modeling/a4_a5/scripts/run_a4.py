"""A4 V3 entrypoint; imports the single repository-root src package.

Relative CLI data/output arguments use the command's working directory.
Primary SelectFromModel is implemented and software-tested.
Official training/evaluation remains pending a separate authorization release.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'src'))
from diabetes_research.modeling import main
if __name__=='__main__':main()
