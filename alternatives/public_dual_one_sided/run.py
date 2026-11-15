"""One explicit local reproduction, without training, downloading or submission."""
import argparse
from pathlib import Path
from dual_yolo.lifecycle import arm
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True)
    p.add_argument('--test-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--python',type=Path,required=True)
    p.add_argument('--entry-epoch',type=float,required=True);p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args()
    arm(min(a.entry_epoch+300,a.deadline_epoch-2400))
    from dual_yolo.pipeline import reproduce
    reproduce(Path(__file__).resolve().parent,a.test_dir,a.output,a.python,a.entry_epoch,a.deadline_epoch)

