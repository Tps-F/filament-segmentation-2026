"""Execute every code cell of the shipped independent reproduction notebook."""
import json,sys,time
from pathlib import Path
from dual_yolo.lifecycle import arm
from dual_yolo.common import require,save,digest
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    args=sys.argv[1:];require('--entry-epoch' in args and '--deadline-epoch' in args,'Original phase clock required')
    entry=float(args[args.index('--entry-epoch')+1]);end=float(args[args.index('--deadline-epoch')+1]);arm(min(entry+300,end-2400))
    notebook=json.loads((ROOT/'reproduce.ipynb').read_text());cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    require(len(cells)==1 and cells[0]['source']==['import runpy\n','runpy.run_path("run.py", run_name="__main__")\n'],'Shipped notebook code differs')
    sys.argv=[str(ROOT/'run.py'),*args];exec(compile(''.join(cells[0]['source']),str(ROOT/'reproduce.ipynb'),'exec'),{'__name__':'__main__'})
    output=Path(args[args.index('--output')+1]);save(output/'notebook_execution.json',{'notebook_sha256':digest(ROOT/'reproduce.ipynb'),
      'code_cells_executed':1,'all_code_cells_succeeded':True,'completed_epoch':time.time()})

