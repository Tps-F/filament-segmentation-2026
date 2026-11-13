"""CPU author amendment: exact v3 assets plus post-constructor guard correction.

No model export, inference, training, canonical candidate change or publication.
The copied original build_local.py is historical, already-bound provenance.
"""
from pathlib import Path
import hashlib
import json
import os

PACKAGE=Path(__file__).resolve().parents[1]
PARENT=PACKAGE.with_name('own_yolo_s_full_equal_quality_disjoint_v3')
PARENT_SHA='bcf32206ed70ef6aee8c904cb8477bb1049f102a8527924efcc6ad0c91743a49'
CANONICAL_SHA='9dd513669eee8b12910e001732eea9fdd0f8a8adeb9eacab4421a544da5d9c3c'
MUTABLE={'worker.py','README.txt','recipe.json','test_release.py','notebooks/reproduce_own_s_full_disjoint_v3.ipynb'}

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def expected_worker(original):
    old="require(policy=={'CUBLAS_WORKSPACE_CONFIG':None,'cudnn_benchmark':False,'cudnn_deterministic':False,'deterministic_algorithms':False},'YOLO backend changed during inference')"
    require(original.count(old)==1,'Original final guard not exact')
    updated=original.replace(old,old.replace("'CUBLAS_WORKSPACE_CONFIG':None","'CUBLAS_WORKSPACE_CONFIG':':4096:8'"))
    old="receipt={'role':role,'runtime':runtime(torch),'backend_policy':policy,'raw_sha256':hashes,"
    require(updated.count(old)==1,'Original receipt not exact')
    return updated.replace(old,old+"\n            'backend_entry_policy':{'CUBLAS_WORKSPACE_CONFIG':None,'cudnn_benchmark':False,'cudnn_deterministic':False,'deterministic_algorithms':False},\n            'native_constructor_transition':'Ultralytics Model.__init__ sets CUBLAS_WORKSPACE_CONFIG=:4096:8; no backend setter added by this worker',")

def prepare():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only author operation')
    require(not (PACKAGE/'bundle_preparation.json').exists(),'Never overwrite a frozen v3.1 bundle')
    require(digest(PARENT/'bundle_preparation.json')==PARENT_SHA,'Immutable v3 parent changed')
    parent=read(PARENT/'bundle_preparation.json');copies={}
    for name,row in parent['files'].items():
        require(digest(PARENT/name)==row['sha256'],'Parent file changed:'+name)
        if name not in MUTABLE:
            require(digest(PACKAGE/name)==row['sha256'],'Unchanged asset differs:'+name);copies[name]=row
    require('run.py' in copies and all(n in copies for n in parent['files'] if n.startswith(('src/','weights/'))),'Every inference/geometry/model asset must remain exact')
    require((PACKAGE/'worker.py').read_text()==expected_worker((PARENT/'worker.py').read_text()),'Worker differs beyond exact assertion/reporting correction')
    recipe=read(PACKAGE/'recipe.json');old_recipe=read(PARENT/'recipe.json')
    require(recipe['status']=='BOUND_ACTUAL_FINAL_ARTIFACTS' and recipe['expected_candidate_sha256']==CANONICAL_SHA,'Same actual bound canonical required')
    changed={'kind','detector_environment_phases','v31_parent_bundle_sha256'}
    require(all(recipe[k]==v for k,v in old_recipe.items() if k not in changed),'Original recipe/canonical binding changed')
    trace=read(PACKAGE/'docs/cpu_constructor_policy_trace.json')
    require([r['CUBLAS_WORKSPACE_CONFIG'] for r in trace['trace']]==[None,None,':4096:8'] and trace['CUDA'] is False and trace['inference'] is False,'Missing actual constructor transition evidence')
    require(all(not r[k] for r in trace['trace'] for k in ['cudnn_benchmark','cudnn_deterministic','deterministic_algorithms','cuda_initialized']),'Unexpected CPU native policy transition')
    write(PACKAGE/'docs/v31_exact_copy_provenance.json',{'parent_bundle_sha256':PARENT_SHA,'copied_files':copies,
        'all_model_archives_quality_inference_geometry_exact':True,'worker_change':'Only post-constructor CUBLAS assertion and accurate entry/runtime receipt fields',
        'native_bf16_and_fresh_start_assertion_unchanged':True,'canonical_candidate_sha256':CANONICAL_SHA,'new_exports':False,'GPU':False})
    files={}
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        require(not path.is_symlink(),'No linked release files')
        files[str(path.relative_to(PACKAGE))]={'sha256':digest(path),'bytes':path.stat().st_size}
    write(PACKAGE/'bundle_preparation.json',{'kind':'own_S20_deterministic_semantic_disjoint_v31','files':files,
        'parent_v3_bundle_sha256':PARENT_SHA,'no_new_weight_export':True,'no_new_training':True,
        'full_GPU_reproduction_executed':False,'publication':False})
    output=PACKAGE.parents[1]/'diagnostics/own_s_full_disjoint_v31_release_preparation'
    write(output/'prepared.json',{'status':'BOUND_PARENT_CANDIDATE_GUARD_ONLY_AMENDMENT_NOT_GPU_REPRODUCED',
        'bundle_sha256':digest(PACKAGE/'bundle_preparation.json'),'parent_bundle_sha256':PARENT_SHA,
        'canonical_candidate_sha256':CANONICAL_SHA,'files':len(files),'exact_parent_copies':len(copies),'worker_sha256':digest(PACKAGE/'worker.py'),
        'CUDA':False,'new_exports':False,'GPU_launch_authorized':False})
    print(json.dumps(read(output/'prepared.json')),flush=True)

if __name__=='__main__':prepare()
