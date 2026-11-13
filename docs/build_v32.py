"""Freeze one CPU-only v3.1 derivative; no export, GPU, fit or publication."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import sys

PACKAGE=Path(__file__).resolve().parents[1]
PARENT=PACKAGE.with_name('own_yolo_s_full_equal_quality_disjoint_v31')
PARENT_SHA='c0380ddf51e507b59c9fb763cadd47cd5b116b113693829676ccb604f1d9cdee'
MUTABLE={'run.py','worker.py','README.txt','recipe.json','test_release.py','THIRD_PARTY_NOTICES.txt',
         'notebooks/reproduce_own_s_full_disjoint_v31.ipynb'}


def require(ok,message):
    if not ok:raise ValueError(message)
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation must hide CUDA')
    require(not (PACKAGE/'bundle_preparation.json').exists(),'Never overwrite a frozen derivative')
    require(digest(PARENT/'bundle_preparation.json')==PARENT_SHA,'Frozen parent changed')
    parent=read(PARENT/'bundle_preparation.json');copies={}
    for name,row in parent['files'].items():
        require(digest(PARENT/name)==row['sha256'],'Parent bytes changed:'+name)
        if name not in MUTABLE:
            require(digest(PACKAGE/name)==row['sha256'],'Unchanged asset differs:'+name);copies[name]=row
    require(all(n in copies for n in parent['files'] if n.startswith(('weights/','src/'))),'Numeric sources/archives must remain exact')
    for name in ['requirements.txt','requirements-notebook.txt','docs/test_inputs.json','docs/canonical_reference.json']:
        require(name in copies,'Required unchanged input/environment source missing')
    sys.path.insert(0,str(PACKAGE));from reference_policy import POLICY
    recipe=read(PACKAGE/'recipe.json');old=read(PARENT/'recipe.json')
    require(recipe['reference_policy']==POLICY and recipe['v32_parent_bundle_sha256']==PARENT_SHA,'Reference contract changed')
    allowed={'kind','limitation','reference_policy','v32_parent_bundle_sha256'}
    require(all(recipe[k]==v for k,v in old.items() if k not in allowed),'Frozen computation/candidate recipe changed')
    license_info=read(PACKAGE/'licenses/ultralytics_source.json')
    require(license_info['version']=='8.4.211' and digest(PACKAGE/'licenses/ULTRALYTICS_AGPL-3.0.txt')==license_info['license_sha256'],
            'Official versioned license text changed')
    for name in ['run.py','worker.py','test_release.py']:
        (PACKAGE/'docs'/(name+'.reference_mode.diff')).write_text(''.join(difflib.unified_diff(
            (PARENT/name).read_text().splitlines(True),(PACKAGE/name).read_text().splitlines(True),
            fromfile='immutable_v31/'+name,tofile='derived_v32/'+name)))
    write(PACKAGE/'docs/v32_exact_copy_provenance.json',{'parent_bundle_sha256':PARENT_SHA,
        'parent_files':parent['files'],'copied_files':copies,'changed_existing_files':sorted(MUTABLE),
        'all_model_archives_quality_and_numeric_sources_exact':True,
        'changes':'Canonical-output reference policy, explicit validity guards, caller ALRM cleanup mask, documentation/notebook/tests and exact official upstream license text.',
        'new_inference_or_fit':False,'cross_hardware_equality_established':False,'publication':False})
    files={}
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        require(not path.is_symlink(),'No linked package files')
        files[str(path.relative_to(PACKAGE))]={'sha256':digest(path),'bytes':path.stat().st_size}
    write(PACKAGE/'bundle_preparation.json',{'kind':'own_S20_disjoint_v32_reference_modes_preparation',
        'files':files,'parent_v31_bundle_sha256':PARENT_SHA,'new_weight_export':False,'new_training':False,
        'GPU_execution':False,'cross_hardware_equality':False,'publication':False})
    output=PACKAGE.parents[1]/'diagnostics/own_s_full_disjoint_v32_reference_mode'
    write(output/'prepared.json',{'status':'PREPARED_NOT_EXECUTED_OR_PUBLISHED','bundle_sha256':digest(PACKAGE/'bundle_preparation.json'),
        'parent_bundle_sha256':PARENT_SHA,'files':len(files),'exact_parent_copies':len(copies),
        'run_sha256':digest(PACKAGE/'run.py'),'worker_sha256':digest(PACKAGE/'worker.py'),
        'reference_policy_sha256':digest(PACKAGE/'reference_policy.py'),'license_sha256':license_info['license_sha256'],
        'GPU_launch_authorized':False})
    print(json.dumps(read(output/'prepared.json')),flush=True)


if __name__=='__main__':prepare()
