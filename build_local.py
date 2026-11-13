"""Author-only CPU preparation/binding; reuse archives without model export."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[2]
PARENT = REPOSITORY/'outputs/release_preview/own_yolo_s_full_equal_quality_deterministic_v2'
PARENT_SHA = 'ec057ed02e0a03381c906b7510152e1da94eb9de12c00b5fb81117c81555d4ad'
PREP = REPOSITORY/'outputs/diagnostics/own_s_full_disjoint_v3_release_preparation'
CORRECTION = 'outputs/diagnostics/yolo_s_disjoint_compliance_v3'
CORRECTED = 'outputs/candidates/own_yolo_s_full_equal_quality_q20_deterministic_disjoint_v3'
OLD = 'outputs/candidates/own_yolo_s_full_equal_quality_q20_deterministic_v1'
FUTURE = tuple([f'{CORRECTION}/{n}' for n in ('plan.json','complete.json','root_review.json','remote_review.json')]
    + [f'{CORRECTED}/{n}' for n in ('candidate.csv','source_ids.json','per_image_audit.json','prediction_seal.json','report.json')])
MUTABLE = {'run.py','build_local.py','test_release.py','README.txt','THIRD_PARTY_NOTICES.txt','recipe.json',
    'docs/canonical_reference.json','notebooks/reproduce_own_s_full_deterministic_v2.ipynb'}


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def cpu_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Author operation must hide CUDA')
    import torch
    require(not torch.cuda.is_initialized(),'No CUDA allowed for author binding')


def source_parity():
    require(digest(PARENT/'bundle_preparation.json')==PARENT_SHA,'Immutable parent v2 changed')
    parent=read(PARENT/'bundle_preparation.json');copied={}
    for name,row in parent['files'].items():
        require(digest(PARENT/name)==row['sha256'],'Parent file changed: '+name)
        if name not in MUTABLE:
            require(digest(PACKAGE/name)==row['sha256'],'Copied archive/source bytes changed: '+name)
            copied[name]=row
    require('worker.py' in copied and all(n in copied for n in parent['files'] if n.startswith('weights/')),
            'All worker/archive/quality bytes must remain exact')
    return copied


def refresh_bundle():
    files={}
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file() or path.name=='bundle_preparation.json' or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        require(not path.is_symlink() and not {'assembled','generated'}&set(path.relative_to(PACKAGE).parts),'Generated/linked release file')
        files[str(path.relative_to(PACKAGE))]={'sha256':digest(path),'bytes':path.stat().st_size}
    write(PACKAGE/'bundle_preparation.json',{'kind':'own_S20_deterministic_semantic_disjoint_v3','files':files,
        'parent_v2_bundle_sha256':PARENT_SHA,'no_new_weight_export':True,'no_new_training':True,
        'full_GPU_reproduction_executed':False,'publication':False})
    return digest(PACKAGE/'bundle_preparation.json')


def prepare():
    cpu_guard();require(not (PACKAGE/'bundle_preparation.json').exists(),'Preparation already frozen')
    from run import DISJOINT_POLICY,DISJOINT_HELPER_SHA
    copied=source_parity();require(digest(PACKAGE/'src/disjoint_masks.py')==DISJOINT_HELPER_SHA,'Helper differs')
    recipe=read(PACKAGE/'recipe.json')
    require(recipe['status']=='UNBOUND_DISJOINT_FINAL_ARTIFACTS' and recipe['expected_candidate_sha256'] is None,'Preparation must be unbound')
    recipe['boundary']['disjoint_clipping']=False;recipe['final_export']=DISJOINT_POLICY
    recipe['future_artifact_sha256']=dict.fromkeys(FUTURE);write(PACKAGE/'recipe.json',recipe)
    write(PACKAGE/'docs/exact_copy_provenance.json',{'parent_bundle_sha256':PARENT_SHA,'copied_files':copied,
        'worker_unchanged':True,'all_weight_archives_and_quality_unchanged':True,'new_weight_exports':False,'new_training':False})
    fingerprint=refresh_bundle();PREP.mkdir(parents=True,exist_ok=True)
    write(PREP/'prepared.json',{'status':'UNBOUND_DISJOINT_PREPARATION_ONLY','bundle_sha256':fingerprint,
        'parent_bundle_sha256':PARENT_SHA,'copied_exact_files':len(copied),'future_hashes_null':len(FUTURE),
        'helper_sha256':DISJOINT_HELPER_SHA,'cuda_initialized':False,'export_training_TEST_GPU_or_publication':False})
    print(json.dumps(read(PREP/'prepared.json')),flush=True)


def validate_review(review,recipe,preparation_sha):
    require(review.get('reviewer')=='root' and review.get('approved_for_actual_export_binding') is True
            and review.get('preparation_sha256')==preparation_sha,'Exact root binding review required')
    require(recipe['status']=='UNBOUND_DISJOINT_FINAL_ARTIFACTS' and all(v is None for v in recipe['future_artifact_sha256'].values()),'Only first binding allowed')
    actual=review['actual_file_sha256']
    require(set(actual)==set(FUTURE) and all(isinstance(v,str) and len(v)==64 for v in actual.values()),'Actual future pins incomplete')
    require(review.get('independent_completed_audit_approved') is True,'Completed corrected candidate needs independent audit')
    independent=review['independent_completed_audit']
    require(isinstance(independent['path'],str) and len(independent['sha256'])==64,'Independent completed audit pin missing')


def bound_readme(text,candidate_sha):
    old='STATUS: Canonical corrected output is unbound until author review.'
    require(text.count(old)==1 and len(candidate_sha)==64,'Unbound README status is not exact')
    return text.replace(old,'STATUS: Actual corrected artifacts are BOUND. Canonical CSV '+candidate_sha+
                        '. Full180 GPU reproduction remains PENDING.')


def verify_actual(recipe,review):
    import numpy as np
    from pycocotools import mask as mu
    from src.disjoint_masks import confidence_disjoint_with_ids,json_rle
    from src.metrics import read_submission
    from run import DISJOINT_HELPER_SHA,DISJOINT_POLICY
    for name,h in review['actual_file_sha256'].items():require(digest(REPOSITORY/name)==h,'Reviewed future artifact changed: '+name)
    independent=review['independent_completed_audit'];require(digest(REPOSITORY/independent['path'])==independent['sha256'],'Independent audit changed')
    for name,h in recipe['upstream_v2_artifact_sha256'].items():require(digest(REPOSITORY/name)==h,'Verified upstream v2 source changed')
    require(digest(REPOSITORY/'src/disjoint_masks.py')==digest(PACKAGE/'src/disjoint_masks.py')==DISJOINT_HELPER_SHA,'Correction helper differs')
    complete=read(REPOSITORY/CORRECTION/'complete.json');seal=read(REPOSITORY/CORRECTED/'prediction_seal.json');report=read(REPOSITORY/CORRECTED/'report.json')
    require(complete['status']=='complete_fixed_disjoint_compliance' and complete['quality_fit'] is False
            and complete['cuda_initialized'] is False and complete['upload'] is False,'Unexpected correction completion')
    require(complete['plan_sha256']==digest(REPOSITORY/CORRECTION/'plan.json')
            and complete['test_prediction_seal_sha256']==digest(REPOSITORY/CORRECTED/'prediction_seal.json')
            and complete['test_report_sha256']==digest(REPOSITORY/CORRECTED/'report.json'),'Correction completion identities differ')
    candidate=REPOSITORY/CORRECTED/'candidate.csv';fingerprint=digest(candidate)
    require(complete['test_candidate']['sha256']==seal['candidate']['sha256']==report['candidate_sha256']==fingerprint,'Corrected CSV identities differ')
    require(seal['original_csv_sha256']==recipe['parent_v2']['candidate_sha256'] and seal['quality_predictions_reused_exactly']
            and seal['annotation_targets_used'] is False and seal['all_native_RLE_disjoint_subset_source_identity_utility_parity'],'Correction changed fixed selection')
    require(seal['totals']['output_excess_overlap_pixels']==0 and report['overlap']['pixel_disjoint'] is True,'Correction not disjoint')
    source_ids=read(REPOSITORY/CORRECTED/'source_ids.json');audits=read(REPOSITORY/CORRECTED/'per_image_audit.json')
    require(seal['source_ids_sha256']==digest(REPOSITORY/CORRECTED/'source_ids.json')
            and seal['per_image_audit_sha256']==digest(REPOSITORY/CORRECTED/'per_image_audit.json'),'Correction audits changed')
    original=read_submission(REPOSITORY/OLD/'candidate.csv');old_audits=read(REPOSITORY/OLD/'per_image_audit.json');corrected=read_submission(candidate)
    reference=read(PACKAGE/'docs/canonical_reference.json');ids=reference['ids']
    require(len(ids)==180 and set(source_ids)==set(audits)==set(ids) and set(corrected)<=set(ids) and set(original)<=set(ids),'Wrong image inventory')
    count=0
    for stem in ids:
        raw_path=REPOSITORY/'outputs/yolo26s_full/test_e020_1536/raw'/(stem+'.json')
        require(digest(raw_path)==reference['native_raw_sha256'][stem],'Original detector cache changed')
        raw=read(raw_path);old_ids=old_audits[stem]['source_ids'];old_rles=original.get(stem,[])
        require(len(old_ids)==len(old_rles) and len(set(old_ids))==len(old_ids),'Original selected identities differ')
        selected=[{'rle':json_rle(r),'score':float(np.float32(raw[i]['score'])),'source_index':i} for r,i in zip(old_rles,old_ids)]
        kept,changes=confidence_disjoint_with_ids(selected,5);final=[json_rle(r) for r in corrected.get(stem,[])]
        require(final==[v['rle'] for v in kept] and source_ids[stem]==[v['source_index'] for v in kept]
                and audits[stem]['changes']==changes,'Exact clipping replay differs: '+stem)
        require(sum(int(mu.area(r)) for r in final)==(int(mu.area(mu.merge(final))) if final else 0),'Corrected CSV overlaps')
        count+=len(final)
    require(count==seal['candidate']['filaments']==seal['totals']['output_instances'],'Final count differs')
    return {'candidate_sha256':fingerprint,'images':180,'instances':count,'all180_original_selection_clipping_replay_exact':True,
        'all180_final_masks_pixel_disjoint':True,'helper_sha256':DISJOINT_HELPER_SHA,'policy':DISJOINT_POLICY,
        'independent_completed_audit':independent,'model_exports':False,'GPU':False}


def bind(review_path):
    cpu_guard()
    from run import verify_bundle
    recipe=verify_bundle();initial=digest(PACKAGE/'bundle_preparation.json');review=read(review_path)
    validate_review(review,recipe,initial);source_parity();proof=verify_actual(recipe,review)
    require(not (PACKAGE/'docs/disjoint_unbound_preparation.json').exists(),'Binding already attempted')
    shutil.copy2(PACKAGE/'bundle_preparation.json',PACKAGE/'docs/disjoint_unbound_preparation.json')
    shutil.copy2(PACKAGE/'recipe.json',PACKAGE/'docs/disjoint_unbound_recipe.json')
    shutil.copy2(PACKAGE/'README.txt',PACKAGE/'docs/disjoint_unbound_README.txt')
    shutil.copy2(review_path,PACKAGE/'docs/disjoint_binding_review.json')
    for name in FUTURE:
        if name.endswith('/candidate.csv'):continue  # Hash/proof only; do not ship TEST prediction rows.
        prefix='correction_' if name.startswith(CORRECTION+'/') else 'candidate_'
        shutil.copy2(REPOSITORY/name,PACKAGE/'docs'/('disjoint_'+prefix+Path(name).name))
    shutil.copy2(REPOSITORY/review['independent_completed_audit']['path'],PACKAGE/'docs/disjoint_independent_completed_audit.json')
    write(PACKAGE/'docs/disjoint_binding_proof.json',proof)
    recipe['status']='BOUND_ACTUAL_FINAL_ARTIFACTS';recipe['future_artifact_sha256']=review['actual_file_sha256']
    recipe['expected_candidate_sha256']=proof['candidate_sha256'];recipe['expected_instances']=proof['instances']
    recipe['limitation']='Actual corrected CSV binding is verified; full180/cross-GPU reproduction remains untested. No new model or quality fit. Only the post-q20 fixed clipping rule changes masks; old overlapping validation scores do not establish post-clipping quality.'
    recipe['binding_proof']='docs/disjoint_binding_proof.json';write(PACKAGE/'recipe.json',recipe)
    (PACKAGE/'README.txt').write_text(bound_readme((PACKAGE/'README.txt').read_text(),proof['candidate_sha256']))
    reference=read(PACKAGE/'docs/canonical_reference.json');reference['candidate_sha256']=proof['candidate_sha256']
    reference['scope']='Original raw/deterministic semantic180 references plus independently audited disjoint-v3 CSV.';write(PACKAGE/'docs/canonical_reference.json',reference)
    fingerprint=refresh_bundle();require(digest(PARENT/'bundle_preparation.json')==PARENT_SHA,'Parent package changed')
    write(PREP/'bound.json',{'status':'BOUND_ACTUAL_CORRECTED_CANDIDATE_NO_GPU_REPRODUCTION','bundle_sha256':fingerprint,
        'unbound_preparation_sha256':initial,'binding_review_sha256':digest(review_path),'proof':proof,
        'full_GPU_reproduction':False,'publication':False})
    print(json.dumps(read(PREP/'bound.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--bind-review',type=Path);args=parser.parse_args()
    require(not(args.prepare and args.bind_review),'Choose one author operation')
    if args.prepare:prepare()
    elif args.bind_review:bind(args.bind_review)
    else:print(json.dumps({'status':'author_plan_only','future':FUTURE,'GPU':False}))
