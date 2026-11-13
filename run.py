"""Portable own S20/deterministic-semantic/q20 disjoint-v3.2 inference.

No annotation manifest, downloads, training, model fitting or upload calls.
The author must bind actual final artifacts before explicit GPU execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import signal
import subprocess
import sys
from reference_policy import POLICY, MODES, validate_mode, check_reference, summarize, validate_probability, validate_raw_rows

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parent
QUALITY_SHA = '05ef8e57def758488c825de77dce9900a1d93788607e2513902403650c1b44b1'
DISJOINT_HELPER_SHA = '22a9856fcb841479b879e761450055f830e90cf6dc947f4e8579ad55923e6e3b'
DISJOINT_POLICY = {'stage': 'after unchanged q20 selection',
    'order': 'original detector confidence descending; stable selected-order ties',
    'min_remaining_area': 5, 'confidence_floor': 0., 'extra_mask_nms': None,
    'dropped_remainder_occupies_pixels': False, 'added_pixels': False,
    'helper_sha256': DISJOINT_HELPER_SHA}
BACKEND = {'cublas_workspace_config': ':4096:8', 'cudnn_benchmark': False, 'cudnn_deterministic': True,
    'deterministic_algorithms': True, 'deterministic_warn_only': False, 'matmul_allow_tf32': False,
    'cudnn_allow_tf32': True, 'float32_matmul_precision': 'highest', 'native_bf16': True,
    'threads': 4, 'cv2_threads': 1, 'amp_dtype': 'bfloat16', 'channels_last': True, 'model_float_dtype': 'torch.float32'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_bundle(require_bound=False):
    receipt = read(ROOT/'bundle_preparation.json')
    for relative, row in receipt['files'].items():
        path = ROOT/relative
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == row['bytes']
                and digest(path) == row['sha256'], 'Bundle file changed: '+relative)
    recipe = read(ROOT/'recipe.json')
    require(recipe['reference_policy'] == POLICY, 'Reference policy changed')
    require(recipe['quality']['sha256'] == QUALITY_SHA and recipe['quality']['threshold'] == .20,
            'Frozen final quality recipe changed')
    require(recipe['semantic_backend_policy'] == BACKEND and recipe['process_policy']['fresh_worker_per_model']
            and recipe['process_policy']['detector_CUBLAS_WORKSPACE_CONFIG'] is None, 'Separate backend policies changed')
    require(recipe['final_export'] == DISJOINT_POLICY
            and digest(ROOT/'src/disjoint_masks.py') == DISJOINT_HELPER_SHA,
            'Fixed disjoint export changed')
    if require_bound:
        require(recipe['status'] == 'BOUND_ACTUAL_FINAL_ARTIFACTS' and recipe['detector']['archive']
                and recipe['detector']['source_checkpoint_sha256'] and recipe['expected_candidate_sha256']
                and all(recipe['future_artifact_sha256'].values()), 'Final S20/candidate binding is still missing')
    return recipe


def equal_probability(first, second):
    """Preserve independent member FP16 rounding before the float32 mean."""
    import numpy as np
    require(first.dtype == second.dtype == np.float16 and first.shape == second.shape,
            'Expected equal-shape stored FP16 member probabilities')
    for value in (first, second):
        require(np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1, 'Invalid probability')
    result = first.astype(np.float32); result *= .5
    other = second.astype(np.float32); other *= .5; result += other
    return np.clip(result, 0, 1).astype(np.float16)


def load_quality(recipe):
    import joblib
    from src.yolo_refined_quality import QUALITY_FEATURES
    from src.yolo_boundary import PARAMETERS
    from src.quality_pool import POOL_CONFIG
    path = ROOT/recipe['quality']['file']
    require(digest(path) == QUALITY_SHA, 'Frozen quality artifact changed')
    saved = joblib.load(path)
    require(saved['features'] == QUALITY_FEATURES and len(QUALITY_FEATURES) == 27
            and saved['geometry'] == PARAMETERS and saved['context_pool'] == POOL_CONFIG
            and saved['threshold'] == .20 and saved['raw_confidence_floor'] == .03
            and saved['native_inference_size'] == 1536 and saved['semantic_weights'] == [.5, .5]
            and saved['model'].n_features_in_ == 27, 'Quality feature/geometry contract differs')
    return saved


def selected_disjoint(entry, scores):
    """Only selected q20 masks enter fixed detector-confidence ownership."""
    import numpy as np
    from src.disjoint_masks import confidence_disjoint_with_ids
    require(scores.shape == (len(entry['rles']),) and np.isfinite(scores).all(), 'Bad quality alignment')
    require(len(entry['source_indices']) == len(entry['confidence']) == len(entry['rles']), 'Source alignment differs')
    selected = [{'rle':r, 'score':float(c), 'source_index':i}
                for r,i,c,q in zip(entry['rles'],entry['source_indices'],entry['confidence'],scores) if q >= .20]
    require(all(type(d['source_index']) is int for d in selected), 'Invalid original IDs')
    return selected, *confidence_disjoint_with_ids(selected, min_area=5)


def selection_audit(entry, scores, raw, exported):
    """Exact clipping replay plus source subset, R12 and zero-overlap proof."""
    import cv2
    import numpy as np
    from pycocotools import mask as mu
    from src.metrics import json_rle
    selected, kept, changes = selected_disjoint(entry, scores)
    require([json_rle(r) for r in exported] == [d['rle'] for d in kept], 'CSV changed fixed disjoint selection')
    ids = [d['source_index'] for d in kept]
    require(len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < len(raw) for i in ids), 'Invalid original IDs')
    original_selected = {d['source_index']:d for d in selected}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    occupied = np.zeros((2048,2048), bool)
    for detection in kept:
        item,index,confidence = detection['rle'],detection['source_index'],detection['score']
        require(np.float32(raw[index]['score']) == confidence, 'Original detector confidence changed')
        mask = mu.decode(item).astype(bool); source = mu.decode(raw[index]['rle'])
        require(mask.shape == (2048, 2048) and mask.sum() >= 5
                and np.array_equal(mask, mu.decode(mu.encode(np.asfortranarray(mask,dtype=np.uint8)))), 'Native RLE mismatch')
        selected_mask = mu.decode(original_selected[index]['rle']).astype(bool)
        require(not np.any(mask & ~selected_mask), 'Disjoint output added pixels beyond selected mask')
        require(not np.any(mask & occupied), 'Disjoint output contains overlapping pixels')
        occupied |= mask
        support = cv2.dilate(source, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        require(not np.any(mask & ~support), 'Refined mask escaped original R12 support')
    rles = [d['rle'] for d in kept]
    pair = mu.iou(rles, rles, [0]*len(rles)) if rles else np.zeros((0, 0))
    require(not np.triu(pair > 0, 1).any(), 'Final masks must have zero pixel overlap')
    area = int(np.asarray(mu.area(rles)).sum()) if rles else 0
    union = int(mu.area(mu.merge(rles))) if rles else 0
    require(area == union, 'Final native area sum differs from union')
    return {'source_indices': ids, 'input_instances':len(selected), 'instances': len(ids),
            'dropped_instances':len(selected)-len(kept), 'changes':changes,
            'overlapping_pairs': 0, 'excess_overlap_pixels': 0, 'pixel_disjoint': True,
            'selected_mask_subset_and_original_R12_verified':True}


def detector_raw(model, image_path):
    """Exact native extraction semantics from the unchanged src.infer_yolo."""
    import numpy as np
    from pycocotools import mask as mu
    result = model.predict(str(image_path), imgsz=1536, conf=.03, iou=.5, max_det=100,
                           retina_masks=True, device='0', verbose=False)[0]
    require(tuple(result.orig_shape) == (2048, 2048), 'Wrong native detector shape')
    detections = []
    if result.masks is not None:
        for mask, score, box in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
            require(mask.shape == (2048, 2048), 'Retina mask is not native')
            require(np.isfinite(mask).all() and np.isfinite(score) and 0 <= score <= 1
                    and box.shape == (4,) and np.isfinite(box).all(), 'Invalid detector tensor/confidence/box')
            binary = np.asfortranarray(mask > .5, dtype=np.uint8)
            if not binary.any():
                continue
            encoded = mu.encode(binary)
            detections.append({'score': float(score), 'box': box.tolist(), 'area': int(mu.area(encoded)),
                               'rle': {'size': [2048, 2048], 'counts': encoded['counts'].decode('ascii')}})
    detections.sort(key=lambda row: -row['score'])
    return detections


def child_environment(role, inherited):
    """Only environment crosses exec; parent torch/backend state cannot leak."""
    require(role in ('detector','unet_b5_full','unet_convnext_tiny_full'), 'Unknown worker')
    result = dict(inherited)
    for key in ('CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
                'PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','PYTHONPATH'):
        result.pop(key, None)
    if role != 'detector': result['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    result.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return result


def process_identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_token':fields[19],'session_id':int(fields[3])}


def stop_group(process):
    blocked=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
    try:
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:os.killpg(process.pid,sig)
            except ProcessLookupError:pass
            if sig==signal.SIGTERM:
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:pass
        process.wait(timeout=5)
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,blocked)


def run_worker(role, argv, output, environment, deadline):
    require(deadline-time.monotonic()>10,'Release wall budget exhausted')
    process=None
    with (output/(role+'.log')).open('x') as stream:
        try:
            blocked=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
            try:
                unmask='import os,signal,sys;signal.pthread_sigmask(signal.SIG_SETMASK,set());os.execvpe(sys.argv[1],sys.argv[1:],os.environ)'
                process=subprocess.Popen([sys.executable,'-c',unmask,*argv],cwd=ROOT,env=environment,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                (output/(role+'_started.json')).write_text(json.dumps({'command':argv,**process_identity(process.pid)},indent=2)+'\n')
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,blocked)
            require(process.wait(timeout=max(1,deadline-time.monotonic()-10))==0,role+' worker failed; preserve evidence, no automatic retry')
        finally:
            if process is not None:stop_group(process)
    (output/(role+'_exited.json')).write_text(json.dumps({'process_exited':True,'pid':process.pid},indent=2)+'\n')


def execute(test_dir, output, reference_mode='strict'):
    validate_mode(reference_mode)
    deadline=time.monotonic()+10800
    def expired(*_):raise TimeoutError('Absolute10800-second release budget exceeded')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(10800)
    try:return execute_impl(test_dir,output,deadline,reference_mode)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


def execute_impl(test_dir, output,deadline,reference_mode='strict'):
    validate_mode(reference_mode)
    recipe=verify_bundle(require_bound=True)
    require(not output.exists(),'Use a new output directory; no prediction reuse/retry')
    inherited=dict(os.environ)
    require(inherited.get('CUDA_VISIBLE_DEVICES')!='','GPU visibility must be provided for the explicit full run')
    # The parent is CPU-only. Each child gets the original GPU selection through an explicit environment.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import cv2
    import joblib
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from src.weight_archive import restore
    from src.inference import atomic_save_probabilities, write_submission
    from src.metrics import read_submission
    from src.ensemble import atomic_json, atomic_write
    from src.yolo_fixed_semantic_evaluator import generic_geometry_features
    require(not torch.cuda.is_initialized(),'CPU parent must not own a CUDA context')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    inventory=read(ROOT/'docs/test_inputs.json')['images'];reference=read(ROOT/'docs/canonical_reference.json')
    actual={p.name for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    require(len(inventory)==180 and actual=={r['file_name'] for r in inventory},'Expected all180 official filenames')
    for row in inventory:require(digest(test_dir/row['file_name'])==row['jpeg_sha256'],'Official input bytes changed')
    restored={}
    for member in recipe['semantic_members']+[recipe['detector']]:
        path=restore(ROOT/member['archive'],ROOT/'assembled')
        require(digest(path)==member['inference_checkpoint_sha256'],'Restored checkpoint hash differs')
        restored[member['name']]=str(path)
    output.mkdir(parents=True);(output/'images').mkdir()
    started=time.monotonic()
    for row in inventory:
        gray=cv2.imread(str(test_dir/row['file_name']),cv2.IMREAD_GRAYSCALE)
        require(gray is not None and gray.shape==(2048,2048) and gray.dtype==np.uint8,'Wrong native grayscale')
        path=output/'images'/(row['image_id']+'.npy');np.save(path,gray,allow_pickle=False)
        require(digest(path)==row['native_npy_sha256'],'Native feature input bytes changed')
    atomic_json(output/'execution_inputs.json',{'bundle_sha256':digest(ROOT/'bundle_preparation.json'),
        'recipe_sha256':digest(ROOT/'recipe.json'),'test_dir':str(test_dir),'output':str(output),'restored':restored,
        'reference_mode':reference_mode})
    # No torch state or model is shared: three exec-created interpreter processes.
    for role in ('detector','unet_b5_full','unet_convnext_tiny_full'):
        environment=child_environment(role,inherited)
        argv=[sys.executable,'-u',str(ROOT/'worker.py'),'--role',role,'--inputs',str(output/'execution_inputs.json')]
        # Defer controller ALRM across child publication and cleanup; worker wait uses the absolute deadline.
        old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
        try:run_worker(role,argv,output,environment,deadline)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
    require(not torch.cuda.is_initialized(),'Worker execution initialized parent CUDA')
    members=[];reference_reports={};equal_checks=[]
    for member in recipe['semantic_members']:
        receipt=read(output/(member['name']+'_receipt.json'))
        require(receipt['backend_policy']==BACKEND and receipt['reference_comparison']['mode']==reference_mode, 'Semantic worker receipt differs')
        require(set(receipt['probability_sha256'])=={r['image_id'] for r in inventory}, 'Incomplete actual member hashes')
        if reference_mode=='strict':require(receipt['all180_reference_probability_hashes_exact'], 'Strict semantic equality failed')
        reference_reports[member['name']]=receipt['reference_comparison']
        members.append({'prob_dir':str(output/member['name']),'probability_sha256':receipt['probability_sha256'],'weight':.5})
    detector_receipt=read(output/'detector_receipt.json')
    require(detector_receipt['reference_comparison']['mode']==reference_mode
            and set(detector_receipt['raw_sha256'])=={r['image_id'] for r in inventory}, 'Detector receipt/inventory differs')
    if reference_mode=='strict':require(detector_receipt['all180_reference_raw_hashes_exact'],'Strict detector equality failed')
    reference_reports['detector']=detector_receipt['reference_comparison']
    equal_dir=output/'equal';equal_dir.mkdir();entries=[]
    with threadpool_limits(limits=4):
        for row in inventory:
            require(time.monotonic()<deadline-10,'Release CPU stage exceeded wall budget')
            stem=row['image_id'];arrays=[]
            for member in members:
                actual_path=Path(member['prob_dir'])/(stem+'.npz')
                require(digest(actual_path)==member['probability_sha256'][stem], 'Actual member bytes changed after worker')
                with np.load(actual_path,allow_pickle=False) as z:array=z['probability']
                validate_probability(array,'float16');arrays.append(array)
            path=equal_dir/(stem+'.npz');atomic_save_probabilities(path,equal_probability(*arrays))
            check_reference(reference_mode,digest(path),reference['equal_probability_sha256'][stem],'equal/'+stem,equal_checks)
            raw_path=output/'raw'/(stem+'.json')
            require(digest(raw_path)==detector_receipt['raw_sha256'][stem], 'Actual raw bytes changed after worker')
            validate_raw_rows(read(raw_path))
            entry=generic_geometry_features((stem,output/'raw'/(stem+'.json'),path,output/'images'/(stem+'.npy'),digest(path),members))
            require(entry['features'].shape==(len(entry['rles']),27) and np.isfinite(entry['features']).all()
                    and np.array_equal(entry['features'][:,-1],entry['confidence']),'Feature/confidence mismatch')
            entries.append(entry)
        atomic_write(output/'geometry_features.joblib',lambda f:joblib.dump(entries,f))
        atomic_json(output/'geometry_sealed.json',{'sha256':digest(output/'geometry_features.joblib'),'annotations_used':False})
        quality=load_quality(recipe);features=np.concatenate([e['features'] for e in entries])
        scores=np.clip(quality['model'].predict(features),0,1) if len(features) else np.empty(0)
        require(np.isfinite(scores).all(),'Nonfinite quality prediction');predictions=[];offset=0
        for entry in entries:
            n=len(entry['rles']);local=np.clip(quality['model'].predict(entry['features']),0,1) if n else np.empty(0)
            require(np.array_equal(local,scores[offset:offset+n]),'Global/per-image quality parity differs')
            _,kept,_ = selected_disjoint(entry,local)
            predictions.append((entry['image_id'],[d['rle'] for d in kept]));offset+=n
    atomic_write(output/'quality_scores.npy',lambda f:np.save(f,scores))
    csv=output/'candidate.csv';export=write_submission(predictions,csv,[r['image_id'] for r in inventory]);native=read_submission(csv)
    audits={};offset=0
    for entry in entries:
        stem=entry['image_id'];n=len(entry['rles'])
        audits[stem]=selection_audit(entry,scores[offset:offset+n],read(output/'raw'/(stem+'.json')),native.get(stem,[]));offset+=n
    atomic_json(output/'native_audit.json',audits)
    atomic_json(output/'source_ids.json',{stem:audit['source_indices'] for stem,audit in audits.items()})
    require(all(a['pixel_disjoint'] and a['excess_overlap_pixels']==0 for a in audits.values()), 'Final zero-overlap audit failed')
    reference_reports['equal']=summarize(reference_mode,equal_checks)
    csv_checks=[]
    # Record final equality before enforcing strict mode, preserving the valid generated CSV and audit.
    check_reference('report-only',digest(csv),recipe['expected_candidate_sha256'],'candidate.csv',csv_checks)
    reference_reports['csv']=summarize(reference_mode,csv_checks)
    reference_summary={'mode':reference_mode,'stages':reference_reports,
        'all_reference_hashes_exact':all(r['all_reference_hashes_exact'] for r in reference_reports.values())}
    atomic_json(output/'reference_comparison.json',reference_summary)
    receipt={'status':'inference_finished','reference_mode':reference_mode,'reference_comparison_sha256':digest(output/'reference_comparison.json'),
        'all_reference_hashes_exact':reference_summary['all_reference_hashes_exact'],'csv_sha256':digest(csv),'expected_csv_sha256':recipe['expected_candidate_sha256'],
        'exact_csv_reproduction':digest(csv)==recipe['expected_candidate_sha256'],'export':export,'images':180,
        'separate_fresh_processes':True,'parent_cuda_initialized':torch.cuda.is_initialized(),'source_ID_R12_RLE_selection_verified':True,
        'final_export':DISJOINT_POLICY,'all180_final_masks_pixel_disjoint':True,
        'seconds':time.monotonic()-started,'quality_fitted':False,'submitted':False,'trained':False,
        'scope':'This actual run only. Report-only permits reference byte differences, never invalid output or changed recipe. Completion does not certify submitted-CSV equality on another GPU; inspect equality fields.'}
    atomic_json(output/'reproduction_receipt.json',receipt)
    if reference_mode=='strict':require(receipt['all_reference_hashes_exact'],'Strict reference equality failed; preserve evidence')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'generated')
    parser.add_argument('--reference-mode',choices=MODES,default='strict',help='strict stops on canonical GB10 byte differences; report-only records them without changing computation')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    def interrupted(signum,_):raise RuntimeError(f'Interrupted by signal{signum}')
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.execute:
            require(args.test_dir is not None,'Explicit --test-dir required');execute(args.test_dir.resolve(),args.output.resolve(),args.reference_mode)
        else:print(json.dumps({'status':'plan_only','reference_mode':args.reference_mode,'recipe':verify_bundle(),'gpu_started':False},indent=2))
    finally:signal.signal(signal.SIGTERM,previous)


