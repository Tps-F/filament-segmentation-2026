"""Portable CPU-only contracts; synthetic images, no TEST prediction."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+'.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
runner = module('run'); builder = module('build_local')


def unbound_fixture():
    """Test the same guards both before and after actual author binding."""
    actual=runner.verify_bundle()
    source=runner.read(ROOT/'docs/disjoint_unbound_recipe.json') if actual['status']=='BOUND_ACTUAL_FINAL_ARTIFACTS' else actual
    fixture=copy.deepcopy(source);fixture['reference_policy']=actual['reference_policy']
    assert fixture['status']=='UNBOUND_DISJOINT_FINAL_ARTIFACTS'
    return fixture


class ReleaseContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        import cv2
        cv2.setNumThreads(1)

    def test_unbound_execution_rejected_before_images_or_cuda(self):
        actual=runner.verify_bundle()  # Always verify the real current package, even when bound.
        self.assertIn(actual['status'],['UNBOUND_DISJOINT_FINAL_ARTIFACTS','BOUND_ACTUAL_FINAL_ARTIFACTS'])
        recipe = unbound_fixture()
        self.assertEqual(recipe['status'], 'UNBOUND_DISJOINT_FINAL_ARTIFACTS')
        self.assertEqual(recipe['detector']['source_checkpoint_sha256'], 'ad76892a01c5bdfb728959cf65bd6c53dbb3753f2e8c1d835594d051a8040cff')
        self.assertIsNone(recipe['expected_candidate_sha256'])
        self.assertTrue(all(v is None for v in recipe['future_artifact_sha256'].values()))
        actual_read=runner.read
        # Only the decoded recipe is a fixture; all real bundle file/hash checks still run.
        def fixture_read(path):
            return copy.deepcopy(recipe) if Path(path)==ROOT/'recipe.json' else actual_read(path)
        with patch.object(runner,'read',side_effect=fixture_read):
            with self.assertRaisesRegex(ValueError, 'binding is still missing'):
                runner.execute(Path('/no/test/data'), Path('/no/output'))
        self.assertFalse(torch.cuda.is_initialized())

    def test_binding_requires_exact_review_and_all_actual_future_pins(self):
        recipe=unbound_fixture()
        prior=ROOT/'docs/disjoint_unbound_preparation.json'
        sha=runner.digest(prior if prior.exists() else ROOT/'bundle_preparation.json')
        approved = {'reviewer':'root', 'approved_for_actual_export_binding':True, 'preparation_sha256':sha,
                    'actual_file_sha256':{p:'a'*64 for p in builder.FUTURE},
                    'independent_completed_audit_approved':True,
                    'independent_completed_audit':{'path':'review.json','sha256':'b'*64}}
        builder.validate_review(approved, recipe, sha)
        for field, bad in [('reviewer','other'), ('approved_for_actual_export_binding',False), ('preparation_sha256','changed')]:
            altered = copy.deepcopy(approved); altered[field] = bad
            with self.assertRaises(ValueError): builder.validate_review(altered, recipe, sha)
        altered = copy.deepcopy(approved); altered['actual_file_sha256'][builder.FUTURE[0]] = None
        with self.assertRaises(ValueError): builder.validate_review(altered, recipe, sha)

    def test_disjoint_confidence_ties_subset_and_dropped_occupancy(self):
        from pycocotools import mask as mu
        from src.metrics import json_rle
        masks=[]
        for points in [[(0,x) for x in range(6)],[(0,5),(1,1),(1,2),(1,3),(1,4)],[(1,x) for x in range(1,7)]]:
            mask=np.zeros((2048,2048),np.uint8)
            for y,x in points:mask[y,x]=1
            masks.append(json_rle(mu.encode(np.asfortranarray(mask))))
        entry={'rles':masks,'source_indices':[0,1,2],'confidence':np.array([.9,.8,.7],np.float32)}
        raw=[{'rle':r,'score':float(c)} for r,c in zip(masks,entry['confidence'])]
        selected,kept,changes=runner.selected_disjoint(entry,np.array([.2,.99,.2]))
        self.assertEqual([d['source_index'] for d in kept],[0,2]);self.assertFalse(changes[1]['retained'])
        self.assertEqual(kept[1]['rle'],masks[2])
        audit=runner.selection_audit(entry,np.array([.2,.99,.2]),raw,[d['rle'] for d in kept])
        self.assertTrue(audit['pixel_disjoint']);self.assertEqual(audit['excess_overlap_pixels'],0)
        self.assertEqual(audit['dropped_instances'],1)
        ties={'rles':[masks[0],masks[0]],'source_indices':[2,0],'confidence':np.array([.5,.5],np.float32)}
        _,kept,_=runner.selected_disjoint(ties,np.array([.9,.2]));self.assertEqual([d['source_index'] for d in kept],[2])

    def test_worker_and_archive_bytes_remain_parent_copies(self):
        proof=runner.read(ROOT/'docs/v32_exact_copy_provenance.json')
        self.assertTrue(proof['all_model_archives_quality_and_numeric_sources_exact'])
        for name,row in proof['copied_files'].items():self.assertEqual(runner.digest(ROOT/name),row['sha256'])
        self.assertEqual(runner.digest(ROOT/'src/disjoint_masks.py'),runner.DISJOINT_HELPER_SHA)

    def test_binding_documentation_distinguishes_binding_from_reproduction(self):
        text='STATUS: Canonical corrected output is unbound until author review.\nOther unchanged text.'
        result=builder.bound_readme(text,'a'*64)
        self.assertIn('Actual corrected artifacts are BOUND',result)
        self.assertIn('Full180 GPU reproduction remains PENDING',result)
        self.assertTrue(result.endswith('Other unchanged text.'))
        with self.assertRaises(ValueError):builder.bound_readme(result,'a'*64)
        notice=(ROOT/'THIRD_PARTY_NOTICES.txt').read_text()
        self.assertNotIn('Actual S20 export is pending',notice)
        self.assertIn('AGPL-3.0',notice)
        self.assertIn('exact final S20 EMA inference archive is included',notice)

    def test_member_rounding_before_mean_is_required(self):
        rng = np.random.default_rng(2026)
        a, b = [rng.random((2,64,64),dtype=np.float32) for _ in range(2)]
        expected = (.5*a.astype(np.float16).astype(np.float32)+.5*b.astype(np.float16).astype(np.float32)).astype(np.float16)
        self.assertTrue(np.array_equal(runner.equal_probability(a.astype(np.float16),b.astype(np.float16)), expected))
        self.assertTrue(np.any(expected != (.5*a+.5*b).astype(np.float16)))
        with self.assertRaises(ValueError): runner.equal_probability(a,b)

    def test_portable_pure_geometry_27features_and_low_confidence_quality_selection(self):
        from src.yolo_fixed_semantic_evaluator import generic_geometry_features
        from src.metrics import encode_mask, json_rle, read_submission
        from src.inference import atomic_save_probabilities, write_submission
        self.assertEqual(Path(__import__('src.yolo_fixed_semantic_evaluator',fromlist=['x']).__file__).resolve(), ROOT/'src/yolo_fixed_semantic_evaluator.py')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); stem = 'synthetic'; image = np.full((2048,2048),180,np.uint8)
            a = np.zeros_like(image); a[100:120,100:160] = 1
            b = np.zeros_like(image); b[110:130,140:200] = 1
            union = (a | b).astype(bool); image[union] = 40; np.save(p/'image.npy',image)
            raw = [{'score':score, 'box':[0.,0.,2048.,2048.], 'area':int(mask.sum()), 'rle':json_rle(encode_mask(mask))}
                   for mask,score in [(a,.05),(b,.8)]]
            (p/'raw.json').write_text(json.dumps(raw))
            probability = np.zeros((2,2048,2048),np.float16); probability[0,union] = .95
            sources = []
            for name in ['a','b']:
                directory = p/name; directory.mkdir(); path = directory/(stem+'.npz')
                atomic_save_probabilities(path,probability)
                sources.append({'prob_dir':str(directory),'weight':.5,'probability_sha256':{stem:runner.digest(path)}})
            equal = p/'equal.npz'; atomic_save_probabilities(equal,runner.equal_probability(probability,probability))
            previous = Path.cwd()
            try:
                os.chdir(p)
                entry = generic_geometry_features((stem,p/'raw.json',equal,p/'image.npy',runner.digest(equal),sources))
            finally:
                os.chdir(previous)
            self.assertEqual(entry['features'].shape,(2,27))
            self.assertTrue(np.array_equal(entry['features'][:,-1],entry['confidence']))
            saved = runner.load_quality(runner.verify_bundle()); scores = saved['model'].predict(entry['features'])
            self.assertTrue(np.isfinite(scores).all())
            fixed_scores = np.array([.3 if index == 0 else .1 for index in entry['source_indices']])
            masks = [r for r,s in zip(entry['rles'],fixed_scores) if s >= .2]
            csv = p/'synthetic.csv'; write_submission([(stem,masks)],csv,[stem]); exported = read_submission(csv)[stem]
            audit = runner.selection_audit(entry,fixed_scores,raw,exported)
            self.assertEqual(audit['source_indices'],[0])
            broken = dict(entry,source_indices=[-1 if i == 0 else i for i in entry['source_indices']])
            with self.assertRaisesRegex(ValueError,'Invalid original IDs'):
                runner.selection_audit(broken,fixed_scores,raw,exported)
        self.assertFalse(torch.cuda.is_initialized())

    def test_fresh_worker_environment_before_torch_and_no_parent_backend_leakage(self):
        worker=module('worker')
        with self.assertRaisesRegex(ValueError,'fresh interpreter'):worker.configure_environment('detector')
        code="import sys,json,os;sys.path.insert(0,sys.argv[1]);import worker;assert 'torch' not in sys.modules;worker.configure_environment(sys.argv[2]);import torch;worker.configure_torch_backend(sys.argv[2],torch);print(json.dumps({'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'benchmark':torch.backends.cudnn.benchmark,'deterministic':torch.are_deterministic_algorithms_enabled(),'cuda_initialized':torch.cuda.is_initialized(),'override':os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}))"
        original_benchmark=torch.backends.cudnn.benchmark;original_det=torch.are_deterministic_algorithms_enabled()
        try:
            torch.backends.cudnn.benchmark=True;torch.use_deterministic_algorithms(True)
            for role in ('detector','unet_b5_full','unet_convnext_tiny_full'):
                env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':'contaminated','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1','CUDA_VISIBLE_DEVICES':''}
                result=json.loads(subprocess.check_output([sys.executable,'-c',code,str(ROOT),role],env=env,text=True))
                self.assertEqual(result['cublas'],None if role=='detector' else ':4096:8')
                self.assertEqual(result['deterministic'],role!='detector');self.assertFalse(result['benchmark'] or result['cuda_initialized'])
                self.assertIsNone(result['override'])
        finally:torch.backends.cudnn.benchmark=original_benchmark;torch.use_deterministic_algorithms(original_det)

    def test_actual_native_constructor_environment_transition_cpu_only(self):
        from src.weight_archive import restore
        recipe=runner.verify_bundle(require_bound=True)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=restore(ROOT/recipe['detector']['archive'],Path(tmp))
            code="import sys,json,os;sys.path.insert(0,sys.argv[1]);import worker;worker.configure_environment('detector');import torch;worker.configure_torch_backend('detector',torch);from ultralytics import YOLO;before=os.environ.get('CUBLAS_WORKSPACE_CONFIG');model=YOLO(sys.argv[2]);print(json.dumps({'before':before,'after':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'deterministic':torch.are_deterministic_algorithms_enabled(),'cuda_initialized':torch.cuda.is_initialized(),'device':str(next(model.model.parameters()).device)}))"
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','CUBLAS_WORKSPACE_CONFIG':'inherited_value_to_clear'}
            actual=json.loads(subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(checkpoint)],env=env,text=True,timeout=60))
            self.assertEqual(actual,{'before':None,'after':':4096:8','benchmark':False,'cudnn_deterministic':False,'deterministic':False,'cuda_initialized':False,'device':'cpu'})
        self.assertFalse(torch.cuda.is_initialized())

    def test_hardware_guard_rejects_T4_emulation_without_precision_fallback(self):
        worker=module('worker')
        def fake(cap,native):
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,get_device_capability=lambda:cap,
                is_bf16_supported=lambda including_emulation: native if including_emulation is False else True))
        for cap,native in [((7,5),True),((8,0),False)]:
            with self.assertRaisesRegex(ValueError,'native BF16'):worker.native_bf16(fake(cap,native))
        worker.native_bf16(fake((12,1),True))

    def test_timeout_and_nonzero_exit_cleanup_grandchildren(self):
        def active(pid):
            try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
            except FileNotFoundError:return False
        for failure in (True,False):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);pidfile=out/'grandchild.pid'
                code="import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid));"+('sys.exit(7)' if failure else 'time.sleep(60)')
                with self.assertRaises((ValueError,subprocess.TimeoutExpired)):
                    runner.run_worker('contract',[sys.executable,'-c',code,str(pidfile)],out,dict(os.environ),time.monotonic()+(30 if failure else 10.1))
                self.assertTrue(pidfile.exists());pid=int(pidfile.read_text());deadline=time.monotonic()+3
                while active(pid) and time.monotonic()<deadline:time.sleep(.02)
                self.assertFalse(active(pid));self.assertFalse((out/'contract_exited.json').exists())

    def test_parent_SIGTERM_cleanup(self):
        def active(pid):
            try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
            except FileNotFoundError:return False
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);pidfile=out/'grandchild.pid'
            child="import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
            controller="import signal,sys,time,os;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from run import run_worker;\ndef stop(*a):raise RuntimeError('terminate')\nsignal.signal(signal.SIGTERM,stop);run_worker('contract',[sys.executable,'-c',sys.argv[3],sys.argv[4]],Path(sys.argv[2]),dict(os.environ),time.monotonic()+30)"
            parent=subprocess.Popen([sys.executable,'-c',controller,str(ROOT),tmp,child,str(pidfile)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+8
                while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(pidfile.exists());grandchild=int(pidfile.read_text());childpid=json.loads((out/'contract_started.json').read_text())['pid']
                parent.terminate();parent.wait(timeout=8);deadline=time.monotonic()+3
                while (active(childpid) or active(grandchild)) and time.monotonic()<deadline:time.sleep(.02)
                self.assertFalse(active(childpid));self.assertFalse(active(grandchild))
            finally:
                if parent.poll() is None:parent.kill();parent.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
