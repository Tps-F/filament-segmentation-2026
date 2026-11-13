"""CPU-only reference-mode and unchanged-computation contracts."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import run
from reference_policy import check_reference, summarize, validate_probability, validate_raw_rows

ROOT=Path(__file__).resolve().parent


def function_ast(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)


class ReferenceModeContracts(unittest.TestCase):
    def test_all_four_output_stages_strict_fail_report_only_records(self):
        for stage in ['detector/image','unet_b5_full/image','equal/image','candidate.csv']:
            with self.subTest(stage=stage):
                checks=[]
                with self.assertRaisesRegex(ValueError,'Reference mismatch'):
                    check_reference('strict','a'*64,'b'*64,stage,checks)
                self.assertFalse(checks[0]['exact'])
                checks=[];self.assertFalse(check_reference('report-only','a'*64,'b'*64,stage,checks))
                report=summarize('report-only',checks)
                self.assertEqual(report['mismatches'],1);self.assertFalse(report['all_reference_hashes_exact'])
                self.assertEqual(report['checks'][0]['actual_sha256'],'a'*64)

    def test_equal_reference_and_malformed_hashes(self):
        for mode in ['strict','report-only']:
            checks=[];self.assertTrue(check_reference(mode,'a'*64,'a'*64,'same',checks))
            self.assertTrue(summarize(mode,checks)['all_reference_hashes_exact'])
            with self.assertRaisesRegex(ValueError,'malformed'):
                check_reference(mode,'not-a-hash','a'*64,'same',[])
        with self.assertRaisesRegex(ValueError,'Reference mode'):
            check_reference('ignore','a'*64,'b'*64,'bad',[])
        self.assertFalse(summarize('report-only',[])['all_reference_hashes_exact'])

    def test_native_shape_dtype_and_finiteness_always_fatal(self):
        array=np.zeros((2,2048,2048),np.float16);validate_probability(array,'float16')
        with self.assertRaises(ValueError):validate_probability(array[:,:8,:8],'float16')
        with self.assertRaises(ValueError):validate_probability(array,'float32')
        for bad in [np.nan,np.inf,-.01,1.01]:
            array[0,0,0]=bad
            with self.assertRaises(ValueError):validate_probability(array,'float16')

    def test_raw_original_confidence_box_native_area_guards(self):
        from pycocotools import mask as mu
        mask=np.zeros((2048,2048),np.uint8);mask[1,1:7]=1
        rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
        row={'score':.5,'box':[1.,1.,7.,2.],'area':6,'rle':rle};validate_raw_rows([row])
        for key,bad in [('score',float('nan')),('box',[1.,2.,3.,float('inf')]),('area',7)]:
            changed=copy.deepcopy(row);changed[key]=bad
            with self.assertRaises(ValueError):validate_raw_rows([changed])
        with self.assertRaises(ValueError):validate_raw_rows([dict(row,score=.4),row])

    def test_numerical_functions_and_backend_configuration_unchanged(self):
        before=(ROOT/'docs/historical_v31_run.py').read_text();after=(ROOT/'run.py').read_text()
        for name in ['equal_probability','load_quality','selected_disjoint','selection_audit','child_environment']:
            self.assertEqual(ast.dump(function_ast(before,name)),ast.dump(function_ast(after,name)),name)
        old=function_ast(before,'detector_raw');new=function_ast(after,'detector_raw')
        class RemoveAddedFiniteGuard(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and node.value.args and isinstance(node.value.args[-1],ast.Constant) and node.value.args[-1].value=='Invalid detector tensor/confidence/box':return None
                return self.generic_visit(node)
        new=RemoveAddedFiniteGuard().visit(new);self.assertEqual(ast.dump(old),ast.dump(new))
        before=(ROOT/'docs/historical_v31_worker.py').read_text();after=(ROOT/'worker.py').read_text()
        for name in ['configure_environment','native_bf16','configure_torch_backend','backend','runtime']:
            self.assertEqual(ast.dump(function_ast(before,name)),ast.dump(function_ast(after,name)),name)
        calls=lambda text:[ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='predict_image']
        self.assertEqual(calls(before),calls(after))

    def test_all_numeric_sources_archives_inputs_and_quality_are_parent_exact(self):
        proof=json.loads((ROOT/'docs/v32_exact_copy_provenance.json').read_text())
        for name,row in proof['copied_files'].items():self.assertEqual(run.digest(ROOT/name),row['sha256'],name)
        self.assertTrue(all(n in proof['copied_files'] for n in proof['parent_files'] if n.startswith(('weights/','src/'))))
        self.assertEqual(run.digest(ROOT/'docs/canonical_reference.json'),proof['copied_files']['docs/canonical_reference.json']['sha256'])
        run.verify_bundle(require_bound=True)

    def test_actual_artifact_tamper_still_fatal_in_any_reference_mode(self):
        original=run.read
        def changed(path):
            if Path(path)==ROOT/'bundle_preparation.json':
                value=original(path);value['files']['reference_policy.py']['sha256']='0'*64;return value
            return original(path)
        with patch.object(run,'read',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Bundle file changed'):run.verify_bundle(require_bound=True)


if __name__=='__main__':unittest.main()
