"""One fresh model worker. Configure environment BEFORE importing torch."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

from run import ROOT, BACKEND, child_environment, require, read, digest, verify_bundle, detector_raw
from reference_policy import validate_mode, check_reference, summarize, validate_probability, validate_raw_rows


def configure_environment(role):
    require('torch' not in sys.modules,'Worker must start in a fresh interpreter before torch import')
    desired=child_environment(role,os.environ)
    for key in ('CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
                'PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','PYTHONPATH'):
        os.environ.pop(key,None)
    os.environ.update(desired)
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==(None if role=='detector' else ':4096:8'),'Wrong per-worker CUBLAS setting')


def native_bf16(torch):
    require(torch.cuda.is_available() and torch.cuda.get_device_capability()[0]>=8
            and torch.cuda.is_bf16_supported(including_emulation=False),
            'This recipe requires a native BF16 CUDA GPU; T4/emulation and precision fallback are unsupported')


def configure_torch_backend(role,torch):
    if role=='detector':
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None and not torch.backends.cudnn.benchmark
            and not torch.backends.cudnn.deterministic and not torch.are_deterministic_algorithms_enabled()
            and not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
            and torch.get_float32_matmul_precision()=='highest','YOLO fresh defaults differ from canonical original process')
    else:
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        torch.set_float32_matmul_precision('highest')


def backend(torch,cv2,model):
    return {'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),'native_bf16':torch.cuda.is_bf16_supported(including_emulation=False),
        'threads':torch.get_num_threads(),'cv2_threads':cv2.getNumThreads(),'amp_dtype':'bfloat16',
        'channels_last':all(p.is_contiguous(memory_format=torch.channels_last) for p in model.parameters() if p.ndim==4),
        'model_float_dtype':str(next(model.parameters()).dtype)}


def runtime(torch):
    import importlib.metadata
    return {'python':sys.version,'interpreter':sys.executable,'pid':os.getpid(),'torch':torch.__version__,
        'torch_git':torch.version.git_version,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
        'module_paths':{'torch':torch.__file__},
        'versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','opencv-python','segmentation-models-pytorch','timm','ultralytics')},
        'environment':{k:os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}}


def worker(role,inputs_path):
    configure_environment(role)
    recipe=verify_bundle(require_bound=True);inputs=read(inputs_path)
    reference_mode=validate_mode(inputs['reference_mode']);reference_checks=[]
    require(inputs['bundle_sha256']==digest(ROOT/'bundle_preparation.json') and inputs['recipe_sha256']==digest(ROOT/'recipe.json'),'Parent bundle identity changed')
    import cv2
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(),'Worker unexpectedly initialized CUDA before backend configuration')
    cv2.setNumThreads(1);torch.set_num_threads(4)
    # Assert fresh YOLO defaults; configure semantic determinism only in its own process.
    configure_torch_backend(role,torch)
    native_bf16(torch)
    output=Path(inputs['output']);test_dir=Path(inputs['test_dir'])
    inventory=read(ROOT/'docs/test_inputs.json')['images'];reference=read(ROOT/'docs/canonical_reference.json')
    started=time.monotonic()
    if role=='detector':
        from ultralytics import YOLO
        from src.prepare_yolo import atomic_json
        member=recipe['detector'];checkpoint=Path(inputs['restored'][member['name']])
        require(digest(checkpoint)==member['inference_checkpoint_sha256'],'Detector checkpoint changed')
        model=YOLO(str(checkpoint));require(model.task=='segment' and not model.model.end2end,'Wrong native detector architecture')
        raw=output/'raw';require(not raw.exists(),'No raw-cache reuse');raw.mkdir();hashes={}
        for index,row in enumerate(inventory,1):
            image=test_dir/row['file_name'];require(digest(image)==row['jpeg_sha256'],'Official JPEG changed')
            path=raw/(row['image_id']+'.json');detections=detector_raw(model,image);validate_raw_rows(detections)
            atomic_json(path,detections);hashes[row['image_id']]=digest(path)
            check_reference(reference_mode,hashes[row['image_id']],reference['native_raw_sha256'][row['image_id']],role+'/'+row['image_id'],reference_checks)
            print(json.dumps({'stage':role,'images':index,'total':180}),flush=True)
        policy={'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
        require(policy=={'CUBLAS_WORKSPACE_CONFIG':':4096:8','cudnn_benchmark':False,'cudnn_deterministic':False,'deterministic_algorithms':False},'YOLO backend changed during inference')
        receipt={'role':role,'runtime':runtime(torch),'backend_policy':policy,'raw_sha256':hashes,
            'backend_entry_policy':{'CUBLAS_WORKSPACE_CONFIG':None,'cudnn_benchmark':False,'cudnn_deterministic':False,'deterministic_algorithms':False},
            'native_constructor_transition':'Ultralytics Model.__init__ sets CUBLAS_WORKSPACE_CONFIG=:4096:8; no backend setter added by this worker',
            'all180_reference_raw_hashes_exact':all(r['exact'] for r in reference_checks),
            'reference_comparison':summarize(reference_mode,reference_checks),'seconds':time.monotonic()-started}
    else:
        from src.model import load_checkpoint
        from src.inference import predict_image,atomic_save_probabilities
        from src.ensemble import atomic_json
        member=next(m for m in recipe['semantic_members'] if m['name']==role);checkpoint=Path(inputs['restored'][role])
        require(digest(checkpoint)==member['inference_checkpoint_sha256'],'Semantic checkpoint changed')
        model,saved=load_checkpoint(checkpoint,device='cuda',use_ema=True)
        require(saved['config']['fold']==-1 and saved['config'].get('input_preprocessing') is None,'Wrong semantic source')
        model.eval().to(memory_format=torch.channels_last);require(backend(torch,cv2,model)==BACKEND,'Actual deterministic backend differs')
        directory=output/role;require(not directory.exists(),'No semantic cache reuse');directory.mkdir();hashes={}
        for index,row in enumerate(inventory,1):
            path=output/'images'/(row['image_id']+'.npy');require(digest(path)==row['native_npy_sha256'],'Native NPY changed')
            probability=predict_image(model,np.load(path,allow_pickle=False),tile_size=768,stride=512,batch_size=4,tta=4,device='cuda',amp=True,scale=1.)
            validate_probability(probability,'float32')
            destination=directory/(row['image_id']+'.npz');atomic_save_probabilities(destination,probability);hashes[row['image_id']]=digest(destination)
            check_reference(reference_mode,hashes[row['image_id']],reference['semantic_probability_sha256'][role][row['image_id']],role+'/'+row['image_id'],reference_checks)
            print(json.dumps({'stage':role,'images':index,'total':180}),flush=True)
        require(backend(torch,cv2,model)==BACKEND,'Semantic backend changed during inference')
        receipt={'role':role,'runtime':runtime(torch),'backend_policy':BACKEND,'probability_sha256':hashes,
            'all180_reference_probability_hashes_exact':all(r['exact'] for r in reference_checks),
            'reference_comparison':summarize(reference_mode,reference_checks),'seconds':time.monotonic()-started}
    atomic_json(output/(role+'_receipt.json'),receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--role',required=True,choices=['detector','unet_b5_full','unet_convnext_tiny_full'])
    parser.add_argument('--inputs',required=True,type=Path);args=parser.parse_args();worker(args.role,args.inputs)
