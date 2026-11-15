"""Pinned local runtime, single candidate and exact public bundle contracts."""
import importlib.metadata,math,os,resource,subprocess,sys,time
from pathlib import Path
from .common import require,read,digest
SCHEMA='public_dual_one_sided_local841211_v1'
ROLES=('yolo26x','yolov8x')
MODELS={'yolo26x':{'ref':'hdjojo/yolo26x-seg-solar-filament/pytorch/default/1','filename':'yolo26x-seg-solar-filament-2048-1109.pt','bytes':142289285},
        'yolov8x':{'ref':'hdjojo/yolov8x-seg-solar-filament/pytorch/default/1','filename':'yolov8x-seg-solar-filament-2048-1110.pt','bytes':144463332}}
CAPS={'preflight':600,'canonical':1800,'audit':600,'assembly':600,'fresh':2700,'finish':900}
FRESH={'setup':300,'dual_work':1800,'audit':450,'cleanup':150}
COMPETITION_DEADLINE=1794722400.
STRICT_BACKEND={'CUBLAS_WORKSPACE_CONFIG':':4096:8','cudnn_benchmark':False,'cudnn_deterministic':True,'deterministic_algorithms':True,
  'deterministic_warn_only':False,'matmul_allow_tf32':False,'cudnn_allow_tf32':True,'float32_matmul_precision':'highest',
  'predict_half':False,'predict_augment':False,'torch_threads':4,'cv2_threads':1,'CUDA_VISIBLE_DEVICES':'0'}
NUMERICAL_ROOTS=('torch','torchvision','ultralytics','numpy','cv2','scipy','sklearn','joblib','pycocotools')

def sha_valid(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)

def production_clock(start,end,now,admission=False):
    require(all(type(x) in (int,float) and math.isfinite(x) for x in (start,end,now)) and start<=now<end==start+7200
      and end+3600<=COMPETITION_DEADLINE,'Fixed7200 and submission3600 do not fit')
    if admission:require(now+7200+3600<=COMPETITION_DEADLINE and now<start+60,'Prospective full7200 admission expired')
    return end

def environment(gpu=False,config=None,parent=None):
    env=dict(os.environ if parent is None else parent)
    require(not any(k in env for k in ('PYTHONPATH','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')),
      'Unreviewed import/backend/allocator override')
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',
      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',CUDA_VISIBLE_DEVICES='0' if gpu else '')
    if config is not None:env['YOLO_CONFIG_DIR']=str(config)
    return env

def resource_guard(admission=False,gpu=False,torch=None):
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(available>=(48 if admission else 32)*2**30 and peak<=8*2**30,'Public dual YOLO host memory bound exceeded')
    if gpu:
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).split()
        require(all(x.isdigit() for x in rows) and not ({int(x) for x in rows}-{os.getpid()}),'Exclusive GPU required')
    reserved=0 if torch is None else torch.cuda.max_memory_reserved()
    require(reserved<=56*2**30,'Public dual YOLO GPU memory bound exceeded')
    return {'available_bytes':available,'peak_RSS_bytes':peak,'GPU_reserved_peak_bytes':reserved}

def local(root,name):
    root=Path(root).resolve();part=Path(name)
    require(not part.is_absolute() and part.parts and '..' not in part.parts,'Relative bundle path required')
    target=root/part
    require(target.is_file() and not any(p.is_symlink() for p in [target,*target.parents] if p!=root and root in p.parents)
      and target.resolve().is_relative_to(root),'Missing, symlinked or escaping bundle file')
    return target

def verify_bundle(bundle):
    bundle=Path(bundle).resolve();manifest=read(bundle/'bundle_manifest.json')
    require(manifest['schema']==SCHEMA and manifest['public_upload'] is False,'Wrong public method package')
    actual={str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file() and '.git' not in p.relative_to(bundle).parts}
    require(actual==set(manifest['files'])|{'bundle_manifest.json'},'Bundle whitelist differs')
    for name,row in manifest['files'].items():
        path=local(bundle,name);require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Bundle bytes changed:'+name)
    recipe=read(local(bundle,'recipe.json'));require(recipe['schema']==SCHEMA and tuple(recipe['models'])==ROLES and recipe['candidate_count']==1
      and recipe['declared_backend']==STRICT_BACKEND and recipe['backend_proof_addendum_sha256']=='366dfb4c16f617b2168e71489cb1950157d1bf294de6c2822ee7996ad4b8de57',
      'Candidate/backend identity differs')
    for role,identity in MODELS.items():
        model=recipe['models'][role]
        require(all(model[k]==v for k,v in identity.items()) and sha_valid(model['sha256']),'Exact official model version differs')
    return recipe

def backend(torch):
    import cv2
    return {'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
      'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
      'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
      'float32_matmul_precision':torch.get_float32_matmul_precision(),'torch_threads':torch.get_num_threads(),'cv2_threads':cv2.getNumThreads(),
      'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}

def configure(torch,cv2):
    require(not torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0',
      'Prospective strict backend must be set before CUDA/model loading')
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest');return assert_backend(torch)

def assert_backend(torch):
    actual=backend(torch);require(actual=={k:v for k,v in STRICT_BACKEND.items() if k not in ('predict_half','predict_augment')},'Declared strict local backend changed')
    return actual


def dependency_contract(lock_path,code_roots=()):
    lock=read(lock_path);prefix=Path(sys.prefix).resolve()
    require(sys.prefix!=sys.base_prefix and not os.environ.get('PYTHONPATH') and os.environ.get('PYTHONNOUSERSITE')=='1','Explicit isolated runtime required')
    require(list(sys.version_info[:3])==lock['python_version'] and digest(Path(sys.executable).resolve())==lock['python_binary_sha256'],'Exact Python runtime differs')
    actual={}
    for name,row in lock['distributions'].items():
        dist=importlib.metadata.distribution(name);info=Path(dist._path).resolve()
        require(info.is_relative_to(prefix) and dist.version==row['version'],'Distribution origin/version differs:'+name)
        for basename,h in row['metadata_sha256'].items():require(digest(info/basename)==h,'Distribution metadata differs:'+name+'/'+basename)
        actual[name]=dist.version
    for relative,h in lock['files_sha256'].items():
        path=prefix/relative;require(path.resolve().is_relative_to(prefix) and path.is_file() and digest(path)==h,'Runtime source/library bytes differ:'+relative)
    origins=loaded_origins(code_roots)
    return {'versions':actual,'lock_sha256':digest(lock_path),'python_binary_sha256':lock['python_binary_sha256'],'module_origins':origins,
      'qualification':lock['qualification'],'fresh_installation_proved':False}

BOOTSTRAP={'module':'sitecustomize','reported_path':'/usr/lib/python3.12/sitecustomize.py','resolved_path':'/etc/python3.12/sitecustomize.py','sha256':'43d81125d92376b1a69d53a71126a041cc9a18d8080e92dea0a2ae23be138b1e','bytes':155}

def trusted_bootstrap(name,file):
    if name!=BOOTSTRAP['module']:return False
    reported=Path(file);resolved=reported.resolve()
    require(str(reported)==BOOTSTRAP['reported_path'] and str(resolved)==BOOTSTRAP['resolved_path']
      and resolved.stat().st_size==BOOTSTRAP['bytes'] and digest(resolved)==BOOTSTRAP['sha256'],'Exact distribution bootstrap changed')
    return True

def loaded_origins(code_roots=()):
    prefix=Path(sys.prefix).resolve();standard=Path(sys.base_prefix).resolve();allowed=[Path(p).resolve() for p in code_roots];origins={}
    for name,module in tuple(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if trusted_bootstrap(name,file):origins[name]=str(path);continue
        if 'site-packages' in path.parts or 'dist-packages' in path.parts:
            require(path.is_relative_to(prefix),'External numerical dependency:'+name)
        else:require(path.is_relative_to(standard) or path.is_relative_to(prefix) or any(path.is_relative_to(p) for p in allowed),'Unbound code origin:'+name)
        if name.split('.')[0] in NUMERICAL_ROOTS or name.startswith('dual_yolo'):origins[name]=str(path)
    return origins
