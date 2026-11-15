"""One literal dual-model pass and independent saved-RLE export verification."""
import csv,os,time
from pathlib import Path
from .common import require,read,save,digest
from .runtime import ROLES,configure,assert_backend,resource_guard

def compare_final_output(actual,reference):
    require(actual['images']==reference['images']==180 and actual['instances']==reference['instances']
      and actual['prediction_sha256']['candidate.csv']==reference['prediction_sha256']['candidate.csv'],
      'Fresh final native rows/masks/gapped IDs/CSV differ')
    for key in ('backend','runtime','resolved_arguments'):require(actual[key]==reference[key],'Fresh declared runtime/backend differs:'+key)

def encode(mask):
    import numpy as np
    from pycocotools import mask as mu
    value=mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {'size':list(value['size']),'counts':value['counts'].decode('ascii')}

def extract(result):
    import numpy as np
    if result.masks is None:return None,None,{'none':True}
    masks=result.masks.data.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
    require(masks.dtype==np.uint8 and masks.ndim==3 and masks.shape[1:]==(2048,2048) and 0<=len(masks)<=100
      and scores.dtype==np.float32 and scores.shape==(len(masks),) and np.isfinite(scores).all()
      and np.all((scores>=.5)&(scores<=1)) and np.all((masks==0)|(masks==1)),'Literal masks/FP32 confidence contract differs; no cast rescue')
    return masks,scores,{'none':False,'shape':list(masks.shape),'mask_dtype':'uint8','score_dtype':'float32',
      'scores':scores.tolist(),'masks':[encode(m) for m in masks]}

def model_arguments(model):
    args=model.predictor.args
    require(args.imgsz==2048 and args.conf==.5 and args.iou==0 and args.max_det==100 and args.retina_masks is False
      and args.augment is False and args.rect is True and args.batch==1 and not args.tile and args.quantize is None,
      'Pinned local literal prediction defaults differ')
    require(model.predictor.model.device.type=='cuda' and not model.predictor.model.fp16,'Literal FP32 CUDA prediction required')
    return {k:getattr(args,k) for k in ('imgsz','conf','iou','max_det','retina_masks','augment','rect','batch','tile','quantize')}

def infer(models,inventory,images,output,check=lambda:None,reference=None):
    import cv2,numpy as np,torch,ultralytics
    from ultralytics import YOLO
    from .geometry import export_instances
    require(ultralytics.__version__=='8.4.211' and not torch.cuda.is_initialized(),'Fresh pinned local Ultralytics runtime required')
    require(tuple(models)==ROLES and len(inventory)==180 and [r['image_id'] for r in inventory]==sorted({r['image_id'] for r in inventory}),'Exact two-model/180 inventory required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rawdir=output/'raw';rawdir.mkdir();configure(torch,cv2)
    # Both constructors are retained before any image, matching the public source.
    instances={role:YOLO(str(models[role])) for role in ROLES};initial_backend=assert_backend(torch)
    require(initial_backend['CUBLAS_WORKSPACE_CONFIG']==':4096:8','Pinned Ultralytics constructor backend differs')
    rows=[];source_ids={};traces={};raw_hashes={};resolved={};started=time.time()
    for item in inventory:
        check();resource_guard(gpu=True,torch=torch);stem=item['image_id'];path=Path(images)/item['file_name']
        require(path.is_file() and not path.is_symlink() and digest(path)==item['jpeg_sha256'],'Official input bytes changed')
        result26=instances['yolo26x'].predict(str(path),imgsz=2048,conf=.5,iou=0.,max_det=100,verbose=False)[0]
        result8=instances['yolov8x'].predict(str(path),imgsz=2048,conf=.5,iou=0.,max_det=100,verbose=False)[0]
        for role in ROLES:resolved[role]=model_arguments(instances[role])
        masks26,scores26,raw26=extract(result26);masks8,scores8,raw8=extract(result8)
        raw={'image_id':stem,'jpeg_sha256':item['jpeg_sha256'],'models':{'yolo26x':raw26,'yolov8x':raw8}}
        save(rawdir/(stem+'.json'),raw);raw_hashes[stem]=digest(rawdir/(stem+'.json'))
        selected,trace=export_instances(masks26,scores26,masks8,scores8);source_ids[stem]=[];traces[stem]=trace
        for instance in selected:
            rle=encode(instance['mask']);rank=instance['rank']
            rows.append({'filament_id':f'{stem}_{rank}','segmentation_rle':rle['counts']})
            source_ids[stem].append({k:instance[k] for k in ('rank','source_index','score')})
        require(assert_backend(torch)==initial_backend,'Backend changed during literal pass')
        torch.cuda.synchronize();del result26,result8,masks26,masks8,scores26,scores8,raw26,raw8,raw,selected
        print(len(raw_hashes),flush=True)
    with (output/'candidate.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['filament_id','segmentation_rle']);writer.writeheader();writer.writerows(rows);stream.flush();os.fsync(stream.fileno())
    save(output/'source_ids.json',source_ids);save(output/'traces.json',traces)
    hashes={name:digest(output/name) for name in ('candidate.csv','source_ids.json','traces.json')}
    runtime={'torch_version':torch.__version__,'torch_git':torch.version.git_version,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
      'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'ultralytics':ultralytics.__version__}
    result={'images':180,'instances':len(rows),'raw_sha256':raw_hashes,'prediction_sha256':hashes,'backend':initial_backend,'runtime':runtime,'resolved_arguments':resolved,
      'candidate_count':1,'TRAIN_accessed':False,'fits':0,'old_prediction_cache_used':False}
    if reference is not None:compare_final_output(result,reference)
    save(output/'inference_complete.json',result);resource_guard(gpu=True,torch=torch);return result

def decode_record(record):
    import numpy as np
    from pycocotools import mask as mu
    if record['none']:
        require(set(record)=={'none'},'None record contains predictions');return None,None,[]
    require(set(record)=={'none','shape','mask_dtype','score_dtype','scores','masks'} and record['mask_dtype']=='uint8' and record['score_dtype']=='float32',
      'Raw serialized type contract differs')
    rles=record['masks'];scores=np.asarray(record['scores'],dtype=np.float32)
    require(0<=len(rles)<=100 and record['shape']==[len(rles),2048,2048] and scores.shape==(len(rles),)
      and np.isfinite(scores).all() and np.all((scores>=.5)&(scores<=1)),'Raw serialized values differ')
    masks=np.asarray([mu.decode(r) for r in rles],dtype=np.uint8) if rles else np.empty((0,2048,2048),dtype=np.uint8)
    require(masks.shape==(len(rles),2048,2048) and np.all((masks==0)|(masks==1)),'Raw binary mask shape differs')
    return masks,scores,rles

def independent_image(raw):
    """NMS via RLE integer intersections; ownership via first-owner label map."""
    import cv2,numpy as np
    from pycocotools import mask as mu
    first=decode_record(raw['models']['yolo26x']);second=decode_record(raw['models']['yolov8x'])
    if first[0] is None and second[0] is None:return []
    if first[0] is None:first=(np.empty((0,2048,2048),np.uint8),np.empty(0,np.float32),[])
    if second[0] is None:second=(np.empty((0,2048,2048),np.uint8),np.empty(0,np.float32),[])
    masks=np.concatenate((first[0],second[0]),axis=0);scores=np.concatenate((first[1],second[1]));rles=first[2]+second[2]
    areas=[int(mu.area(r)) for r in rles];pending=[int(i) for i in np.argsort(scores)[::-1]];kept=[]
    while pending:
        current=pending.pop(0);kept.append(current);survive=[]
        for other in pending:
            intersection=int(mu.area(mu.merge([rles[current],rles[other]],intersect=True)));union=areas[current]+areas[other]-intersection
            if 2*intersection<=union:survive.append(other)
        pending=survive
    ownership=[kept[int(i)] for i in np.argsort(scores[kept])[::-1]]
    owner=np.full(masks.shape[1:],-1,dtype=np.int16)
    for rank,index in enumerate(ownership):owner[(owner==-1)&(masks[index]!=0)]=rank
    expected=[]
    for rank,index in enumerate(ownership):
        mask=cv2.resize((owner==rank).astype(np.uint8),(2048,2048),interpolation=cv2.INTER_NEAREST)
        if int(mask.sum())<5:continue
        expected.append({'rank':rank+1,'source_index':index,'score':float(scores[index]),'rle':encode(mask)})
    return expected

def audit(inventory,output,check=lambda:None,reference=None):
    from .submit_validation import validate_payload
    output=Path(output);complete=read(output/'inference_complete.json');ids=[r['image_id'] for r in inventory]
    require(len(ids)==len(set(ids))==180 and complete['images']==180 and set(complete['raw_sha256'])==set(ids),'Complete180 audit inventory required')
    for name,h in complete['prediction_sha256'].items():require(digest(output/name)==h,'Sealed prediction bytes changed')
    sources=read(output/'source_ids.json');require(set(sources)==set(ids),'Full180 source IDs required');expected_rows=[];count=0;empty=[]
    for item in inventory:
        check();resource_guard();stem=item['image_id'];path=output/'raw'/(stem+'.json')
        require(digest(path)==complete['raw_sha256'][stem],'Sealed raw mask/score bytes changed')
        raw=read(path);require(raw['image_id']==stem and raw['jpeg_sha256']==item['jpeg_sha256'] and tuple(raw['models'])==ROLES,'Raw model/input identity differs')
        selected=independent_image(raw)
        require(sources[stem]==[{k:x[k] for k in ('rank','source_index','score')} for x in selected],'Independent source/order ownership differs')
        if not selected:empty.append(stem)
        for x in selected:expected_rows.append({'filament_id':f"{stem}_{x['rank']}",'segmentation_rle':x['rle']['counts']})
        count+=len(selected)
    with (output/'candidate.csv').open(newline='') as stream:
        reader=csv.DictReader(stream);require(reader.fieldnames==['filament_id','segmentation_rle'] and list(reader)==expected_rows,'Independent native rows/gapped IDs differ')
    _,strict=validate_payload((output/'candidate.csv').read_bytes(),set(ids))
    require(count==complete['instances'],'Native instance count differs')
    if reference is not None:compare_final_output(complete,reference)
    result={'status':'PASS_independent_saved180_native_pixel_disjoint','images':180,'instances':count,'empty_images':empty,
      'prediction_sha256':complete['prediction_sha256'],'raw_sha256':complete['raw_sha256'],'strict_payload':strict,
      'estimator_loaded_or_called':False,'TRAIN_accessed':False,'source_RLE_integer_NMS':True,'first_owner_label_map':True,'submission':False}
    save(output/'audit_complete.json',result);return result
