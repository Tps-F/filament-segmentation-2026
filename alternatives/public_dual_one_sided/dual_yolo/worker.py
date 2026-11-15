"""One published fresh inference or independent CPU audit child."""
import argparse,os,time
from pathlib import Path
from .common import require,read,save,digest
from .lifecycle import admitted,identity,parent_live
from .runtime import verify_bundle,dependency_contract,loaded_origins,resource_guard,sha_valid
def execute(stage,path):
    inputs=admitted(path,stage);bundle=Path(inputs['bundle']);output=Path(inputs['output'])
    require(stage in ('infer','audit') and os.environ.get('CUDA_VISIBLE_DEVICES')==('0' if stage=='infer' else ''),'Explicit selected child role required')
    recipe=verify_bundle(bundle);require(inputs['recipe_sha256']==digest(bundle/'recipe.json'),'Published recipe changed')
    resource_guard(admission=True,gpu=stage=='infer');runtime=dependency_contract(bundle/'runtime_lock.json',(bundle,))
    inventory=read(bundle/'test_inputs.json')['images'];reference=read(bundle/'canonical_reference.json')
    for role,row in recipe['models'].items():require(digest(inputs['models'][role])==row['sha256'],'Exact model bytes changed')
    from .operations import infer,audit
    if stage=='infer':
        require(inputs['preceding_inference_complete_sha256'] is None,'Inference cannot reuse an existing pass')
        result=infer(inputs['models'],inventory,inputs['images'],output,lambda:parent_live(inputs),reference)
    else:
        require(sha_valid(inputs['preceding_inference_complete_sha256'])
          and digest(output/'inference_complete.json')==inputs['preceding_inference_complete_sha256'],'Published actual inference seal changed')
        result=audit(inventory,output,lambda:parent_live(inputs),reference)
    parent_live(inputs);resource_guard();runtime['module_origins']=loaded_origins((bundle,))
    save(output/(stage+'_receipt.json'),{'worker':identity(),'parent':inputs['parent'],'stage':stage,'recipe_sha256':digest(bundle/'recipe.json'),
      'deadline_epoch':inputs['deadline_epoch'],'completed_epoch':time.time(),'result':result,'runtime':runtime,'external_exit_status':None})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();execute(a.stage,a.inputs)

