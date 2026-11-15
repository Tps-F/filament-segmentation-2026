"""Fresh local clone, exact model bytes and all180 outputs, one fixed attempt."""
import os,time
from pathlib import Path
from .common import require,read,save,digest
from .runtime import verify_bundle,dependency_contract,resource_guard,environment,ROLES
from .lifecycle import arm,identity,run_child

def sealed_worker_result(output,stage,receipt):
    require(stage in ('infer','audit'),'Unknown fresh stage artifact')
    path=Path(output)/('inference_complete.json' if stage=='infer' else 'audit_complete.json')
    require(read(path)==receipt['result'],'Fresh saved output differs from actual worker receipt')
    return digest(path)

def reproduce(bundle,images,output,python,entry,deadline):
    bundle=Path(bundle).resolve();output=Path(output).resolve();images=Path(images).resolve()
    require(not output.exists() and not output.is_relative_to(bundle) and entry<=time.time()<deadline<=entry+2700,'One fixed fresh2700 attempt required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONNOUSERSITE')=='1','CPU parent required')
    cpus=sorted(os.sched_getaffinity(0));require(len(cpus)>=4,'Four CPUs required');os.sched_setaffinity(0,set(cpus[:4]))
    setup_end=min(entry+300,deadline-2400);arm(setup_end);resource_guard(admission=True);recipe=verify_bundle(bundle);output.mkdir()
    closure=dependency_contract(bundle/'runtime_lock.json',(bundle,))
    from .weight_archive import restore
    restored={}
    for role in ROLES:
        row=recipe['models'][role];path=restore(bundle/row['archive'],output/'assembled'/role)
        require(path.name==row['filename'] and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Exact model restore differs');restored[role]=str(path)
    inventory=read(bundle/'test_inputs.json')['images']
    require(len(inventory)==180 and [r['image_id'] for r in inventory]==sorted({r['image_id'] for r in inventory}),'Ordered180 inventory required')
    for row in inventory:
        name=Path(row['file_name']);require(name.name==str(name) and name.suffix=='.jpeg','Official JPEG basename required')
        path=images/name;require(not path.is_symlink() and path.is_file() and digest(path)==row['jpeg_sha256'],'Official external JPEG changed')
    require(time.time()<=setup_end,'Fresh setup300 exceeded');save(output/'setup_complete.json',{'recipe_sha256':digest(bundle/'recipe.json'),'runtime':closure,'completed_epoch':time.time()})
    exits=[];inference_seal=None
    for stage,end in [('infer',min(entry+2100,deadline-600)),('audit',min(entry+2550,deadline-150))]:
        arm(end);inputs=output/(stage+'_inputs.json')
        save(inputs,{'bundle':str(bundle),'output':str(output),'images':str(images),'models':restored,'parent':identity(),
          'recipe_sha256':digest(bundle/'recipe.json'),'deadline_epoch':end,'preceding_inference_complete_sha256':inference_seal})
        actual=run_child([str(python),'-B','-m','dual_yolo.worker','--stage',stage,'--inputs',str(inputs)],cwd=bundle,
          env=environment(stage=='infer',output/(stage+'_config')),output=output,stage=stage,deadline=end,inputs=inputs)
        receipt=read(output/(stage+'_receipt.json'))
        require(receipt['worker']==actual['worker'] and receipt['parent']==actual['parent'] and receipt['recipe_sha256']==digest(bundle/'recipe.json')
          and receipt['deadline_epoch']==end and receipt['completed_epoch']<=end and receipt['external_exit_status'] is None
          and type(actual['normal_wait_returncode']) is int and actual['normal_wait_returncode']==0,'Fresh owned child handoff differs')
        seal=sealed_worker_result(output,stage,receipt)
        if stage=='infer':inference_seal=seal
        exits.append({'stage':stage,'actual':actual,'receipt_sha256':digest(output/(stage+'_receipt.json'))})
    arm(deadline);require(time.time()<deadline,'Fresh2700 exceeded');verified=read(output/'audit_complete.json')
    save(output/'reproduction_receipt.json',{'status':'PASS_full180_exact_local_clone','recipe_sha256':digest(bundle/'recipe.json'),'candidate_sha256':digest(output/'candidate.csv'),
      'images':180,'instances':verified['instances'],'stages':exits,'entry_epoch':entry,'deadline_epoch':deadline,'completed_epoch':time.time(),
      'exact_final_output_only':True,'raw_intermediates_byte_equal_claim':False,'each_pass_independently_raw_sealed_and_audited':True,
      'fresh_installation_claim':False,'historical_Kaggle_runtime_claim':False,'outer_owned_session_cleanup_required':True,'submission':False})

