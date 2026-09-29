"""Continue saved v2 pilots, including Adam and both sampling RNG states.

Fixed 6k/12k checkpoints are validation diagnostics, not test-selected models.
"""
import json
import time
from common import HERE, dump, fresh_dir, digest_tensor, state_digest, update_status, torch, o, r
from object_perception import images_only
from object_perception_v2 import LinearImageSlots, balanced_loss, evaluate


def freeze():
    cp=HERE/'configs/object_perception_curve_v1.json'
    if not cp.exists():
        parents={k:r.sha(HERE/f'runs/object_perception_pilot_v2/{k}_s0/model.pt') for k in ['flat','slot']}
        dump(cp,{'kinds':['flat','slot'],'seed':0,'checkpoints':[6000,12000],'start_step':3000,'batch':16,'lr':.0004,'slots':5,'dim':32,
            'parent_checkpoints':parents,'data_manifest_sha256':r.sha(HERE/'data/object_world_v1/manifest.json'),
            'resumption':'Saved model, Adam state, sample RNG and slot-noise RNG. Learning rate continues at 4e-4; no second warmup.',
            'selection':'Evaluate all prespecified checkpoints on validation only. No test/count/occlusion results used to tune.',
            'purpose':'Distinguish undertraining from persistent decomposition failure. One seed; not a final independent result.',
            'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def restore(folder,device):
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=LinearImageSlots(ck['kind'],ck['dim'],ck['slots']).to(device);model.load_state_dict(ck['state_dict']);model.train()
    opt=torch.optim.Adam(model.parameters(),lr=.0004);saved=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
    opt.load_state_dict(saved['optimizer'])
    rng=torch.Generator();rng.set_state(saved['sampling_rng'])
    noise_rng=torch.Generator();noise_rng.set_state(saved['noise_rng'])
    return model,opt,rng,noise_rng


def continue_run(parent,kind,end,c,device):
    folder=HERE/f'runs/object_perception_curve_v1/{kind}_s0_step{end}'
    if (folder/'run.json').exists():
        rr=json.loads((folder/'run.json').read_text());assert r.sha(folder/'model.pt')==rr['checkpoint_sha256'];return folder
    previous=json.loads((parent/'run.json').read_text());assert r.sha(parent/'model.pt')==previous['checkpoint_sha256']
    start_step=previous['steps'];assert start_step<end
    model,opt,rng,noise_rng=restore(parent,device);x=images_only('train');assert digest_tensor(x)==previous['training_image_sha256']
    assert digest_tensor(rng.get_state())==previous['sampling_rng_sha256']
    for state in opt.state.values():assert int(state['step'])==start_step
    fresh_dir(folder);initial=state_digest(model)
    dump(folder/'started.json',{'kind':kind,'parent':str(parent.relative_to(HERE)),'parent_model_sha256':r.sha(parent/'model.pt'),
        'parent_optimizer_sha256':r.sha(parent/'optimizer.pt'),'start_step':start_step,'target_step':end,'code_sha256':r.sha(__file__)})
    o.sync(device);start=time.perf_counter();logs=[]
    for step in range(start_step+1,end+1):
        idx=torch.randint(len(x),(c['batch'],),generator=rng);target=x[idx].to(device)
        eps=torch.randn(c['batch'],c['slots'],c['dim'],generator=noise_rng).to(device)
        pred=model(target,eps);loss=balanced_loss(pred['image'],target)
        if not torch.isfinite(loss):raise FloatingPointError('Curve loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Curve gradient')
        opt.step()
        if step%500==0:
            o.sync(device);logs.append({'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-start});print(folder.name,logs[-1],flush=True)
    o.sync(device);seconds=time.perf_counter()-start
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'kind':kind,'dim':c['dim'],'slots':c['slots'],'rgb':'linear'},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state(),'noise_rng':noise_rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'kind':kind,'steps':end,'continued_from_step':start_step,'new_steps':end-start_step,
        'parent':str(parent.relative_to(HERE)),'parent_model_sha256':r.sha(parent/'model.pt'),'parent_optimizer_sha256':r.sha(parent/'optimizer.pt'),
        'initial_weights_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),'incremental_train_seconds':seconds,
        'training_image_sha256':digest_tensor(x),'sampling_rng_sha256':digest_tensor(rng.get_state()),'noise_rng_sha256':digest_tensor(noise_rng.get_state()),
        'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':r.sha(HERE/'configs/object_perception_curve_v1.json'),
        'code_sha256':r.sha(__file__),'model_code_sha256':r.sha(HERE/'object_perception_v2.py'),'loss_log':logs})
    return folder


def main():
    c=freeze();torch.set_num_threads(4);device=o.choose_device('mps')
    parents={k:HERE/f'runs/object_perception_pilot_v2/{k}_s0' for k in c['kinds']};rows=[]
    update_status('B04','in_progress',['configs/object_perception_curve_v1.json','reports/object_perception_pilot_v2/RESULTS_KO.md'],
                  note='Continuing saved 3k pilots through prespecified 6k/12k validation checkpoints; full evaluation pending.')
    for end in c['checkpoints']:
        pair=[]
        for kind in c['kinds']:
            folder=continue_run(parents[kind],kind,end,c,device);evaluate(folder,c,device);parents[kind]=folder
            run=json.loads((folder/'run.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
            rows.append({'kind':kind,'steps':end,'run':run,'metrics':metrics});pair.append(run)
            dump(HERE/'reports/object_perception_curve_v1/progress.json',{'completed_cells':len(rows),'expected_cells':4,'rows':rows})
        assert pair[0]['sampling_rng_sha256']==pair[1]['sampling_rng_sha256']
        assert pair[0]['noise_rng_sha256']==pair[1]['noise_rng_sha256']
    dump(HERE/'reports/object_perception_curve_v1/summary.json',{'rows':rows,'new_training_steps':18000,'validation_only':True,
        'claim_boundary':c['purpose'],'config_sha256':r.sha(HERE/'configs/object_perception_curve_v1.json')})


if __name__=='__main__':main()
