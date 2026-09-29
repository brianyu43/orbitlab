"""Describe sampling and early/late state-distribution effects without refitting."""
import json
import torch
from common import HERE,dump,r
from dynamics_models import load_split


def main():
    torch.set_num_threads(2);rows=[]
    for study in ['dynamics_pilot_v1','dynamics_sampling_v1']:
        cp=HERE/f'configs/{study}.json';c=json.loads(cp.read_text())
        for mode in c['modes']:
            data=load_split(HERE/'data/dynamics_world_v1',mode,'val');live=data['state'][...,-1]
            time=torch.arange(len(data['state']))%64
            for kind in c['kinds']:
                folder=HERE/f'runs/{study}/{mode}/{kind}_s0';saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True)
                for name,key in [('learned','prediction'),('inertial','inertial')]:
                    diff=(saved[key]-data['next']).abs();pos=(diff[...,:2].sum(-1)*live).sum(-1)*31.5/(2*live.sum(-1))
                    vel=(diff[...,2:].sum(-1)*live).sum(-1)*3/(2*live.sum(-1))
                    for label,condition in [('train_time_range',time<19),('later_states',time>=19)]:
                        for category,contact in [('all',torch.ones_like(condition)),('no_contact',data['events'].sum(-1)==0),('contact',data['events'].sum(-1)>0)]:
                            per_episode=[]
                            for episode in range(64):
                                select=condition&contact&(data['episode']==episode)
                                if select.any():per_episode.append((float(pos[select].mean()),float(vel[select].mean())))
                            rows.append({'study':study,'mode':mode,'kind':kind,'baseline':name,'state_time':label,'category':category,
                                'episodes':len(per_episode),'position_mae_pixels':sum(v[0] for v in per_episode)/len(per_episode),
                                'velocity_mae_pixels_per_frame':sum(v[1] for v in per_episode)/len(per_episode)})
    out=HERE/'reports/dynamics_pilot_v1';r.write_csv(out/'horizon_diagnostics.csv',rows)
    dump(out/'horizon_diagnostics.json',{'rows':rows,'diagnostic_sha256':r.sha(__file__),
        'definition':'Train time range uses transitions t=0..18, as present in train; later states use t=19..63. Both receive the true state at every step.',
        'claim_boundary':'Post-hoc descriptive diagnostic on validation, not autonomous multi-step prediction or a test-based confirmation.'})


if __name__=='__main__':main()
