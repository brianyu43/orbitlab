"""Check nontrivial C4 equivariance and exactly paired base-model initializations."""
import torch
from common import HERE,dump,r,state_digest,digest_tensor
from svib_preview_data import NAME,ALPHAS,alpha_name
from svib_preview_models import PreviewPredictor
from svib_preview_metrics import load_images


@torch.no_grad()
def main():
    torch.set_num_threads(2);initials=[]
    for seed in [0,1,2]:
        models=[]
        for kind in ['plain','c4']:
            torch.manual_seed(773000+seed);models.append(PreviewPredictor(kind))
        assert state_digest(models[0])==state_digest(models[1])
        initials.append({'seed':seed,'initial_weights_sha256':state_digest(models[0]),'parameters':sum(p.numel() for p in models[0].parameters())})
    torch.manual_seed(98173);x=torch.rand(2,3,128,128)
    model=PreviewPredictor('c4')
    torch.testing.assert_close(model(x),x,atol=0,rtol=0)
    # Avoid validating only the trivial identity initialization.
    model.decoder[-1].weight.normal_(0,.03);model.decoder[-1].bias.normal_(0,.01)
    output=model(x);assert float((output-x).abs().max())>.001
    gaps=[]
    for k in [1,2,3]:
        rotated=model(torch.rot90(x,k,(-2,-1)));expected=torch.rot90(output,k,(-2,-1))
        torch.testing.assert_close(rotated,expected,atol=2e-6,rtol=2e-6);gaps.append(float((rotated-expected).abs().max()))
    source_checks=[]
    for alpha in ALPHAS:
        source,target,_=load_images(f'{alpha_name(alpha)}/train');assert source.shape==target.shape==(70,3,128,128)
        same_input=source[:2].clone();before=model(same_input)
        target=target.flip(0) # Labels are not arguments to the predictor.
        torch.testing.assert_close(model(same_input),before,atol=0,rtol=0)
        assert torch.isfinite(before).all() and before.min()>=0 and before.max()<=1
        source_checks.append({'alpha':alpha,'train_source_sha256':digest_tensor(source)})
    dump(HERE/f'reports/{NAME}/model_preflight.json',{'all_passed':True,'paired_initializations':initials,
        'nontrivial_C4_max_gaps':gaps,'initial_prediction_is_identity':True,'source_only_call_contract_checked':True,
        'training_sources':source_checks,'sources':{f:r.sha(HERE/f) for f in ['svib_preview_models.py','preflight_svib_preview_models.py','svib_preview_metrics.py']},
        'scope':'Architecture and input preflight, not trained external-task accuracy or matched compute.'})
    print('SVIB predictor preflight passed',initials[0]['parameters'],'parameters',flush=True)


if __name__=='__main__':main()
