"""Replay every fixed-panel target and test pixel-metric identities."""
import json
from pathlib import Path
import numpy as np
import r2_decoder_core as c
import r2_decoder_data as d
import r2_decoder_evaluate as e
from v3_common import dump,sha

def main():
    c.freeze();e.freeze();checks=[]
    for split,count in [('train',2),('val',2),('val',3),('val',4),('val',6)]:
        folder=c.BASE/f'data/d{c.w.DEVELOPMENT}/{split}/n{count}';m=json.loads((folder/'manifest.json').read_text())
        assert m['protocol_sha256']==sha(c.BASE/'protocol.json')
        for name,h in m['files'].items():assert sha(folder/name)==h
        expected=d.build(c.w.DEVELOPMENT,split,count);saved=np.load(folder/'targets.npz')
        np.testing.assert_array_equal(saved['targets'],expected['targets'])
        for key in ['panels','non_targets','foreground']:np.testing.assert_array_equal(saved[key],np.rint(expected[key]*255).astype(np.uint8))
        assert expected['commands']==json.loads((folder/'commands.json').read_text());assert expected['metadata']==json.loads((folder/'metadata.json').read_text())
        checks.append({'split':split,'count':count,'all_targets_replayed':len(expected['targets']),'fixed_opaque_panel_and_identity_checked':True,'manifest_sha256':sha(folder/'manifest.json')})
        print('Decoder targets verified',split,count,flush=True)
    source,objects,edited,commands,indices=e.inputs(2,'truth_state');folder=c.BASE/f'data/d{c.w.DEVELOPMENT}/val/n2';archive=np.load(folder/'targets.npz');meta=json.loads((folder/'metadata.json').read_text())
    n=96;target=archive['targets'][indices[:n]];records=json.loads((c.w.BASE/f'data/d{c.w.DEVELOPMENT}/val/n2/labels.json').read_text());fg=[];after=[]
    for idx in indices[:n]:
        fg.append(c.truth_render(records[idx//4]['objects'],np.zeros((64,64,3),np.float32),np.zeros((64,64),np.float32))[1]);after.append(c.truth_render(meta[idx]['objects'],np.zeros((64,64,3),np.float32),np.zeros((64,64),np.float32))[1])
    rows=e.rows_for(2,'truth_state',source[:n],target,np.stack(fg),np.stack(after),objects[:n],commands[:n],indices[:n])
    assert all(r['pixel_gate']==1 and r['whole_mae']==0 and r['source_foreground_alpha_iou']==1 and r['edited_foreground_alpha_iou']==1 for r in rows)
    copy=e.rows_for(2,'copy_input',source[:n],source[:n],None,None,objects[:n],commands[:n],indices[:n])
    assert all(r['preserved_mae']==0 for r in copy);assert any(r['changed_mae']>.05 and not r['pixel_gate'] for r in copy)
    # Fractional occluder boundary pixels mix visible object and panel; changing
    # the object legitimately changes this composite. Only fully opaque panel
    # pixels are required to stay exactly equal, as checked for every target.
    mixed=sum(r['panel_mae']>0 for r in copy)
    assert mixed>0
    dump(c.BASE/'data_verification.json',{'groups':checks,'target_images':sum(r['all_targets_replayed'] for r in checks),'all_targets_and_fixed_opaque_panels_replayed':True,'metric_identity_checks':n,'copy_baseline_has_exact_unchanged_region_preservation':True,
        'fractional_panel_composites_with_nonzero_copy_error':mixed,'metric_preflight_correction':'An initial inline test incorrectly required zero copy-input error in fractional panel boundary composites. No dataset, model, protocol, thresholds or evaluator was changed. Only the test expectation was corrected: the panel itself is fixed, while visible object contribution at fractional edges may change.',
        'verifier_sha256':sha(__file__),'data_source_sha256':sha(Path(__file__).with_name('r2_decoder_data.py')),'evaluator_sha256':sha(Path(__file__).with_name('r2_decoder_evaluate.py'))})
    print('Decoder target and metric verification passed',flush=True)

if __name__=='__main__':main()
