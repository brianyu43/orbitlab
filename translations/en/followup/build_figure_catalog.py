"""Inventory the complete core figure set and its original numerical evidence."""
import json
import time
from PIL import Image
from common import HERE,dump,r


GROUPS=[
 ('Generation quality and calculation cost','generation_cost_figure_v2',['generation_quality_cost.png','generation_quality_cost.svg'],['verification.json','figure_visual_review.json'],'Same number of times/hours, 576 page generation cost, independent verification and AE+flow total time.'),
 ('Object separation and failure','object_failure_audit_v1',['heldout_segmentation.png','heldout_segmentation.svg'],['verification.json','figure_manifest.json'],'Number of objects, obscuration, aggregation, fragments remaining; one independent scene segment, encoder initialization.'),
 ('Actual intervention and combination','object_image_edit_v1',['capability_curves.png','capability_curves.svg','edit_sequence.gif'],['verification.json','analysis_verification.json','video_manifest.json'],'Separate the selection of the target, compliance with the estimated state, and actual overall correct answers; fixed first case video.'),
 ('Separation of state and environmental awareness','dynamics_observation_eval_v1',['state_context_substitution.png','state_context_substitution.svg'],['verification.json','report_manifest.json'],'Correct answer status/environment change comparison; one step of one data t=3 start.'),
 ('A long future and counter-acting behavior','dynamics_autonomous_v1',['long_horizon_curves.png','counterfactual_response.png','stability_overview.png','autonomous_examples.gif'],['verification.json','aggregate_verification.json','report_manifest.json','video_manifest.json'],'Failure rate·limited error·target/non-target reaction, fixed first case; one data.'),
 ('Independent data repetition','dynamics_repeats_v1',['independent_state_comparison.png','independent_rollout_stability.png'],['report_verification.json'],'Separate data sets and initialization repetition; preserve large finite expansion.'),
 ('Environmental information missing','dynamics_context_omission_v1',['context_information_effect.png'],['report_verification.json','figure_visual_review.json'],'Relearn models that cover both 108 conditions, combined/resistance/both.'),
 ('A different future from the same observation','observation_identifiability_v1',['identical_past.png'],['verification.json'],'The boundary pair composed; not the error limit of the entire data.'),
 ('Observation length and all prediction times','dynamics_observation_length_v1',[f'report_v1/{x}.png' for x in ['observation_length_effect','length_rollout_stability','length_horizon_by_data','length_paired_contact','length_action_response']],['report_verification.json','figure_visual_review.json'],'Same t=7, 1/2/4/8 sheets, matching of common limited scenes, all 520 blocks connected.'),
 ('External SVIB public example','svib_preview_shape_swap_v1',['report_v1/preview_comparison.png','report_v1/examples_alpha_0p0.png','report_v1/examples_alpha_0p6.png'],['report_verification.json','figure_visual_review.json'],'Overall error, change area, copy criteria, calculation cost and fixed failure examples.')]


def main():
    rows=[];assets=[]
    for label,folder,files,receipts,note in GROUPS:
        base=HERE/'reports'/folder;evidence={}
        for name in receipts:
            p=base/name;d=json.loads(p.read_text())
            if 'all_passed' in d:assert d['all_passed']
            evidence[str(p.relative_to(HERE))]=r.sha(p)
        for name in files:
            p=base/name;entry={'path':str(p.relative_to(HERE)),'sha256':r.sha(p),'bytes':p.stat().st_size,'group':label}
            if p.suffix in ['.png','.gif']:
                with Image.open(p) as im:
                    entry.update(width=im.width,height=im.height,frames=getattr(im,'n_frames',1))
                    for frame in range(entry['frames']):im.seek(frame);im.load()
            assets.append(entry)
        rows.append({'group':label,'files':files,'folder':folder,'evidence_sha256':evidence,'scope':note})
    lines=['# Key pictures and video list','',
        'Connect the cause separation, calculation costs, condition changes, combinations, and long-term predictions of plan D02 to the following materials. Since the denominator and range of the experiments vary for each figure, they should not be interpreted as continuous performance improvement of a single model. PNG, some SVG, and GIF files are local files.','',
        '| Questions | Representative pictures / videos | Numerical basis and range |','| --- | --- | --- |']
    for row in rows:
        links=' · '.join(f"[{name.split('/')[-1]}]({row['folder']}/{name})" for name in row['files'])
        lines.append(f"| {row['group']} | {links} | {row['scope']} |")
    lines+=['','The observation length diagrams and generation cost diagrams were directly verified during this final review. The intervention curve, the final intervention scene, the information exchange diagram, and the 16-stage scene of the long video were also re-verified. The numerical verification of existing documents and separate visual verification are not the same test. All raster files and GIF frames were opened and verified using an image decoder, and SHA-256 hashes were generated. Color gamut simulation or separate browser rendering for all SVG files were not performed.','',
        '[File list and reference hash](figure_catalog_verification.json) · [Research memo](RESEARCH_MEMO_KO.md)']
    report=HERE/'reports/FIGURE_CATALOG_KO.md';report.write_text('\n'.join(lines)+'\n')
    dump(HERE/'reports/figure_catalog_verification.json',{'all_passed':True,'groups':rows,'assets':assets,'group_count':len(rows),
        'file_count':len(assets),'all_raster_frames_decoded':True,'report_sha256':r.sha(report),'script_sha256':r.sha(__file__),
        'checked_unix':time.time(),'scope':'Complete D02 coverage inventory, source receipts, hashes and image decoding. Original numerical audits retained. Visual review is limited to the views explicitly described in the catalog and earlier recorded reviews.'})
    print('core figure catalog',len(rows),'groups',len(assets),'assets',flush=True)


if __name__=='__main__':main()
