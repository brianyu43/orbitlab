"""Fixed first-episode examples; draw predictions without correcting physics."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import findfont
from common import HERE,dump,r
from dynamics_world import render
from dynamics_autonomous_study import NAME


def main():
    root=HERE/f'reports/{NAME}';assert json.loads((root/'verification.json').read_text())['all_passed']
    font_path=findfont('DejaVu Sans');font=ImageFont.truetype(font_path,13);title_font=ImageFont.truetype(font_path,19)
    groups=['test','count4','force_ood'];rows=[];inputs=[]
    for split in groups:
        with np.load(HERE/f'data/dynamics_world_v1/variable_force/{split}/trajectories.npz') as a:
            actual=a['states'][0,3:].copy()
            opposite=a['counterfactual_states'][0,3:].copy() if 'counterfactual_states' in a else None
        series=[actual,opposite]
        for method in ['oracle','measurement_mlp']:
            folder=HERE/f'runs/{NAME}/{method}_s0/variable_force/{split}'
            with np.load(folder/'initial_inputs.npz') as a:initial=a['initial'][0].copy()
            with np.load(folder/'joint_exact/trajectories.npz') as a:
                for key in ['motion','cf_motion']:
                    if key not in a:series.append(None);continue
                    sequence=np.broadcast_to(initial,(62,4,7)).copy();sequence[:,:,:4]=a[key][0]
                    series.append(sequence)
            inputs.append({'method':method,'split':split,'episode':0,'seed':0,
                           'prediction_sha256':r.sha(folder/'joint_exact/trajectories.npz')})
        rows.append(series)
    frames=[];size=160;stride=172;width=1052;height=745
    columns=[('True future','original action'),('True future','opposite action'),
             ('Oracle + joint','original action'),('Oracle + joint','opposite action'),
             ('Measured + joint','original action'),('Measured + joint','opposite action')]
    for h in range(62):
        frame=Image.new('RGB',(width,height),'#f6f7fa');draw=ImageDraw.Draw(frame)
        draw.text((15,10),f'Autonomous future: horizon {h:02d} / 61   |   same past, changed action',font=title_font,fill='#222')
        for col,(a,b) in enumerate(columns):
            x=15+col*stride;draw.text((x,45),a,font=font,fill='#222');draw.text((x,63),b,font=font,fill='#555')
        for row,(split,series) in enumerate(zip(groups,rows)):
            y=95+row*204;draw.text((15,y),f'Variable force / {split} / fixed episode 0',font=font,fill='#333')
            for col,sequence in enumerate(series):
                x=15+col*stride;top=y+22
                if sequence is None:
                    draw.rectangle((x,top,x+size,top+size),fill='#e5e7eb')
                    draw.text((x+7,top+61),'No paired data',font=font,fill='#555')
                    continue
                state=sequence[h];present=state[:,6]>.5
                if not np.isfinite(state[present]).all():
                    draw.rectangle((x,top,x+size,top+size),fill='#442222')
                    draw.text((x+8,top+60),'Numerical failure',font=font,fill='white')
                elif not present.any():
                    draw.rectangle((x,top,x+size,top+size),fill='black')
                    draw.text((x+15,top+60),'No object found',font=font,fill='white')
                else:
                    rgb=render(state[present])['image'];tile=Image.fromarray(rgb).resize((size,size),Image.Resampling.NEAREST)
                    frame.paste(tile,(x,top))
                draw.rectangle((x-1,top-1,x+size,top+size),outline='#adb5c0',width=1)
        draw.text((15,715),'Fixed first examples, seed 0. Colors persist by construction. Offscreen predictions are not moved back.',font=font,fill='#555')
        frames.append(frame)
    frames[0].save(root/'autonomous_examples.gif',save_all=True,append_images=frames[1:],duration=120,loop=0,optimize=False)
    for h in [0,16,61]:frames[h].save(root/f'autonomous_examples_frame{h}.png')
    names=['autonomous_examples.gif']+[f'autonomous_examples_frame{h}.png' for h in [0,16,61]]
    dump(root/'video_manifest.json',{'selection':'First episode 0 of variable_force test/count4/force_ood; seed 0, no outcome-based selection.',
        'frames':62,'duration_ms_per_frame':120,'future_corrections':False,'count4_counterfactual':'No paired trajectory exists; explicitly left unavailable.',
        'inputs':inputs,'files':{name:r.sha(root/name) for name in names},'source_sha256':r.sha(__file__)})
    print('62-frame fixed-example GIF and frames 0/16/61 written')


if __name__=='__main__':main()
