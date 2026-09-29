"""Fixed first examples of composed intervention, from verified frozen predictions."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from common import HERE,dump,r
from object_edit_confirmation import NAME,freeze
from object_image_edit import load_controller,apply_commands
from object_oracle_study import image_from_slots


@torch.no_grad()
def main():
    freeze();torch.set_num_threads(2);root=HERE/f'reports/{NAME}'
    assert json.loads((root/'verification.json').read_text())['all_passed']
    examples=[]
    for split in ['test','count4','occlusion']:
        folder=HERE/f'data/{NAME}/{split}';records=json.loads((folder/'tasks.json').read_text());j=next(i for i,v in enumerate(records) if v['operation']=='color_then_rotate');rec=records[j]
        with np.load(folder/'queries.npz') as a:commands=torch.from_numpy(a['commands'].copy());indices=a['source_indices'].copy()
        with np.load(folder/'rgb.npz') as a:input_rgb=a['images'][rec['source_index']].copy()
        with np.load(folder/'scene_labels.npz') as a:truth=torch.from_numpy(a['states'][indices].copy())
        ids=torch.tensor([v['target_object_id'] for v in records]);gt1=apply_commands(truth,ids,commands[:,:1],'analytic');gt2=apply_commands(truth,ids,commands,'analytic')
        with np.load(folder/'targets.npz') as a:np.testing.assert_array_equal(image_from_slots(gt2[j]),a['images'][j])
        frames={'ground_truth':[input_rgb,image_from_slots(gt1[j]),image_from_slots(gt2[j])]}
        for kind in ['flat','slot']:
            path=HERE/f'runs/{NAME}/{kind}_s0_object/{split}/predictions.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
            model=load_controller(0);p1=apply_commands(saved['source_states'],saved['selected_slots'],commands[:,:1],model)
            p2=apply_commands(saved['source_states'],saved['selected_slots'],commands,model)
            torch.testing.assert_close(p2,saved['predicted_states'],atol=0,rtol=0)
            frames[kind]=[image_from_slots(saved['source_states'][j]),image_from_slots(p1[j]),image_from_slots(p2[j])]
        examples.append({'split':split,'task_index':j,'source_index':rec['source_index'],'input':input_rgb,'click':rec['click_xy'],'frames':frames})
    try:
        font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',16)
        row_font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14)
    except OSError:font=row_font=ImageFont.load_default()
    width=900;height=790;patch=192;lefts=[90,294,498,702];tops=[90,318,546];rendered=[]
    titles=['0: Read the scene','1: Change target color','2: Rotate that target +90 degrees']
    for step,title in enumerate(titles):
        canvas=Image.new('RGB',(width,height),'#f4f4f1');draw=ImageDraw.Draw(canvas)
        draw.text((15,10),title,font=font,fill='#222');draw.text((15,38),'Fixed first composed task per condition; seed 0; no intermediate truth to models',font=font,fill='#333')
        for x,label in zip(lefts,['Input + click','Ground truth','Flat output','Slot output']):draw.text((x,67),label,font=font,fill='#222')
        for y,example in zip(tops,examples):
            draw.text((4,y+75),example['split'],font=row_font,fill='#222')
            arrays=[example['input']]+[example['frames'][key][step] for key in ['ground_truth','flat','slot']]
            for col,(x,array) in enumerate(zip(lefts,arrays)):
                canvas.paste(Image.fromarray(array).resize((patch,patch),Image.Resampling.NEAREST),(x,y))
                if col==0:
                    cx,cy=example['click'];cx=x+(cx+.5)*3;cy=y+(cy+.5)*3
                    draw.ellipse((cx-5,cy-5,cx+5,cy+5),outline='white',width=2)
        rendered.append(canvas);canvas.save(root/f'edit_sequence_frame_{step}.png')
    rendered[0].save(root/'edit_sequence.gif',save_all=True,append_images=rendered[1:],duration=[1800,1800,2400],loop=0,disposal=2)
    with Image.open(root/'edit_sequence.gif') as gif:assert gif.n_frames==3 and gif.size==(width,height)
    dump(root/'video_manifest.json',{'selection':'First color_then_rotate task in test/count4/occlusion; fixed head/controller seed 0, no quality selection.',
        'examples':[{k:v for k,v in e.items() if k not in ['input','frames']} for e in examples],
        'files':{f:r.sha(root/f) for f in ['edit_sequence.gif']+[f'edit_sequence_frame_{i}.png' for i in range(3)]},
        'final_predictions_replayed_exactly':True,'intermediate_ground_truth_excluded_from_models':True,'script_sha256':r.sha(__file__)})
    print('Saved three-frame composed-edit GIF and lossless PNG frames')


if __name__=='__main__':main()
