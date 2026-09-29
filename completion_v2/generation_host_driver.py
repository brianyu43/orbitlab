"""Host execution for the pre-recorded confirmation accelerator amendment."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from driver import run,wait_for,BASE

wait_for(BASE/'generation/runs/d880101_s0/training_summary.json')
run('generation_dev_evaluation','generation_evaluate.py')
for seed in [880201,880202,880203]:
    for init in range(3):
        run(f'g_train_{seed}_{init}','generation_mps.py','--seed',seed,'--init',init)
        run(f'g_eval_{seed}_{init}','generation_evaluate.py','--seed',seed,'--init',init)
(BASE/'generation_driver_completed.json').write_text(json.dumps({'completed_unix':time.time(),'branch':'generation','device_amendment':True})+'\n')
