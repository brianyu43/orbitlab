from driver import run,BASE
import json,time
run('pdetail_prepare','perception_detail_evaluate.py','prepare')
for kind in ['binary','alpha']:
    for init in range(3):
        run(f'pdetail_train_{kind}_{init}','perception_detail.py','--kind',kind,'--seed',init)
        for seed in [885201,885202,885203]:
            run(f'pdetail_eval_{kind}_{init}_{seed}','perception_detail_evaluate.py','evaluate','--kind',kind,'--init',init,'--seed',seed)
(BASE/'perception_detail_driver_completed.json').write_text(json.dumps({'completed_unix':time.time()}))
