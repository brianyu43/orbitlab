"""Generate the R2 report after exhaustive saved-output verification completes."""
import json,os,subprocess,sys,time
import r2_decoder_core as c
from v3_common import HERE,ROOT,dump

def main():
    while not (c.BASE/'evaluation_complete.json').exists():
        live=json.loads((c.BASE/'live_evaluation_process.json').read_text())
        try:os.kill(live['driver_pid'],0)
        except ProcessLookupError:raise RuntimeError('Decoder evaluator stopped before completion; investigate, do not infer success')
        except PermissionError:pass
        time.sleep(10)
    command=[sys.executable,str(HERE/'r2_report.py')]
    with (c.w.BASE/'report.log').open('a') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    dump(c.w.BASE/'report_generated.json',{'ended_unix':time.time(),'visual_review':'pending','research_scope':'R2 only; R4/R5 unfinished'})
    print('R2 report ready for visual review',flush=True)

if __name__=='__main__':main()
