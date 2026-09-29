"""Close only after every prescribed experiment and replay has completed."""
from driver import BASE,wait_for,run
import json,time
ready=BASE/'reports_ready.json'
if not ready.exists():
    wait_for(BASE/'verification_driver_completed.json')
    wait_for(BASE/'generation_reconstruction/summary.json')
    run('final_cost_inventory','cost_inventory.py')
    run('final_scope_audit','close_audit.py')
    run('final_live_summary','summarize.py')
    run('final_figures','make_figures.py')
    run('final_reports','write_reports.py')
    ready.write_text(json.dumps({'ready_unix':time.time(),'pending':'Visual inspection and review archive verification'})+'\n')
