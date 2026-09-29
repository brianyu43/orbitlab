"""One final verification pass per completed branch; no duplicate concurrent writers."""
from driver import BASE,wait_for,run
# Launch only when earlier snapshot verifier processes have exited.
wait_for(BASE/'dynamics_driver_completed.json')
run('final_dynamics_verify','verify_new_results.py','dynamics')
run('final_dynamics_cf_verify','verify_counterfactual.py')
run('final_dynamics_geometry','dynamics_geometry.py')
wait_for(BASE/'perception_detail_driver_completed.json')
run('final_perception_detail_verify','verify_perception_detail.py')
wait_for(BASE/'generation_driver_completed.json')
run('final_generation_verify','verify_new_results.py','generation')
# SVIB has a last evaluation instead of a driver marker.
wait_for(BASE/'svib/runs/c4_s2/evaluation.json')
run('final_svib_verify','verify_svib.py')
run('final_aggregate','aggregate_results.py')
(BASE/'verification_driver_completed.json').write_text('{"completed":true}\n')
