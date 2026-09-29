# Mac GPU scheduling amendment

The user requested more aggressive Mac GPU use to reduce completion time. The host is an Apple M5 Pro with 64GB unified memory and 18 logical CPU cores. Primary external-model training already used MPS float32.

A short discarded-weight benchmark retained batch32, Adam, clipping, finite checks and native128px inputs. Two concurrent primary workers achieved an estimated compute throughput of1.629x serial, while three achieved1.655x. The third worker added only1.54%, below the predeclared10% threshold, so the live scheduler keeps two primary workers. These measurements exclude PNG loading and do not establish a1.63x reduction in total project time. Separate worker startup and the different architecture mixture are retained in the receipts.

The first benchmark briefly suspended the original C4 worker for35.62seconds. The three-worker check suspended the C4 and object workers for37.97seconds. Both checks resumed all workers and discarded benchmark weights. Scientific training wall times include these pauses; they must not be described as isolated architecture speed measurements.

The new controller temporarily stops only the original dispatcher while independent models train, then resumes it to run its unchanged validation selection and evaluation sequence. Original dispatcher PIDs remain available to the existing dependent jobs. Architecture, initialization, data order, batch size,36000updates, optimizer, precision and evaluation thresholds are unchanged. Each model keeps its own output directory and atomic checkpoint; no duplicate writers are intentionally started. A separate watchdog can release a dispatcher if its coordinator disappears, after ending the coordinator's unfinished additional workers.

See `gpu_parallel_protocol.json`, `GPU_SCHEDULING_DECISION.json`, and `gpu_schedule_benchmark/`. The original12hour/30GB stage caps still apply. At most two primary training workers does not exclude short concurrent GPU validation/readout jobs. CPU-only completed studies and frozen measurement instruments are not silently relabeled as GPU runs.
