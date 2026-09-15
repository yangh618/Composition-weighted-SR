#!/bin/bash

# Composition-weighted Symbolic Regression (CWSR) runner script (legacy example).
# Invokes scripts/run_cwsr.py next to this file, keeping the caller's cwd so
# relative paths like --output_dir ./test resolve where you started.

echo "Running CWSR with all available options..."

python "$(dirname "$0")/run_cwsr.py" \
       --task matbench_expt_gap \
       --fold -1 \
       --ops mul sub add div sqrt exp log Max Min Pow R\
       --var_count 3 \
       --K 1000 \
       --gp_rate 0.2 \
       --mutation_rate 0.5 \
       --max_constants 8 \
       --exploration_rate 0.2 \
       --max_depth 6 \
       --max_expressions 10000 \
       --save_every 5000 \
       --save_checkpoint_every 5000 \
       --optimization_method LD_LBFGS \
       --output_dir ./test \
       --num_parallel 12 \
       --num_batches 48      \
       --num_trials 2 #2>/tmp/NUL

echo "CWSR run completed!"
