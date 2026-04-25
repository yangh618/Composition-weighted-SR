#!/bin/bash

# Tabulated Invariant Symbolic Regression (TISR) Runner Script

echo "Running TISR with all available options..."

python run_cwsr.py \
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
       --max_expressions 1000 \
       --save_every 1000 \
       --optimization_method LD_LBFGS \
       --seed 42 \
       --output_dir ./test \
       --num_parallel 12 \
       --num_batches 48      \
       --num_trials 2 #2>/tmp/NUL

echo "TISR run completed!"
