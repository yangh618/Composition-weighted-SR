#!/bin/bash

# Tabulated Invariant Symbolic Regression (TISR) Runner Script

echo "Running TISR with all available options..."

python run_tisr.py \
       --task matbench_glass \
       --fold -1 \
       --ops mul sub add div sqrt exp log Max Min Pow R\
       --var_count 3 \
       --K 1000 \
       --gp_rate 0.2 \
       --mutation_rate 0.5 \
       --max_constants 8 \
       --exploration_rate 0.2 \
       --max_depth 6 \
       --max_expressions 100 \
       --optimization_method LD_LBFGS \
       --num_parallel 1 \
       --num_batches 1      \
       --num_trials 1 #2>/tmp/NUL

echo "TISR run completed!"
