#!/bin/bash

echo "Running TISR evaluation..."

python eval.py \
       --model_path ./tisr_outputs_matbench_expt_is_metal_fold0_1769506456.json \
       --task matbench_expt_is_metal \
       --fold 0 \
       --ops mul sub add div exp log Max Min Pow R\
       --var_count 2 \
       --K 1000 \
       --gp_rate 0.2 \
       --mutation_rate 0.5 \
       --max_constants 8 \
       --exploration_rate 0.2 \
       --max_depth 6 \
       --max_expressions 10000 \
       --optimization_method LD_LBFGS \
       --num_parallel 16 \
       --num_batches 64 \
       --num_trials 4 #2>/tmp/NUL

echo "TISR evaluation completed!"
