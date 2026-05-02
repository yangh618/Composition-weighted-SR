
# Table of Contents

1.  [Quick start](#org751be22)
2.  [Usage](#orga368360)
3.  [CLI (Matbench example)](#org54774a4)
4.  [Evaluate](#orgde6de4b)
5.  [License](#orgdb2d686)

Symbolic regression for materials science. Discovers interpretable mathematical
expressions that map chemical compositions to target properties.


<a id="org751be22"></a>

# Quick start

    conda env create -f environment.yml
    conda activate cwsr
    pip install -e .


<a id="orga368360"></a>

# Usage

Write your own script using the `Regressor` class. Here is a minimal example:

    import numpy as np
    from iMCTS import Regressor
    
    # Your data: compositions (n_samples, 118) and targets (n_samples,)
    x_train = np.load("compositions.npy")
    y_train = np.load("targets.npy")
    
    model = Regressor(
        x_train=x_train,
        y_train=y_train,
        ops=["mul", "sub", "add", "div", "sqrt", "exp", "log"],
        var_count=3,
        max_expressions=5000,
        max_depth=6,
        K=100,
        num_parallel=8,
    )
    
    sym_exp, vec_exp, evals, path, outputs = model.fit()
    print(f"Best expression: {sym_exp}")

See `run_cwsr.py` for a complete example using Matbench datasets.


<a id="org54774a4"></a>

# CLI (Matbench example)

The included `run_cwsr.py` demonstrates CWSR on Matbench benchmarks:

    python run_cwsr.py --max_expressions 200

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Argument</th>
<th scope="col" class="org-right">Default</th>
<th scope="col" class="org-left">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>--ops</code></td>
<td class="org-right">mul sub add div sqrt exp log R</td>
<td class="org-left">Operators</td>
</tr>

<tr>
<td class="org-left"><code>--var_count</code></td>
<td class="org-right">4</td>
<td class="org-left">Number of latent variables</td>
</tr>

<tr>
<td class="org-left"><code>--max_expressions</code></td>
<td class="org-right">200</td>
<td class="org-left">Max expressions to evaluate</td>
</tr>

<tr>
<td class="org-left"><code>--max_depth</code></td>
<td class="org-right">6</td>
<td class="org-left">Max expression tree depth</td>
</tr>

<tr>
<td class="org-left"><code>--max_constants</code></td>
<td class="org-right">6</td>
<td class="org-left">Max number of constants</td>
</tr>

<tr>
<td class="org-left"><code>--K</code></td>
<td class="org-right">10</td>
<td class="org-left">MCTS exploration parameter</td>
</tr>

<tr>
<td class="org-left"><code>--gp_rate</code></td>
<td class="org-right">0.0</td>
<td class="org-left">Genetic programming rate</td>
</tr>

<tr>
<td class="org-left"><code>--mutation_rate</code></td>
<td class="org-right">0.2</td>
<td class="org-left">Mutation rate</td>
</tr>

<tr>
<td class="org-left"><code>--exploration_rate</code></td>
<td class="org-right">0.2</td>
<td class="org-left">Exploration rate</td>
</tr>

<tr>
<td class="org-left"><code>--optimization_method</code></td>
<td class="org-right">LD<sub>LBFGS</sub></td>
<td class="org-left">NLopt optimization method</td>
</tr>

<tr>
<td class="org-left"><code>--num_parallel</code></td>
<td class="org-right">8</td>
<td class="org-left">Parallel processes</td>
</tr>

<tr>
<td class="org-left"><code>--num_batches</code></td>
<td class="org-right">64</td>
<td class="org-left">Batches per MCTS iteration</td>
</tr>

<tr>
<td class="org-left"><code>--num_trials</code></td>
<td class="org-right">1</td>
<td class="org-left">Optimization trials</td>
</tr>

<tr>
<td class="org-left"><code>--seed</code></td>
<td class="org-right">None</td>
<td class="org-left">Random seed</td>
</tr>

<tr>
<td class="org-left"><code>--output_dir</code></td>
<td class="org-right">.</td>
<td class="org-left">Output directory</td>
</tr>

<tr>
<td class="org-left"><code>--checkpoint</code></td>
<td class="org-right">None</td>
<td class="org-left">Resume from checkpoint</td>
</tr>

<tr>
<td class="org-left"><code>--save_checkpoint_every</code></td>
<td class="org-right">0</td>
<td class="org-left">Save checkpoint every N evaluations</td>
</tr>

<tr>
<td class="org-left"><code>--save_every</code></td>
<td class="org-right">0</td>
<td class="org-left">Save intermediate results every N</td>
</tr>
</tbody>
</table>


<a id="orgde6de4b"></a>

# Evaluate

    python eval.py --model_path ./outputs.json


<a id="orgdb2d686"></a>

# License

MIT

