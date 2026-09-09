from __future__ import annotations
import random
import copy
from typing import List, Tuple, Optional
import numpy as np
import multiprocessing as mp
from cwsr.exp_tree import ExpTree
from cwsr.exp_queue import Exp_Queue
from cwsr.reward import Optimizer
from cwsr.gp import GPManager
from collections import deque

class MCTS_Node:
    """Monte Carlo Tree Search node implementation."""
    
    def __init__(
        self,
        mcts: MCTS,
        parent: Optional[MCTS_Node] = None,
        move: str = ""
    ) -> None:
        self.mcts = mcts  # Reference to the MCTS instance for parameters
        self.parent = parent
        self.move = move
        self.children: List[MCTS_Node] = []
        self.visits: int = 0
        self.value: float = 0
        self.unexpanded_moves: List[str] = []
        self.is_terminal: bool = False

    def expand(self, state: ExpTree) -> MCTS_Node:
        """Expand the node by creating a new child node."""
        if not self.unexpanded_moves:
            self.unexpanded_moves = list(state.available_ops)
        
        child = None
        if self.unexpanded_moves:
            selected_index = random.randrange(len(self.unexpanded_moves))
            move = self.unexpanded_moves.pop(selected_index)
            child = MCTS_Node(mcts=self.mcts, parent=self, move=move)
            self.children.append(child)
        
        return child

    @property
    def ucb(self) -> float:
        """Calculate Upper Confidence Bound (UCB) value."""
        q_value = self.value

        exploration_weight = (
            self.mcts.c * np.log(self.parent.visits) / self.visits
        ) ** self.mcts.gamma
        return q_value + exploration_weight
    
    def choose(self) -> MCTS_Node:
        """Select child node with highest UCB value."""
        selected_node = max(self.children, key=lambda c: c.ucb)
        
        if selected_node.is_terminal and selected_node.visits > 0:
            selected_node = max(
                (c for c in self.children if not c.is_terminal),
                default=selected_node,
                key=lambda c: c.ucb
            )
        return selected_node
    
    def random_child(self) -> MCTS_Node:
        """Select a random child node."""
        valid_children = [child for child in self.children if not child.is_terminal]
        if valid_children:
            return random.choice(valid_children)
        return random.choice(self.children)
    
    def backpropagate(self, path: List[str], value: float) -> None:
        """Backpropagate the simulation results through the tree."""
        current_node = self
        while current_node:
            if current_node.move == "":
                break  # Reached root node
            path = [current_node.move] + path
            current_node.value = max(value, current_node.value)
            current_node = current_node.parent
        return path

    def propagate(self, path: List[Tuple], value: float) -> None:
        """Propagate results down the tree hierarchy."""
        current_node = self
        for idx, action in enumerate(path):
            current_node = next(
                (child for child in current_node.children if child.move == action),
                None
            )
            if not current_node:
                break
            current_node.value = max(value, current_node.value)
            current_node.visits += 1

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        # no children or has unexpanded moves
        return not self.children or self.unexpanded_moves

def _rollout_task(state: ExpTree, seed: int = None) -> Tuple[ExpTree, List[str]]:
    if seed is not None:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
    filled_state, path, complex = state.random_fill()
    return filled_state, path, complex

def _optimize_task(optimizer, filled_state: ExpTree, path, seed: int = None) -> Tuple[float, List[str]]:
    if seed is not None:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
    """Rollout task for parallel execution.
    Define globally to avoid pickling issues with multiprocessing.
    """
    expression1, train_reward1, valid_reward1 = optimizer.optimize_constants(filled_state, is_positive_init=True)
    expression2, train_reward2, valid_reward2 = optimizer.optimize_constants(filled_state, is_positive_init=False)
    if valid_reward1 > valid_reward2:
        return expression1, train_reward1, valid_reward1, path
    else:
        return expression2, train_reward2, valid_reward2, path

def _evalpath_task(num_trials: int, optimizer, state: ExpTree, path: List[str]) -> Tuple[float, List[str]]:
    """Evaluate a given path."""
    cloned_state = copy.deepcopy(state)
    for op in path:
        cloned_state.add_op(op)
    expression1, train_reward1, valid_reward1 = optimizer.optimize_constants(cloned_state, is_positive_init=True)
    expression2, train_reward2, valid_reward2 = optimizer.optimize_constants(cloned_state, is_positive_init=False)
    if valid_reward1 > valid_reward2:
        return expression1, train_reward1, valid_reward1, path
    else:
        return expression2, train_reward2, valid_reward2, path

def _mutation_task(gp_manager, optimizer, old_path, state: ExpTree, seed: int = None) -> float:
    if seed is not None:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
    """Execute mutation operation."""
    try:
        new_path = gp_manager.mutate(state, old_path)
        expression, train_reward, valid_reward = _evalpath_task(optimizer, state, new_path)
    except:
        train_reward, valid_reward = 0, 0
        return None, train_reward, valid_reward, old_path
    return expression, train_reward, valid_reward, new_path

def _crossover_task(gp_manager, num_trials, optimizer, path1, path2, state: ExpTree, seed: int = None) -> float:
    if seed is not None:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
    """Execute crossover operation."""
    new_path1, new_path2 = gp_manager.crossover(state, path1, path2)
    try:
        expression1, train_reward1, valid_reward1 = _evalpath_task(num_trials, optimizer, state, new_path1)
        expression2, train_reward2, valid_reward2 = _evalpath_task(num_trials, optimizer, state, new_path2)
    except:
        return None, None, 0, 0, 0, 0, path1, path2
    return expression1, expression2, train_reward1, train_reward2, valid_reward1, valid_reward2, path1, path2
    
class MCTS:
    """Monte Carlo Tree Search implementation for symbolic regression."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        gp_manager: GPManager,
        gp_rate: float = 0.2,
        mutation_rate: float = 0.2,
        exploration_rate: float = 0.2,
        K: int = 500,
        c: float = 4,
        gamma: float = 0.5,
        verbose: bool = False,
        succ_error_tol: float = 1e-6,
        num_parallel: int = 4,
        num_batches: int = 16,
        num_trials: int = 1,
        seed: int = None,
    ) -> None:
        self.optimizer = optimizer
        self.gp_manager = gp_manager
        self.gp_rate = gp_rate
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.K = K
        self.c = c
        self.gamma = gamma
        self.verbose = verbose
        self.succ_error_tol = succ_error_tol
        
        # Initialize root node with reference to this MCTS instance
        self.root = MCTS_Node(mcts=self)
        self.exp_queue = Exp_Queue(max_size=50)
        self.path_queue = Exp_Queue(max_size=self.K)
        self.count_num: int = 0
        self.best_reward: float = -np.inf
        self.total_nodes: int = 1  # Root node
        self.num_parallel = num_parallel
        self.num_batches = num_batches
        self.num_trials = num_trials
        self.seed = seed

    def _task_seed(self, offset: int = 0) -> int:
        """Generate a deterministic seed for a task based on the global seed and an offset."""
        return (self.seed + offset) if self.seed is not None else None

    def mcts_select_node(self, exp_tree: ExpTree) -> Tuple[MCTS_Node, ExpTree]:
        """"""
        node = self.root
        self.count_num += 1
        node.visits += 1
        state = copy.deepcopy(exp_tree)
        
        # Selection phase
        while not node.is_leaf():
            if random.random() < self.exploration_rate:
                node = node.random_child()
            else:
                node = node.choose()
            node.visits += 1
            state.add_op(node.move)
        return node, state
    
    def mcts_expand_node(self, node: MCTS_Node, state: ExpTree) -> Tuple[MCTS_Node, ExpTree]:
        # Expansion phase
        if not state.is_terminal():
            node = node.expand(state)
            self.total_nodes += 1
            node.visits += 1
            state.add_op(node.move)
        return node, state
    
    def search(self, exp_tree: ExpTree) -> None:
        """Execute the MCTS search algorithm."""

        # Genetic Programming phase
        # Genetic operations are performed for leave nodes in the MCTS tree
        if random.random() < self.gp_rate:
            # Muation and crossover only occurs in the root node
            # root node has the best gnome expressions
            if random.random() < self.mutation_rate:
                # select leave nodes.
                nodes = deque()
                for i in range(self.num_batches):
                    path = self.path_queue.random_sample()[0]
                    state = copy.deepcopy(exp_tree)
                    nodes.append((self.root, path, state))
                nodes = list(nodes)

                with mp.Pool(processes=self.num_parallel) as pool:
                    results = pool.starmap(
                        _mutation_task,
                        [
                            (self.gp_manager, self.optimizer, path, state, self._task_seed(idx))
                            for idx, (node, path, state) in enumerate(
                                (node, path, state)
                                for node, path, state in nodes
                                for _ in range(self.num_trials)
                            )
                        ]
                        )
                results = self._select_best_trials(results)
                # update mcts tree
                for (node, _, _), (expression, train_reward, valid_reward, path) in zip(nodes, results):
                    if expression is not None:
                        self.exp_queue.append(expression, train_reward, valid_reward)
                        self.path_queue.append(path, train_reward, valid_reward)
                        self.best_reward = max(self.best_reward, valid_reward)
                        node.propagate(path, valid_reward)
            else:
                nodes = deque()
                for i in range(self.num_batches):
                    state = copy.deepcopy(exp_tree)
                    node = self.root
                    path1 = self.path_queue.random_sample()[0]
                    path2 = self.path_queue.random_sample()[0]
                    #print(path1, path2)
                    nodes.append((node, path1, path2, state))
                nodes = list(nodes)
                # compute crossover in parallel
                with mp.Pool(processes=self.num_parallel) as pool:
                    results = pool.starmap(
                        _crossover_task,
                        [
                            (self.gp_manager, self.num_trials, self.optimizer, path1, path2, state, self._task_seed(idx))
                            for idx, (node, path1, path2, state) in enumerate(
                                (node, path1, path2, state)
                                for node, path1, path2, state in nodes
                                for _ in range(self.num_trials)
                            )
                        ]
                            )
                results = self._select_best_trials_crossover(results)
                # update mcts tree
                for (node, _, _, _), (expression1, expression2, train_reward1, train_reward2, valid_reward1, valid_reward2, path1, path2) in zip(nodes, results):
                    if expression1 is not None:
                        self.exp_queue.append(expression1, train_reward1, valid_reward1)
                        self.path_queue.append(path1, train_reward1, valid_reward1)
                        self.exp_queue.append(expression2, train_reward2, valid_reward2)
                        self.path_queue.append(path2, train_reward2, valid_reward2)
                        self.best_reward = max(self.best_reward, valid_reward1)
                        self.best_reward = max(self.best_reward, valid_reward2)
                        node.propagate(path1, valid_reward1)
                        node.propagate(path2, valid_reward2)

        # MCTS simulation phase
        # This phase performs the core MCTS algorithm: Selection, Expansion, Simulation, and Backpropagation
        # to estimate the value of leaf nodes and update the tree statistics.
        nodes = deque()
        for i in range(self.num_batches):
            # Selection: Traverse the tree from root to a leaf node using UCB policy
            node, state = self.mcts_select_node(exp_tree)
            # Expansion: If the selected node is not terminal, expand it by adding a child node
            node, state = self.mcts_expand_node(node, state)
            # Collect nodes for simulation to minimize the influence of initialization
            nodes.append((node, state))
        nodes = list(nodes)

        # Simulation: Perform random fill to complete the expressions from the current state
        with mp.Pool(processes=self.num_parallel) as pool:
            results = pool.starmap(
                _rollout_task,
                [(state, self._task_seed(idx)) for idx, (node, state) in enumerate(nodes)]
            )

        # Sort the results by complexity (descending) to prioritize more complex expressions
        # complex expressions are optmizized first
        vals = np.array([complexity for _, _, complexity in results])
        idx = np.argsort(-vals).tolist()
        res = [results[i] for i in idx]
        nodes = [nodes[i] for i in idx]
        #print(res)
        # Prepare multiple trials with different initialization for rollout
        rollout_args = [
            (self.optimizer, state, path)
            for state, path, complexity in res
            for _ in range(self.num_trials)
        ]

        # Rollout: Optimize constants for each filled state to estimate reward
        with mp.Pool(processes=self.num_parallel) as pool:
            results = pool.starmap(
                _optimize_task,
                [(optimizer, state, path, self._task_seed(idx)) for idx, (optimizer, state, path) in enumerate(rollout_args)]
            )

        # Aggregate results: For each batch, select the best reward across multiple trials
        results = self._select_best_trials(results)

        # Backpropagation: Update the MCTS tree with simulation results, propagating rewards up the tree
        for (node, state), (expression, train_reward, valid_reward, path) in zip(nodes, results):
            self.exp_queue.append(expression, train_reward, valid_reward)
            self.best_reward = max(self.best_reward, valid_reward)
            if not path:
                node.is_terminal = True
            path = node.backpropagate(path, valid_reward)
            self.path_queue.append(path, train_reward, valid_reward)
            self._update_terminal_status(node)

        return self.best_reward
    
    def _select_best_trials(self, results: Tuple) -> Tuple[str, float, list]:
        best_results = deque()
        for ibatch in range(self.num_batches):
            best_valid_reward = -float('inf')
            save_train_reward = -float('inf')
            best_expr = None
            best_path = None
            for itrial in range(self.num_trials):
                idx = ibatch * self.num_trials + itrial
                expression, train_reward, valid_reward, path = results[idx]
                if valid_reward > best_valid_reward:
                    best_valid_reward = valid_reward
                    save_train_reward = train_reward
                    best_expr = expression
                    best_path = path
            best_results.append((best_expr, save_train_reward, best_valid_reward, best_path))
        results = list(best_results)
        return results

    def _select_best_trials_crossover(self, results: Tuple) -> Tuple[str, float, list]:
        best_results = deque()
        for ibatch in range(self.num_batches):
            save_train_reward1, save_train_reward2 = -float('inf'), -float('inf')
            best_valid_reward1, best_valid_reward2 = -float('inf'), -float('inf')
            best_expr1, best_expr2 = None, None
            best_path1, best_path2 = None, None
            for itrial in range(self.num_trials):
                idx = ibatch * self.num_trials + itrial
                expression1, expression2, train_reward1, train_reward2, valid_reward1, valid_reward2, path1, path2 = results[idx]
                if valid_reward1 > best_valid_reward1:
                    best_valid_reward1 = valid_reward1
                    save_train_reward1 = train_reward1
                    best_expr1 = expression1
                    best_path1 = path1
                if valid_reward2 > best_valid_reward2:
                    best_valid_reward2 = valid_reward2
                    save_train_reward2 = train_reward2
                    best_expr2 = expression2
                    best_path2 = path2

            best_results.append((best_expr1, best_expr2, save_train_reward1, save_train_reward2, best_valid_reward1, best_valid_reward2, best_path1, best_path2))
        results = list(best_results)
        return results

    def _update_terminal_status(self, node: MCTS_Node) -> None:
        """Recursively update terminal status of nodes."""
        current = node
        while current and current.parent:
            parent = current.parent
            # Check if all children of parent are terminal
            if (parent.children and 
                all(child.is_terminal for child in parent.children) and
                not parent.unexpanded_moves):
                if not parent.is_terminal:
                    parent.is_terminal = True
                    current = parent  # Continue checking upward
                else:
                    break  # Already terminal, no need to continue
            else:
                break  # Not all children are terminal, stop here
