from typing import Tuple
from matbench.bench import MatbenchBenchmark
import numpy as np
from pymatgen.core import Composition
from ase.data import atomic_numbers
from tqdm import tqdm

def form2comp(formula: str) -> np.ndarray:
    """
    Convert a chemical formula string into a normalized 118-dimensional composition vector.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. "Ag0.5Ge1Pb1.75S4", "CaCO3", "Fe2O3".

    Returns
    -------
    vec : np.ndarray, shape (118,)
        Normalized composition vector ordered by atomic number (H=1 ... Og=118).
        Values are atomic fractions that sum to 1.
    """
    # Parse formula robustly (supports fractional stoichiometry)
    comp = Composition(formula)
    el_amt = comp.get_el_amt_dict()

    vec = np.zeros(118, dtype=np.float64)

    for element, amount in el_amt.items():
        atomic_number = atomic_numbers[element] - 1  # zero-based index
        vec[atomic_number] = amount

    # Normalize to atomic fractions
    vec = vec / vec.sum()
    return vec

def load_matbench(task_key: str, fold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from a Matbench task for a specific fold.

    Parameters
    ----------
    task_key : str
        The Matbench task identifier (e.g., 'matbench_mp_gap').
    fold : int, optional
        The fold number for cross-validation (default is 0).

    Returns
    -------
    compositions : np.ndarray, shape (n_samples, 118)
        Normalized composition vectors for each sample.
    targets : np.ndarray, shape (n_samples,)
        Target values for the task.
    """
    # Valid task keys for Matbench
    valid_tasks = [
        'matbench_dielectric',
        'matbench_expt_gap',
        'matbench_expt_is_metal',
        'matbench_glass',
        'matbench_jdft2d',
        'matbench_log_gvrh',
        'matbench_log_kvrh',
        'matbench_mp_e_form',
        'matbench_mp_gap',
        'matbench_mp_is_metal',
        'matbench_perovskites',
        'matbench_phonons',
        'matbench_steels'
    ]

    if task_key not in valid_tasks:
        raise ValueError(f"Invalid task_key '{task_key}'. Must be one of {valid_tasks}")

    # Load Matbench benchmark
    mb = MatbenchBenchmark(autoload=False)
    task = getattr(mb, task_key)
    task.load()

    # Get training and validation data for the fold
    train_inputs, train_outputs = task.get_train_and_val_data(fold)

    num_samples = len(train_outputs)
    num_elements = 118
    compositions = np.zeros((num_samples, num_elements))

    # Process each input entry to extract composition
    for i, entry in tqdm(enumerate(train_inputs), total=num_samples):
        if hasattr(entry, 'to_ase_atoms'):
            # Entry is a structure object, extract composition from ASE atoms
            atoms = entry.to_ase_atoms()
            atomic_nums = atoms.get_atomic_numbers()
            # Create composition vector by counting atomic fractions
            composition_vector = np.sum(np.eye(num_elements)[atomic_nums - 1], axis=0) / len(atomic_nums)
        else:
            # Entry is a formula string, parse it directly
            composition_vector = form2comp(entry)

        compositions[i] = composition_vector

    targets = np.array(train_outputs)
    #print(targets)
    return compositions, targets

def load_matbench_test(task_key: str, fold: int = 0, include_target: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from a Matbench task for a specific fold.

    Parameters
    ----------
    task_key : str
        The Matbench task identifier (e.g., 'matbench_mp_gap').
    fold : int, optional
        The fold number for cross-validation (default is 0).

    Returns
    -------
    compositions : np.ndarray, shape (n_samples, 118)
        Normalized composition vectors for each sample.
    targets : np.ndarray, shape (n_samples,)
        Target values for the task.
    """
    # Valid task keys for Matbench
    valid_tasks = [
        'matbench_dielectric',
        'matbench_expt_gap',
        'matbench_expt_is_metal',
        'matbench_glass',
        'matbench_jdft2d',
        'matbench_log_gvrh',
        'matbench_log_kvrh',
        'matbench_mp_e_form',
        'matbench_mp_gap',
        'matbench_mp_is_metal',
        'matbench_perovskites',
        'matbench_phonons',
        'matbench_steels'
    ]

    if task_key not in valid_tasks:
        raise ValueError(f"Invalid task_key '{task_key}'. Must be one of {valid_tasks}")

    # Load Matbench benchmark
    mb = MatbenchBenchmark(autoload=False)
    task = getattr(mb, task_key)
    task.load()

    # Get training and validation data for the fold
    #train_inputs, train_outputs = task.get_train_and_val_data(fold)
    if include_target:
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    else:
        test_inputs = task.get_test_data(fold, include_target=False)
        test_outputs = None

    num_samples = test_inputs.shape[0]
    num_elements = 118
    compositions = np.zeros((num_samples, num_elements))

    # Process each input entry to extract composition
    for i, entry in tqdm(enumerate(test_inputs), total=num_samples):
        if hasattr(entry, 'to_ase_atoms'):
            # Entry is a structure object, extract composition from ASE atoms
            atoms = entry.to_ase_atoms()
            atomic_nums = atoms.get_atomic_numbers()
            # Create composition vector by counting atomic fractions
            composition_vector = np.sum(np.eye(num_elements)[atomic_nums - 1], axis=0) / len(atomic_nums)
        else:
            # Entry is a formula string, parse it directly
            composition_vector = form2comp(entry)

        compositions[i] = composition_vector
    targets = np.array(test_outputs)

    return compositions, targets

def test_load_matbench_invalid_task():
    """Test that load_matbench raises ValueError for invalid task keys."""
    try:
        load_matbench('invalid_task_key')
        assert False, "Expected ValueError for invalid task key"
    except ValueError as e:
        assert "Invalid task_key" in str(e), f"Unexpected error message: {e}"
        print("Invalid task key test passed")


def test_load_matbench_all_tasks():
    """Test load_matbench with all valid task keys (may require data download)."""
    valid_tasks = [
        'matbench_dielectric',
        'matbench_expt_gap',
        'matbench_expt_is_metal',
        'matbench_glass',
        'matbench_jdft2d',
        'matbench_log_gvrh',
        'matbench_log_kvrh',
        'matbench_mp_e_form',
        'matbench_mp_gap',
        'matbench_mp_is_metal',
        'matbench_perovskites',
        'matbench_phonons',
        'matbench_steels'
    ]

    for task_key in valid_tasks:
        try:
            compositions, targets = load_matbench(task_key, fold=0)
            assert isinstance(compositions, np.ndarray), f"Compositions should be ndarray for {task_key}"
            assert isinstance(targets, np.ndarray), f"Targets should be ndarray for {task_key}"
            assert compositions.shape[1] == 118, f"Compositions should have 118 columns for {task_key}"
            assert len(compositions) == len(targets), f"Lengths should match for {task_key}"
            print(f"Test passed for {task_key}: {len(compositions)} samples")
        except Exception as e:
            print(f"Test failed for {task_key}: {e}")
            # Continue testing other tasks


if __name__ == "__main__":
    test_load_matbench_invalid_task()
    test_load_matbench_all_tasks()
    print("All tests completed!")
