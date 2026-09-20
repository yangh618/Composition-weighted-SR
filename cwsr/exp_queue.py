from sortedcontainers import SortedList
from typing import Any, Tuple, Iterable
from abc import ABC, abstractmethod
import math
import random


#: Default weight of the validation reward in the combined search score.
#: ``1.0`` reproduces the original CWSR behaviour (candidates are ranked by
#: ``valid_reward`` alone); ``0.0`` ranks by the training reward alone.
DEFAULT_VALID_REWARD_WEIGHT: float = 1.0


def combine_rewards(train_reward: float,
                    valid_reward: float,
                    valid_weight: float = DEFAULT_VALID_REWARD_WEIGHT) -> float:
    """Mixed search score ``valid_weight * valid_reward + (1 - valid_weight) * train_reward``.

    The search optimises a *single* scalar, and CWSR's original choice was
    ``valid_weight = 1`` (rank purely by the validation reward). A smaller weight
    blends in the training reward, which stabilises selection when the validation
    split is small or noisy. ``valid_weight`` is clipped to ``[0, 1]``.
    """
    weight = max(0.0, min(1.0, float(valid_weight)))
    return weight * float(valid_reward) + (1.0 - weight) * float(train_reward)


class Queue_Base(ABC):
    """Abstract base for prioritized experience / expression queues.

    Stored elements are tuples of ``(state, train_reward, valid_reward)``. The
    internal SortedList is maintained in descending order of the *combined* score
    ``combine_rewards(train_reward, valid_reward, valid_weight)`` — by default
    (``valid_weight=1.0``) that is exactly the old "sort by valid_reward"
    behaviour, while ``valid_weight < 1`` blends in the training reward. Both raw
    rewards are kept in every entry, so results/checkpoints are unaffected.
    """

    def __init__(self, max_size: int,
                 valid_weight: float = DEFAULT_VALID_REWARD_WEIGHT):
        self.max_size = max_size
        self.valid_weight = max(0.0, min(1.0, float(valid_weight)))
        # Main container: sorted descending by the combined score (via negative key).
        # ``valid_weight`` is fixed at construction time (the sort key depends on it).
        self.list: SortedList[Tuple[Any, float, float]] = SortedList(
            key=lambda x: -combine_rewards(x[1], x[2], self.valid_weight))
        # Maintain a secondary sorted structure of scores (ascending) for O(log n) near-duplicate checks
        self._reward_values: SortedList[float] = SortedList()
        self.min_reward: float = float('-inf')  # Cached minimum score in queue (last element)

    def score(self, train_reward: float, valid_reward: float) -> float:
        """Combined score of one ``(train_reward, valid_reward)`` pair."""
        return combine_rewards(train_reward, valid_reward, self.valid_weight)

    def score_of(self, entry: Tuple[Any, float, float]) -> float:
        """Combined score of a stored ``(state, train_reward, valid_reward)`` entry."""
        return self.score(entry[1], entry[2])


    @abstractmethod
    def append(self, state: Any, reward: float) -> bool:
        pass

    def __len__(self) -> int:  # Convenience
        return len(self.list)

    def __iter__(self) -> Iterable[Tuple[Any, float]]:
        return iter(self.list)

    # Best reward value in the queue
    def best_reward(self) -> float:
        if not self.list:
            return 0.0, 0.0
        return self.list[0][1], self.list[0][2]
    
    def best(self) -> Any:
        if not self.list:
            return None, None
        return self.list[0]

    def random_sample(self) -> Any:
        if not self.list:
            return None, None
        return random.choice(self.list)

    def is_empty(self) -> bool:
        return not self.list

class Exp_Queue(Queue_Base):
    """Experience / expression priority queue with fast approximate duplicate suppression.

    Performance improvements over previous version:
    - Near-duplicate reward detection reduced from O(n) scan to O(log n) by using a
      secondary SortedList of reward values and only inspecting neighbors.
    - Fewer attribute lookups inside hot append path via local bindings.
    - Early exits for invalid rewards (inf / nan).
    """

    def append(self, state: Any, train_reward: float, valid_reward: float = 0.0, threshold: float = 1e-5) -> bool:
        """Attempt to insert (state, train_reward, valid_reward).

        Ordering and duplicate suppression use the *combined* score
        ``combine_rewards(train_reward, valid_reward, self.valid_weight)`` (the
        validation reward alone when ``valid_weight == 1``, the default). Near-
        duplicate scores (|Δ| < threshold) are rejected to avoid bloating the
        queue with numerically indistinguishable entries.
        Returns True if inserted, False otherwise.
        """
        score = self.score(train_reward, valid_reward)
        # Reject invalid numeric cases early (math.isinf faster; also guard nan)
        if math.isinf(score) or math.isnan(score):
            return False

        lst = self.list
        if (len(lst) >= self.max_size) and (score <= self.min_reward):
            return False  # Fast reject if not better than current minimum

        reward_values = self._reward_values  # local binding
        # Binary search position (ascending order). Only immediate neighbors can be within threshold.
        pos = reward_values.bisect_left(score)
        if pos > 0 and abs(reward_values[pos - 1] - score) < threshold:
            return False
        if pos < len(reward_values) and abs(reward_values[pos] - score) < threshold:
            return False

        lst = self.list
        if len(lst) < self.max_size:
            lst.add((state, train_reward, valid_reward))
            reward_values.add(score)
        else:
            # Fast check: if not better than current minimum, discard.
            if score <= self.min_reward:
                return False
            # Remove worst (last element due to descending order via key)
            removed_entry = lst.pop(-1)
            # Remove its combined score from the secondary structure
            reward_values.remove(self.score_of(removed_entry))
            # Insert new element
            lst.add((state, train_reward, valid_reward))
            reward_values.add(score)

        # Update cached minimum score (worst at index -1)
        self.min_reward = self.score_of(lst[-1]) if lst else float('-inf')
        return True
