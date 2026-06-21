from copy import deepcopy
from typing import Any, Dict


class ImmutableState:
    """
    Represents an immutable snapshot of the agent's workflow state.
    Updates are applied via JSON Merge Patch (RFC 7396), returning
    a new ImmutableState instance without mutating the original.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = deepcopy(data)

    def apply_patch(self, patch: Dict[str, Any]) -> "ImmutableState":
        """
        Applies a recursive JSON Merge Patch (RFC 7396) and returns
        a new ImmutableState instance containing the merged state.
        """
        new_data = deepcopy(self._data)
        self._merge(new_data, patch)
        return ImmutableState(new_data)

    def _merge(self, target: Dict[str, Any], patch: Dict[str, Any]) -> None:
        """
        In-place recursive merge patch logic.
        """
        for key, value in patch.items():
            if value is None:
                # Null values act as explicit deletions
                target.pop(key, None)
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                # Recursively merge nested dictionaries
                self._merge(target[key], value)
            else:
                # Replace primitives, lists, or mismatched structures entirely
                target[key] = deepcopy(value)

    def merge(self, other: "ImmutableState", policy: str = "last_write_wins") -> "ImmutableState":
        """
        Merges another state snapshot into this one.
        Supports conflict resolution policies: 'last_write_wins', 'combine_lists', 'raise'.
        """
        new_data = deepcopy(self._data)
        self._reconcile(new_data, other.to_dict(), policy)
        return ImmutableState(new_data)

    def _reconcile(self, target: Dict[str, Any], source: Dict[str, Any], policy: str) -> None:
        for key, val in source.items():
            if key not in target:
                target[key] = deepcopy(val)
            else:
                # Key exists in both. Reconcile based on policy
                target_val = target[key]
                if isinstance(target_val, dict) and isinstance(val, dict):
                    self._reconcile(target_val, val, policy)
                elif (
                    isinstance(target_val, list)
                    and isinstance(val, list)
                    and policy == "combine_lists"
                ):
                    target[key] = target_val + deepcopy(val)
                elif target_val == val:
                    # Values are identical, no conflict
                    pass
                else:
                    # Conflict detected!
                    if policy == "raise":
                        raise ValueError(
                            f"Conflict detected at key '{key}': target value '{target_val}' "
                            f"does not match source value '{val}'."
                        )
                    elif policy == "combine_lists":
                        # For non-list conflicts under combine_lists, fall back to last_write_wins
                        target[key] = deepcopy(val)
                    else:  # last_write_wins
                        target[key] = deepcopy(val)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the state data.
        """
        return deepcopy(self._data)
