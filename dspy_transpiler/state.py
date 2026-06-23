from copy import deepcopy
from typing import Any, Dict


class ImmutableState:
    """
    Represents an immutable snapshot of the agent's workflow state.
    Updates are applied via JSON Merge Patch (RFC 7396), returning
    a new ImmutableState instance without mutating the original.
    """

    def __init__(self, data: Dict[str, Any], _skip_copy: bool = False):
        if _skip_copy:
            self._data = data
        else:
            self._data = deepcopy(data)

    def apply_patch(self, patch: Dict[str, Any]) -> "ImmutableState":
        """
        Applies a recursive JSON Merge Patch (RFC 7396) and returns
        a new ImmutableState instance containing the merged state using COW optimization.
        """
        new_data = self._merge_cow(self._data, patch)
        return ImmutableState(new_data, _skip_copy=True)

    def _merge_cow(self, target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy-on-write merge routine. Reuses reference pointers for unchanged keys.
        """
        new_data = target.copy()
        for key, value in patch.items():
            if value is None:
                new_data.pop(key, None)
            elif isinstance(value, dict) and isinstance(new_data.get(key), dict):
                new_data[key] = self._merge_cow(new_data[key], value)
            else:
                if isinstance(value, (dict, list)):
                    new_data[key] = deepcopy(value)
                else:
                    new_data[key] = value
        return new_data

    def merge(self, other: "ImmutableState", policy: str = "last_write_wins") -> "ImmutableState":
        """
        Merges another state snapshot into this one.
        Supports conflict resolution policies: 'last_write_wins', 'combine_lists', 'raise'.
        """
        new_data = self._reconcile_cow(self._data, other.to_dict(), policy)
        return ImmutableState(new_data, _skip_copy=True)

    def _reconcile_cow(
        self, target: Dict[str, Any], source: Dict[str, Any], policy: str
    ) -> Dict[str, Any]:
        new_data = target.copy()
        for key, val in source.items():
            if key not in new_data:
                if isinstance(val, (dict, list)):
                    new_data[key] = deepcopy(val)
                else:
                    new_data[key] = val
            else:
                target_val = new_data[key]
                if isinstance(target_val, dict) and isinstance(val, dict):
                    new_data[key] = self._reconcile_cow(target_val, val, policy)
                elif (
                    isinstance(target_val, list)
                    and isinstance(val, list)
                    and policy == "combine_lists"
                ):
                    new_data[key] = target_val + deepcopy(val)
                elif target_val == val:
                    pass
                else:
                    if policy == "raise":
                        raise ValueError(
                            f"Conflict detected at key '{key}': target value '{target_val}' "
                            f"does not match source value '{val}'."
                        )
                    elif policy == "combine_lists":
                        if isinstance(val, (dict, list)):
                            new_data[key] = deepcopy(val)
                        else:
                            new_data[key] = val
                    else:
                        if isinstance(val, (dict, list)):
                            new_data[key] = deepcopy(val)
                        else:
                            new_data[key] = val
        return new_data

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the state data to protect mutability boundaries.
        """
        return deepcopy(self._data)
