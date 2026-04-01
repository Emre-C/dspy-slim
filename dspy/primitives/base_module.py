import copy
import logging
from collections import deque
from collections.abc import Generator
from pathlib import Path

import orjson

from dspy.utils.saving import get_dependency_versions

# NOTE: Note: It's important (temporary decision) to maintain named_parameters that's different in behavior from
# named_sub_modules for the time being.


logger = logging.getLogger(__name__)


class BaseModule:
    def __init__(self):
        pass

    def named_parameters(self):
        """
        Unlike PyTorch, handles (non-recursive) lists of parameters too.
        """

        import dspy
        from dspy.predict.parameter import Parameter

        visited = set()
        named_parameters = []

        def add_parameter(param_name, param_value):
            if isinstance(param_value, Parameter):
                if id(param_value) not in visited:
                    visited.add(id(param_value))
                    named_parameters.append((param_name, param_value))

            elif isinstance(param_value, dspy.Module):
                # When a sub-module is pre-compiled, keep it frozen.
                if not getattr(param_value, "_compiled", False):
                    for sub_name, param in param_value.named_parameters():
                        add_parameter(f"{param_name}.{sub_name}", param)

        if isinstance(self, Parameter):
            add_parameter("self", self)

        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                add_parameter(name, value)

            elif isinstance(value, dspy.Module):
                # When a sub-module is pre-compiled, keep it frozen.
                if not getattr(value, "_compiled", False):
                    for sub_name, param in value.named_parameters():
                        add_parameter(f"{name}.{sub_name}", param)

            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    add_parameter(f"{name}[{idx}]", item)

            elif isinstance(value, dict):
                for key, item in value.items():
                    add_parameter(f"{name}['{key}']", item)

        return named_parameters

    def named_sub_modules(self, type_=None, skip_compiled=False) -> Generator[tuple[str, "BaseModule"], None, None]:
        """Find all sub-modules in the module, as well as their names.

        Say `self.children[4]['key'].sub_module` is a sub-module. Then the name will be
        `children[4]['key'].sub_module`. But if the sub-module is accessible at different
        paths, only one of the paths will be returned.
        """
        if type_ is None:
            type_ = BaseModule

        queue = deque([("self", self)])
        seen = {id(self)}

        def add_to_queue(name, item):
            if id(item) not in seen:
                seen.add(id(item))
                queue.append((name, item))

        while queue:
            name, item = queue.popleft()

            if isinstance(item, type_):
                yield name, item

            if isinstance(item, BaseModule):
                if skip_compiled and getattr(item, "_compiled", False):
                    continue
                for sub_name, sub_item in item.__dict__.items():
                    add_to_queue(f"{name}.{sub_name}", sub_item)

            elif isinstance(item, (list, tuple)):
                for i, sub_item in enumerate(item):
                    add_to_queue(f"{name}[{i}]", sub_item)

            elif isinstance(item, dict):
                for key, sub_item in item.items():
                    add_to_queue(f"{name}[{key}]", sub_item)

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def deepcopy(self):
        """Deep copy the module.

        This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
        attributes, we just do the shallow copy.
        """
        try:
            # If the instance itself is copyable, we can just deep copy it.
            # Otherwise we will have to create a new instance and copy over the attributes one by one.
            return copy.deepcopy(self)
        except Exception:
            pass

        # Create an empty instance.
        new_instance = self.__class__.__new__(self.__class__)
        # Set attribuetes of the copied instance.
        for attr, value in self.__dict__.items():
            if isinstance(value, BaseModule):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    # Try to deep copy the attribute
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    logging.warning(
                        f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, "
                        "falling back to shallow copy or reference copy."
                    )
                    try:
                        # Fallback to shallow copy if deep copy fails
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        # If even the shallow copy fails, we just copy over the reference.
                        setattr(new_instance, attr, value)

        return new_instance

    def reset_copy(self):
        """Deep copy the module and reset all parameters."""
        new_instance = self.deepcopy()

        for param in new_instance.parameters():
            param.reset()

        return new_instance

    def dump_state(self, json_mode=True):
        return {name: param.dump_state(json_mode=json_mode) for name, param in self.named_parameters()}

    def load_state(self, state, *, allow_unsafe_lm_state=False):
        from dspy.predict.predict import Predict

        for name, param in self.named_parameters():
            if isinstance(param, Predict):
                param.load_state(state[name], allow_unsafe_lm_state=allow_unsafe_lm_state)
            else:
                param.load_state(state[name])

    def save(self, path):
        """Save module state to a ``.json`` file (architecture is not preserved; reconstruct the program in code)."""
        metadata: dict = {"dependency_versions": get_dependency_versions()}
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError(f"`path` must end with `.json`, got: {path}")
        state = self.dump_state()
        state["metadata"] = metadata
        try:
            with open(path, "wb") as f:
                f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
        except Exception as e:
            raise RuntimeError(
                f"Failed to save state to {path} with error: {e}. Your program may contain non-json-serializable objects."
            ) from e

    def load(self, path, *, allow_unsafe_lm_state=False):
        """Load module state from a ``.json`` file written by :meth:`save`."""
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError(f"`path` must end with `.json`, got: {path}")
        with open(path, "rb") as f:
            state = orjson.loads(f.read())

        dependency_versions = get_dependency_versions()
        saved_dependency_versions = state["metadata"]["dependency_versions"]
        for key, saved_version in saved_dependency_versions.items():
            if dependency_versions[key] != saved_version:
                logger.warning(
                    f"There is a mismatch of {key} version between saved model and current environment. "
                    f"You saved with `{key}=={saved_version}`, but now you have "
                    f"`{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade "
                    "on the loaded model, please consider loading the model in the same environment as the "
                    "saving environment."
                )
        self.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
