"""Definition-time shapes and decision-scoped native forecast caches.

Serialized keys describe values, never extension behavior. Non-native decisions
retain isolated actual models and bypass shared caches, including nested forecasts.
"""

import inspect
import json
from contextvars import ContextVar
from functools import lru_cache, wraps

from pydantic import BaseModel

_MISSING = object()
_SHAPES = {}
_FIELDS = {}
_HOOKS = {}
_VALUES = []
_CLASS_DICT = type.__dict__["__dict__"]
_CLASS_MRO = type.__dict__["__mro__"]
_HELPERS = []
_COPY_APIS = {name: inspect.getattr_static(BaseModel, name)
              for name in ("model_copy", "__copy__", "__deepcopy__")}
# Eligibility, actual-model references, local results and defining-class verdicts
# all live only until the outermost decision exits.
_STATE = ContextVar("native_forecast_state", default=None)


def register_native_model(owner):
    """Called at defining-module completion, before consumers can patch it."""
    _FIELDS[owner] = dict(owner.model_fields)
    names = set(owner.model_fields) | {"__getattribute__", "__getattr__", "__dict__",
                                       "__pydantic_extra__", "__pydantic_private__",
                                       "__pydantic_fields_set__", *_COPY_APIS}
    names.update(name for name, value in vars(owner).items()
                 if not name.startswith("_") and name != "evolve" and
                 (callable(value) or isinstance(value, (property, classmethod, staticmethod))))
    _SHAPES[owner] = {name: inspect.getattr_static(owner, name, _MISSING) for name in names}
    _SHAPES[owner].update(_COPY_APIS)
    _HOOKS[owner] = frozenset(names - owner.model_fields.keys())


def _class_dictionaries(owner):
    # Invoke type's own descriptors directly. Neither model properties nor a
    # replacement metaclass getter executes during static eligibility checks.
    return tuple(_CLASS_DICT.__get__(base) for base in _CLASS_MRO.__get__(owner))


def _static_member(dictionaries, name):
    for values in dictionaries:
        if name in values:
            return values[name]
    return _MISSING


def _inspect_native_model_class(owner):
    shape = _SHAPES.get(owner)
    if shape is None:
        return False
    dictionaries = _class_dictionaries(owner)
    fields = _static_member(dictionaries, "__pydantic_fields__")
    if (type(fields) is not dict or fields.keys() != _FIELDS[owner].keys()
            or any(fields[name] is not field for name, field in _FIELDS[owner].items())):
        return False
    # getattr_static's class lookup searches the class MRO before its metaclass.
    # Fetch the dictionaries once rather than rewalking them for every field.
    dictionaries += _class_dictionaries(type(owner))
    return all(_static_member(dictionaries, name) is value for name, value in shape.items())


def native_model_class(owner):
    state = _STATE.get()
    return state[3].get(owner, False) if state is not None else _inspect_native_model_class(owner)


def _native_model_instance(model, classes):
    owner = type(model)
    if owner not in classes:
        classes[owner] = _inspect_native_model_class(owner)
    if not classes[owner]:
        return False
    values = object.__getattribute__(model, "__dict__")
    extra = object.__getattribute__(model, "__pydantic_extra__") or {}
    hooks = _HOOKS[owner]
    if type(values) is dict and type(extra) is dict:
        if not hooks.isdisjoint(values) or not hooks.isdisjoint(extra):
            return False
    elif any(name in values or name in extra for name in hooks):
        # Custom mappings can define membership independently of their keys.
        return False
    for name in ("sectors", "active_aero_zones"):
        for item in values.get(name, ()):
            if not _native_model_instance(item, classes):
                return False
    return True


def native_model(model):
    state = _STATE.get()
    return _native_model_instance(model, {} if state is None else state[3])


def register_forecast_helpers(namespace, names):
    """Keep defining-module originals for helpers used inside serialized caches."""
    _HELPERS.extend((namespace, name, namespace[name]) for name in names)


def register_forecast_values(namespace, names):
    """Record immutable calibration/configuration values omitted from cache keys."""
    _VALUES.extend((namespace, name, namespace[name]) for name in names)


def _same_native_value(value, original):
    if type(value) is not type(original):
        return False
    if isinstance(original, tuple):
        return len(value) == len(original) and all(
            _same_native_value(item, native) for item, native in zip(value, original)
        )
    return value == original


def _physics_snapshot(models):
    from f1sim.models.tire import TIRE_COMPOUNDS
    from f1sim.simulation import lap

    classes = {owner: _inspect_native_model_class(owner) for owner in _SHAPES}
    native = (lap.native_lap_physics()
              and all(namespace[name] is original for namespace, name, original in _HELPERS)
              and all(_same_native_value(namespace[name], original)
                      for namespace, name, original in _VALUES)
              and all(classes.values())
              and all(_native_model_instance(model, classes)
                      for model in (*models, *TIRE_COMPOUNDS.values())))
    return native, classes


def native_physics(*models):
    state = _STATE.get()
    if state is not None:
        return state[0] and all(_native_model_instance(model, state[3]) for model in models)
    return _physics_snapshot(models)[0]


def shared_forecast_available():
    state = _STATE.get()
    return state[0] if state is not None else native_physics()


def _key(owner, values):
    return owner, json.dumps(values, sort_keys=True)


def _schema_values(value):
    if isinstance(value, BaseModel):
        owner = next((cls for cls in _SHAPES if isinstance(value, cls)), type(value))
        values = object.__getattribute__(value, "__dict__")
        return {name: _schema_values(values[name]) for name in owner.model_fields if name in values}
    if isinstance(value, (list, tuple)):
        return [_schema_values(item) for item in value]
    if isinstance(value, dict):
        return {name: _schema_values(item) for name, item in value.items()}
    return value


def forecast_dump(model):
    state = _STATE.get()
    if state is None or state[0] or native_model(model):
        return model.model_dump(mode="json")
    owner = next((cls for cls in _SHAPES if isinstance(model, cls)), type(model))
    # This private identity only lives inside one decision. Equal schema values
    # need not mean equal extension dispatch; callable instance hooks need never
    # be serialized. Keep a strong reference until the context is reset.
    values = _schema_values(model)
    values["__forecast_identity__"] = id(model)
    state[1][_key(owner, values)] = model
    return values


def forecast_json(model):
    if shared_forecast_available():
        return model.model_dump_json()
    return json.dumps(forecast_dump(model))


def restore_model(owner, values):
    """Recover an actual extension model on uncached paths, with isolated state."""
    if isinstance(values, str):
        values = json.loads(values)
    state = _STATE.get()
    if state is not None and not state[0]:
        original = state[1].get(_key(owner, values))
        if original is not None:
            return original.model_copy(deep=True)
    # Car permits validated callers to set deterministic service std to zero.
    values.pop("__forecast_identity__", None)
    if owner.__name__ == "Car":
        return owner.model_construct(**values)
    return owner.model_validate(values)


def forecast_decision(function):
    """Check native eligibility once at the public decision boundary."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        from f1sim.models.tire import TIRE_COMPOUNDS

        models = [value for value in (*args, *kwargs.values()) if isinstance(value, BaseModel)
                  and any(isinstance(value, cls) for cls in _SHAPES)]
        previous = _STATE.get()
        if previous is None:
            native, classes = _physics_snapshot(models)
        else:
            classes = previous[3]
            native = previous[0] and all(_native_model_instance(model, classes) for model in models)
        state = (native, {} if previous is None else previous[1],
                 {} if previous is None else previous[2], classes)
        token = _STATE.set(state)
        try:
            if not native:
                def isolate(value):
                    if any(value is model for model in models):
                        return value.model_copy(deep=True)
                    return value

                args = tuple(isolate(value) for value in args)
                kwargs = {name: isolate(value) for name, value in kwargs.items()}
                models = [value for value in (*args, *kwargs.values())
                          if isinstance(value, BaseModel)
                          and any(isinstance(value, cls) for cls in _SHAPES)]
                for model in (*models, *TIRE_COMPOUNDS.values()):
                    forecast_dump(model)
            return function(*args, **kwargs)
        finally:
            _STATE.reset(token)
    return wrapped


def native_forecast_cache(maxsize):
    """Keep bounded native hits; behavioral extensions never share results."""
    def decorate(function):
        cached = lru_cache(maxsize=maxsize)(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            if shared_forecast_available():
                return cached(*args, **kwargs)
            state = _STATE.get()
            if state is None:
                return function(*args, **kwargs)
            # Shared caches must opt out, but a deterministic decision still
            # needs local memoization for its recursive suffix search.
            memo = state[2].setdefault(function, {})
            key = (args, tuple(sorted(kwargs.items())))
            if key not in memo:
                memo[key] = function(*args, **kwargs)
            return memo[key]

        wrapped.cache_info = cached.cache_info
        wrapped.cache_clear = cached.cache_clear
        wrapped.__wrapped__ = function
        return wrapped
    return decorate
