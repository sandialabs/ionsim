"""Define the ionsim_api decorator that converts user inputs from one
of several possible formats to pint Quantity values."""

import functools
import inspect
import warnings
import typing
import numbers
from typing import Any, Union
from collections.abc import Mapping, Set, Sequence

# This changed in Python 3.10
try:
    from typing import UnionType
except ImportError:
    from types import UnionType

import pint
import numpy as np

from ionsim import ureg


class Unit:
    """Used as an annotation, Unit[D, U] means "take whatever the user
    passed in ('5 m', (5,'m'), or a Quantity) and convert it
    into U."  The D argument is pure documentation, ignored at
    runtime."""
    def __init__(self, dtype: Any, target_unit: str):
        self.dtype = dtype
        self.target_unit = target_unit
        self.target_unit_as_quantity = ureg.Unit(target_unit)

    def __class_getitem__(cls, params):
        if not (isinstance(params, tuple) and len(params) == 2):
            raise TypeError("Unit[...] must be subscripted as Unit[<type>, <units>]")
        dtype, units = params
        return cls(dtype, units)

    def __repr__(self):
        return f"Unit[{self.dtype!r}, '{self.target_unit}']"

    def __or__(self, other):
        return Union[self, other]


def ionsim_api(obj):
    if inspect.isclass(obj):
        return _ionsim_api_class(obj)
    else:
        return _ionsim_api_function(obj)


def _ionsim_api_class(cls):
    for name, attr in list(vars(cls).items()):
        if isinstance(attr, staticmethod):
            fn = attr.__func__
            decorated = _ionsim_api_function(fn)
            setattr(cls, name, staticmethod(decorated))

        elif isinstance(attr, classmethod):
            fn = attr.__func__
            decorated = _ionsim_api_function(fn)
            setattr(cls, name, classmethod(decorated))

        elif inspect.isfunction(attr):
            decorated = _ionsim_api_function(attr)
            setattr(cls, name, decorated)
    return cls


def _ionsim_api_function(func):
    sig = inspect.signature(func)

    if sig.return_annotation == inspect.Signature.empty:
        raise ValueError(f"Function {func} must have a return annotation")

    # If none of the annotations on the function parameters are units,
    # return the original function unchanged. Also check that the Unit
    # annotation is not used incorrectly.
    for name, param in sig.parameters.items():
        if name == 'return':
            continue
        if param.annotation == Unit:
            raise TypeError(f"Annotation uses bare Unit. Please provide arguments, e.g. Unit[float, 'm']")
        if _has_unit_type(param.annotation):
            break
    else:
        return func

    @functools.wraps(func)
    def inner(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        for name, value in ba.arguments.items():
            annotation = sig.parameters[name].annotation
            if annotation is not None:
                ba.arguments[name], _ = _convert_to_annotation(value, annotation)
        ret = func(*ba.args, **ba.kwargs)
        conv, _ = _convert_to_annotation(ret, sig.return_annotation, is_return=True)
        return conv

    return inner


def _convert_to_annotation(value, annotation, is_return=False):
    """Convert a value to properly have units according to an
    annotation."""

    if isinstance(annotation, Unit):
        return _convert_unit(value, annotation, is_return=is_return)

    origin = typing.get_origin(annotation)

    if origin is Union or origin is UnionType:
        return _convert_union(value, annotation, is_return=is_return)

    if isinstance(origin, type):
        if issubclass(origin, Mapping):
            return _convert_mapping(value, annotation, is_return=is_return)
        elif issubclass(origin, Set):
            return _convert_set(value, annotation, is_return=is_return)
        elif issubclass(origin, tuple):
            return _convert_tuple(value, annotation, is_return=is_return)
        elif issubclass(origin, Sequence):
            return _convert_sequence(value, annotation, is_return=is_return)

    # Ignore all other annotations
    return value, False


def _convert_unit(value, annotation, is_return=False):
    """Convert a value give by the user to an appropriate pint unit."""
    if isinstance(value, ureg.Quantity):
        ret = value
    elif isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Value with units given as a tuple must have two elements, found {value}")
        ret = ureg.Quantity(*value)
    elif isinstance(value, str):
        ret = ureg.Quantity(value)
    elif isinstance(value, pint.Quantity):
        if value.units == annotation.target_unit_as_quantity:
            return value, False
        ret = value
    elif is_return and isinstance(value, (numbers.Number, np.ndarray)):
        ret = ureg.Quantity(value, annotation.target_unit_as_quantity)
    else:
        raise ValueError(f"Value not understood or does not have units as required: {value}")

    conv = ret.to(annotation.target_unit_as_quantity)
    if is_return:
        return conv, True
    return conv.magnitude, True


def _convert_mapping(value, annotation, is_return=False):
    was_converted = False

    def convert(arg, annotation):
        nonlocal was_converted
        conv, ok = _convert_to_annotation(arg, annotation, is_return=is_return)
        was_converted = was_converted or ok
        return conv

    key_annotation = typing.get_args(annotation)[0]
    value_annotation = typing.get_args(annotation)[1]

    itr = ((convert(k, key_annotation), convert(v, value_annotation)) for k, v in value.items())
    if hasattr(value, 'default_factory'):
        # Likely a defaultdict; make sure to reuse the function
        # creating default values.
        new = type(value)(value.default_factory, itr)
    else:
        new = type(value)(itr)

    if was_converted:
        return new, True

    return value, False


def _convert_set(value, annotation, is_return=False):
    return _convert_sequence(value, annotation, is_return=is_return)


def _convert_tuple(value, annotation, is_return=False):
    if len(value) != len(typing.get_args(annotation)):
        raise ValueError(f"Tuple {value} expected to have {len(typing.get_args(annotation))} values")

    was_converted = False

    def convert(arg, annotation):
        nonlocal was_converted
        conv, ok = _convert_to_annotation(arg, annotation, is_return=is_return)
        was_converted = was_converted or ok
        return conv

    new = tuple(convert(arg, annotation) for arg, annotation in zip(value, typing.get_args(annotation)))

    if was_converted:
        return new, True

    return value, False


def _convert_sequence(value, annotation, is_return=False):
    inner_annotation = typing.get_args(annotation)[0]
    if not _has_unit_type(inner_annotation):
        # If no Unit conversions are called for, skip a potentially
        # lengthy process
        return value, False

    was_converted = False

    def convert(arg):
        nonlocal was_converted
        conv, ok = _convert_to_annotation(arg, inner_annotation, is_return=is_return)
        was_converted = was_converted or ok
        return conv

    new = type(value)(convert(arg) for arg in value)
    if was_converted:
        return new, True

    return value, False


def _convert_union(value, annotation, is_return=False):
    if any(isinstance(sub, Unit) for sub in typing.get_args(annotation)):
        return _convert_union_unit(value, annotation, is_return=is_return)
    else:
        return _convert_union_any(value, annotation, is_return=is_return)

    return value, False


def _convert_union_unit(value, annotation, is_return=False):
    """Convert based off a Union annotation that contains at least one
    Unit at its top level. We do special handling to try to eliminate
    ambiguity."""

    # First see if we can match more than one Unit, and if so, fail
    converted = None
    was_converted = False
    for sub in typing.get_args(annotation):
        if isinstance(sub, Unit):
            new = None
            try:
                new, was_converted = _convert_unit(value, sub, is_return=is_return)
            except Exception:
                pass
            if new is not None and converted is not None:
                raise ValueError(f"Ambiguous conversion for {value} under {annotation}")
            converted = new
    if converted is not None:
        return converted, was_converted

    # We don't do general type checking, but we want to be sure as
    # much as possible that users have to provide units in numeric
    # contexts
    if not is_return and isinstance(value, (numbers.Number, np.ndarray)) and not isinstance(value, bool) and not any(sub == Any for sub in typing.get_args(annotation)):
        raise ValueError(f"Could not convert {value} under {annotation}")

    return _convert_union_any(value, annotation, is_return=is_return)


def _convert_union_any(value, annotation, is_return=False):
    """Convert a union of arbitrary types. The first matching choice
    is used, even if ambiguity might exist between it and a later
    choice. This avoids a potentially costly search at the cost of
    less type checking."""

    for sub in typing.get_args(annotation):
        try:
            return _convert_to_annotation(value, sub, is_return=is_return)
        except Exception:
            pass

    raise ValueError(f"Could not convert {value} under {annotation}")


def _has_unit_type(annotation):
    if isinstance(annotation, slice):
        warnings.warn(f"Annotation {annotation} is a slice. This almost always happens when you use a colon when defining a dict type instead of a comma.")
    if isinstance(annotation, Unit):
        return True
    return any(_has_unit_type(arg) for arg in typing.get_args(annotation))
