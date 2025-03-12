from typing import Any, Type
from nptyping import NDArray, Shape

from icecream import ic

Vector = NDArray[Shape['*'], Any] # TODO: make this actually specifiy a vector. At the moment, I think it takes any shape. 
Matrix = NDArray[Shape['Size, Size'], Any]

# def update_annotations(annotations: dict['str', Any], bases: list[type]) -> None:
#     for base in bases:
#         annotations.update(getattr(base, '__annotations__', dict()))