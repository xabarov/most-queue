"""
Utilities for parameter validation and type checking.
"""

from typing import Any


def validate_positive(value: float, name: str = "value") -> None:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate.
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str = "value") -> None:
    """
    Validate that a value is non-negative.

    Args:
        value: Value to validate.
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_integer(value: Any, name: str = "value") -> None:
    """
    Validate that a value is an integer.

    Args:
        value: Value to validate.
        name: Name of the parameter for error messages.

    Raises:
        TypeError: If value is not an integer.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def validate_positive_integer(value: int, name: str = "value") -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate.
        name: Name of the parameter for error messages.

    Raises:
        TypeError: If value is not an integer.
        ValueError: If value is not positive.
    """
    validate_integer(value, name)
    validate_positive(float(value), name)


def validate_list_not_empty(value: list[Any], name: str = "value") -> None:
    """
    Validate that a list is not empty.

    Args:
        value: List to validate.
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If list is empty.
    """
    if not value:
        raise ValueError(f"{name} must not be empty")


def validate_list_length(value: list[Any], min_length: int, name: str = "value") -> None:
    """
    Validate that a list has at least a minimum length.

    Args:
        value: List to validate.
        min_length: Minimum required length.
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If list is shorter than min_length.
    """
    if len(value) < min_length:
        raise ValueError(f"{name} must have at least {min_length} elements, got {len(value)}")
