"""Decorator registry for swappable pipeline stages."""

from __future__ import annotations

from typing import Any, Callable


class RegistryError(KeyError):
    """Raised when a config names a stage implementation that is not registered."""


_REGISTRY: dict[tuple[str, str], type] = {}


def register(stage: str, name: str) -> Callable[[type], type]:
    """Register ``cls`` as the implementation ``name`` of pipeline ``stage``."""

    def decorator(cls: type) -> type:
        key = (stage, name)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise RegistryError(f"{stage}:{name} already registered to {_REGISTRY[key].__name__}")
        _REGISTRY[key] = cls
        cls.stage = stage  # type: ignore[attr-defined]
        cls.impl_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def get(stage: str, name: str) -> type:
    try:
        return _REGISTRY[(stage, name)]
    except KeyError as exc:
        known = ", ".join(available(stage)) or "(none)"
        raise RegistryError(f"unknown {stage} implementation '{name}'. Known: {known}") from exc


def available(stage: str) -> list[str]:
    return sorted(name for (stg, name) in _REGISTRY if stg == stage)


def build(stage: str, spec: dict[str, Any] | None) -> Any:
    """Instantiate a registered stage from a config mapping.

    ``spec`` must include ``type`` (or ``name``). Remaining keys are constructor kwargs.
    """

    if not spec:
        raise RegistryError(f"{stage} config is empty")
    params = dict(spec)
    type_name = params.pop("type", None) or params.pop("name", None)
    if not type_name:
        raise RegistryError(f"{stage} config missing 'type'")
    cls = get(stage, str(type_name))
    return cls(**params)


def clear() -> None:
    """Drop registrations. Used by tests that want a clean registry."""

    _REGISTRY.clear()
