"""Compatibility layer for Pydantic & Prometheus in bare Python environments.

Uses native `pydantic` and `prometheus_client` when installed (Docker / venv),
and provides a stdlib-compatible fallback so CLI tools and unit tests also run
cleanly on bare host Python installations before `pip install`.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict

try:
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:
    class _FieldSpec:
        def __init__(
            self,
            default: Any = ...,
            default_factory: Callable[[], Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self.default = default
            self.default_factory = default_factory
            self.metadata = kwargs

    def Field(  # type: ignore # noqa: N802
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return _FieldSpec(default=default, default_factory=default_factory, **kwargs)

    class BaseModel:  # type: ignore
        """Lightweight stdlib fallback matching Pydantic v2 BaseModel semantics."""

        def __init__(self, **kwargs: Any) -> None:
            for cls in reversed(self.__class__.__mro__):
                if cls is object:
                    continue
                annotations = getattr(cls, "__annotations__", {})
                for attr_name in annotations:
                    if attr_name in kwargs:
                        continue
                    if hasattr(cls, attr_name):
                        raw_val = getattr(cls, attr_name)
                        if isinstance(raw_val, _FieldSpec):
                            if raw_val.default_factory is not None:
                                setattr(self, attr_name, raw_val.default_factory())
                            elif raw_val.default is not ...:
                                setattr(self, attr_name, copy.deepcopy(raw_val.default))
                        elif not isinstance(raw_val, property) and not callable(raw_val):
                            setattr(self, attr_name, copy.deepcopy(raw_val))

            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            return self._to_primitive(self.__dict__)

        @classmethod
        def _to_primitive(cls, obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                return {k: cls._to_primitive(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, dict):
                return {k: cls._to_primitive(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [cls._to_primitive(v) for v in obj]
            return obj

__all__ = ["BaseModel", "Field"]
