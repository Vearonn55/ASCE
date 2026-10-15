"""Import built-in stages and task plugins so their registrations exist."""

from __future__ import annotations

_LOADED = False


def load_builtins() -> None:
    global _LOADED
    if _LOADED:
        return
    import asce.stages.decoders.argmax  # noqa: F401
    import asce.stages.features.logmel  # noqa: F401
    import asce.stages.features.mfcc  # noqa: F401
    import asce.stages.models.cnn  # noqa: F401
    import asce.stages.models.hf_backbone  # noqa: F401
    import asce.stages.models.mlp  # noqa: F401
    import asce.stages.preprocess.standard  # noqa: F401
    import asce.stages.segment.fixed  # noqa: F401
    import asce.stages.segment.onset  # noqa: F401
    import asce.stages.sources.file  # noqa: F401
    import asce.stages.sources.mic  # noqa: F401
    import plugins.cpuhum.decoder  # noqa: F401
    import plugins.cpuhum.features  # noqa: F401
    import plugins.keytap.decoder  # noqa: F401

    _LOADED = True
