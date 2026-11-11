"""Run the pinned Ultralytics locally without analytics or upload integrations."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import random
import sys

from scripts.environment import configure_caches

LOCAL_FLAGS = ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")
_MISSING = object()


def _disabled(*args, **kwargs):
    return None


@contextmanager
def local_ultralytics():
    """Guard constructors and execution; leave user credentials untouched."""
    configure_caches()
    previous_sentry = sys.modules.get("sentry_sdk", _MISSING)
    # set_sentry() imports this optional SDK during the first utils import.
    sys.modules["sentry_sdk"] = None
    try:
        ultralytics = importlib.import_module("ultralytics")
        utils = importlib.import_module("ultralytics.utils")
        ultralytics.settings.update(dict.fromkeys(LOCAL_FLAGS, False))
        utils.set_sentry = _disabled
        state = random.getstate()
        try:
            event_module = importlib.import_module("ultralytics.utils.events")
        finally:
            # The upstream singleton consumes random.random() even when disabled.
            random.setstate(state)
        event_module.events.enabled = False
        event_module.events.events.clear()
        callbacks = importlib.import_module("ultralytics.utils.callbacks")
    finally:
        if previous_sentry is _MISSING:
            sys.modules.pop("sentry_sdk", None)
        else:
            sys.modules["sentry_sdk"] = previous_sentry
    integration_loader = callbacks.add_integration_callbacks
    callbacks.add_integration_callbacks = _disabled
    try:
        yield ultralytics
    finally:
        callbacks.add_integration_callbacks = integration_loader
