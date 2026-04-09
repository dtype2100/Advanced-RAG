"""Provider abstraction layer.

Each module exposes a factory/singleton that resolves the concrete
implementation from the ``PROVIDER`` environment variables defined in
``app.config``.  Consumers always import from this package rather than
from the concrete provider libraries directly.
"""
