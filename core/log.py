"""Central logging setup. Import via `from core.log import get_logger`."""
import logging
import os

_configured = False


def configure_logging(level=None):
    global _configured
    if _configured:
        return
    raw = level or os.environ.get("ATCS_LOG_LEVEL", "WARNING")
    try:
        lvl = getattr(logging, str(raw).upper())
    except Exception:
        lvl = logging.WARNING
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Quiet noisy third-party libs by default; app code uses WARNING+ unless VERBOSE.
    for noisy in ("matplotlib", "PIL", "ultralytics"):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass
    _configured = True


def get_logger(name):
    configure_logging()
    return logging.getLogger(name)
