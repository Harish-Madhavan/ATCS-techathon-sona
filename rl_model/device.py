"""Central device selection helper (xpu > cuda > cpu)."""
import torch


def get_best_device():
    """Return torch.device for best available accelerator."""
    xpu = getattr(torch, 'xpu', None)
    if xpu is not None:
        try:
            if xpu.is_available():
                return torch.device('xpu')
        except Exception:
            pass
    try:
        if torch.cuda.is_available():
            return torch.device('cuda')
    except Exception:
        pass
    return torch.device('cpu')


def get_best_device_str():
    """Return device as plain string for workers that expect 'xpu'/'cuda'/'cpu'."""
    return str(get_best_device())
