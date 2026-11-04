"""Stream event timing for diagnostics; host overlap is reported separately."""
import time
import torch


class StageTimer:
    def __init__(self,device,enabled=True):
        self.device=torch.device(device)
        self.enabled=enabled
        self.events = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) if enabled and self.device.type=='cuda' else None

    def __enter__(self):
        self.started=time.perf_counter()
        if self.events: self.events[0].record()
        return self

    def __exit__(self,*_):
        if self.events:
            self.events[1].record()
            self.events[1].synchronize()
            self.cuda_ms=self.events[0].elapsed_time(self.events[1])
        else:self.cuda_ms=None
        self.host_ms=(time.perf_counter()-self.started)*1000.
