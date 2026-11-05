# model_architecture.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridComputeManager:
    """Manages dynamic GPU/CPU selection, thread pools, and OOM fallback."""
    def __init__(self, force_cpu=False):
        self.device = self._select_optimal_device(force_cpu)
        self._configure_resources()

    def _select_optimal_device(self, force_cpu):
        if force_cpu:
            print("[COMPUTE] Force-CPU flag active. Running on CPU.")
            return torch.device("cpu")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[COMPUTE] NVIDIA GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM). Using CUDA.")
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[COMPUTE] Apple Silicon GPU Detected. Using MPS acceleration.")
            return torch.device("mps")
        else:
            print("[COMPUTE] No dedicated GPU detected. Optimizing for Multi-Core CPU execution.")
            return torch.device("cpu")

    def _configure_resources(self):
        if self.device.type == "cpu":
            num_cores = max(1, os.cpu_count() - 1)
            torch.set_num_threads(num_cores)
            print(f"[COMPUTE] CPU Thread Pool locked to {num_cores} threads.")
        elif self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def safe_forward(self, model, tensor_in):
        """Executes forward pass; automatically falls back to CPU if CUDA OOM occurs."""
        try:
            model = model.to(self.device)
            tensor_in = tensor_in.to(self.device)
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = model(tensor_in)
                    return out.float().cpu()
                else:
                    return model(tensor_in).cpu()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[WARN] GPU VRAM exceeded! Dynamically failing over to System RAM (CPU)...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                model = model.to("cpu")
                tensor_in = tensor_in.to("cpu")
                with torch.no_grad():
                    return model(tensor_in).cpu()
            raise e


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class WeatherSpatioTemporalUNet(nn.Module):
    def __init__(self, in_channels=30, out_channels=6):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

        nn.init.normal_(self.head.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        pad_h = (4 - orig_h % 4) % 4
        pad_w = (4 - orig_w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        b = self.bottleneck(F.avg_pool2d(e2, 2))

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        delta = self.head(d1)
        return delta[:, :, :orig_h, :orig_w]
