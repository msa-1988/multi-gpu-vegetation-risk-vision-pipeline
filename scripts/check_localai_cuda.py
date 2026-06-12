from __future__ import annotations

import sys

import torch


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"gpu count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"gpu name: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
        print(f"arch list: {torch.cuda.get_arch_list()}")
        x = torch.randn(256, 256, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"cuda tensor test: ok, mean={y.mean().item():.6f}")


if __name__ == "__main__":
    main()

