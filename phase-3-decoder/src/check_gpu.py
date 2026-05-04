from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            memory_gb = props.total_memory / (1024 ** 3)
            print(f"cuda:{index} {props.name} ({memory_gb:.1f} GB)")
    print(f"mps available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
