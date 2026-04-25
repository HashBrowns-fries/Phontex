#!/usr/bin/env python3
"""Smoke test — verify all core dependencies are importable."""
import torch
import transformers
import peft
import fastapi

print("All deps OK")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"peft={peft.__version__}")
print(f"fastapi={fastapi.__version__}")