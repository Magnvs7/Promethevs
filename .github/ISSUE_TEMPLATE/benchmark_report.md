---
name: Benchmark report
about: Share PROMETHEVS speed and quality results from your machine
title: "Benchmark: <CPU> / <RAM> / <GPU>"
labels: benchmark
---

Thanks for running PROMETHEVS on your machine. Reports from other hardware are the most useful thing you can send.

## Your machine
- CPU:
- RAM (size, type and speed, e.g. 64 GB DDR4-3200):
- GPU and VRAM:
- Drives used for the stores (model, PCIe generation):
- Windows version and NVIDIA driver version:
- PROMETHEVS release (e.g. 0.2):

## The install check
With the engine freshly started and the PC otherwise idle, run:

```
python scripts\py\verify_install.py --speed
```

Paste its full output here:

```
(paste here)
```

## Engine log
Attach the newest `logs\engine_*.log` (drag the file into this box).

## Anything else
Anything that went wrong during the install or store build, how long the build took, or other notes.
