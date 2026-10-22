# GPU-local CPU affinity

The P800 SFT launchers honor an existing `CUDA_VISIBLE_DEVICES` value. From the
repository root, select either eight-card group on server 53:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/launch_sft_vision_nano_p800.sh
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 bash examples/launch_sft_vision_nano_p800.sh
```

The runtime resolves logical device indices after CUDA visibility/device-order
remapping. CPU placement looks up each device's physical PCI address and Linux
NUMA node, then divides that node's available physical cores among its local
training ranks. Available SMT siblings of one physical core stay with the same
rank. CPU allocations are intersected with the original process CPU allowance;
repeated calls do not repartition an already narrowed mask. A severely restricted
cpuset may require core sharing. If topology cannot be determined or no local CPU
is allocated, the original allocation is retained with a warning.

On server 53, cards 0-7 are on NUMA 0 and cards 8-15 on NUMA 1. With all CPUs
available, each rank of an eight-card run gets 16 physical cores and their SMT
siblings. CPU numbers are not contiguous by NUMA: node 0 has 0-127,256-383 and
node 1 has 128-255,384-511. Launch logs include each local rank's PCI address,
NUMA node and selected CPU list. DataLoader workers normally inherit the mask;
external worker-affinity hooks can override it and should be checked separately.

This corrects CPU placement. It does not force memory policy, migrate existing
pages, change GPU clocks/power, or guarantee identical step times for all recipes.
Use the same recipe, precision, batch, dataset and warm-up window when comparing
card groups. Do not infer the cause of transient slow steps from one short run.
