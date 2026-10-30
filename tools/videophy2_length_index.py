#!/usr/bin/env python3
"""Build a per-sample token-length index for the VideoPhy-2 manifest.

Why this exists: FSDP all_gather in this recipe is dominated by RANK ARRIVAL SKEW
(measured 2026-10-28: skew median 17.29 ms vs transfer 5.49 ms, and end-of-
collective spread only 0.068 ms). The skew comes from sample length: ranks whose
microstep holds a long sample arrive late and every other rank waits. Balancing
length across the 8 ranks at each microstep needs the length BEFORE the sample is
read, which the manifest does not carry (only id / media / conversation).

Fortunately the dataset is quantised: a metadata-only scan of all 3343 training
videos yields exactly THREE (sampled_frames, width, height) combinations, and each
maps 1:1 onto a grid_thw observed in training logs and hence onto an exact
input_ids length. So this is a lookup table, not a regression -- no decoding and
no processor run required (full scan ~9 s).

    (T, W, H)         grid_thw      input_ids
    (8,  512, 320)    [4, 20, 32]   976
    (13, 720, 480)    [7, 30, 44]   2672
    (11, 1280, 704)   [6, 44, 80]   5632

Frame count replicates videophy2_sft_nano.py:_decode_video_to_pil_frames:
stride = max(1, round(fps / 2.0)); T = min(32, len(range(0, num_frames, stride))).

Unknown combinations are recorded with length `null` so the consumer can fall back
to the unbalanced order instead of silently mis-sorting.

Usage:
    python tools/videophy2_length_index.py \
        --root $VIDEOPHYSICS_ROOT/videophy2_train \
        --out  $VIDEOPHYSICS_ROOT/videophy2_train/length_index.json
"""

import argparse
import json
import os
import time

MAX_VIDEO_FRAMES = 32
TARGET_VIDEO_FPS = 2.0

# (sampled_frames, width, height) -> exact input_ids length, from the log-observed
# grid_thw of each class. Extend only with a value confirmed against a real run.
SPEC_TO_LENGTH = {
    (8, 512, 320): 976,
    (13, 720, 480): 2672,
    (11, 1280, 704): 5632,
}


def sampled_frames(num_frames: int, fps: float) -> int:
    stride = max(1, round((fps or 30.0) / TARGET_VIDEO_FPS))
    return min(MAX_VIDEO_FRAMES, len(range(0, num_frames, stride)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="split dir holding meta.json and media/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from torchcodec.decoders import VideoDecoder  # noqa: PLC0415 - heavy optional dep

    manifest = json.load(open(os.path.join(args.root, "meta.json")))
    lengths, specs, failed, unknown = {}, {}, [], []
    t0 = time.time()
    for entry in manifest:
        key = entry.get("id") or entry.get("key")
        try:
            md = VideoDecoder(os.path.join(args.root, entry["media"])).metadata
            spec = (
                sampled_frames(md.num_frames or 0, float(md.average_fps or 0.0)),
                md.width,
                md.height,
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the index
            failed.append({"id": key, "error": repr(exc)})
            lengths[key] = None
            continue
        specs[str(spec)] = specs.get(str(spec), 0) + 1
        lengths[key] = SPEC_TO_LENGTH.get(spec)
        if lengths[key] is None:
            unknown.append({"id": key, "spec": list(spec)})

    payload = {
        "source_manifest": os.path.join(args.root, "meta.json"),
        "built_at": datetime_now(),
        "samples": len(manifest),
        "scan_seconds": round(time.time() - t0, 2),
        "spec_histogram": specs,
        "spec_to_length": {str(k): v for k, v in SPEC_TO_LENGTH.items()},
        "unknown_spec_count": len(unknown),
        "unknown_spec_examples": unknown[:10],
        "failed_count": len(failed),
        "failed_examples": failed[:10],
        "lengths": lengths,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    os.replace(tmp, args.out)

    covered = sum(1 for v in lengths.values() if v is not None)
    print(f"扫描 {len(manifest)} 条,耗时 {payload['scan_seconds']}s")
    print(f"规格直方图: {specs}")
    print(f"覆盖 {covered}/{len(manifest)} ({100*covered/len(manifest):.1f}%)"
          f"  未知规格 {len(unknown)}  读取失败 {len(failed)}")
    print(f"已写入 {args.out}")
    if covered != len(manifest):
        print("⚠ 存在未覆盖样本:消费端应对 null 回退到未配平顺序,不要静默排序")


def datetime_now() -> str:
    import datetime

    return datetime.datetime.now().astimezone().isoformat()


if __name__ == "__main__":
    main()
