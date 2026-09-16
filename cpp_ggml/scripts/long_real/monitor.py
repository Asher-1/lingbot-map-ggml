#!/usr/bin/env python3
"""Resource monitor wrapper for the long_real comparison campaign.

Runs a command as a child process and samples every --interval seconds:
  - GPU: global memory used + per-process memory for the child PID tree +
    GPU utilization (nvidia-smi);
  - host: aggregated RSS of the child PID tree + VmHWM (peak RSS) per PID.

Writes a JSON record (command, exit code, wall time, per-sample timeline,
peaks) to --out and forwards the child's exit code. The timeline is capped
(--max-samples) so hour-long runs stay bounded; peaks are tracked live.

Usage:
  monitor.py --out run.json -- java ...            # any command
  monitor.py --out run.json --label strict_cuda -- lingbot-map-cli ...
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time


def pid_tree(root):
    """root + descendants (best effort; the engines are shallow trees)."""
    tree = {root}
    try:
        children = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    parts = f.read().rsplit(") ", 1)[1].split()
                    ppid = int(parts[1])
                children.setdefault(ppid, set()).add(int(entry))
            except (OSError, IndexError, ValueError):
                continue
        frontier = [root]
        while frontier:
            cur = frontier.pop()
            for ch in children.get(cur, ()):
                if ch not in tree:
                    tree.add(ch)
                    frontier.append(ch)
    except OSError:
        pass
    return tree


def rss_kb(pids):
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except OSError:
            continue
    return total


def peak_rss_kb(pids):
    peak = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak = max(peak, int(line.split()[1]))
                        break
        except OSError:
            continue
    return peak


def gpu_snapshot():
    """(global_used_mb, util_pct, {pid: mb}) or (None, None, {}) if nvidia-smi fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        used, util = (None, None)
        if out:
            first = out.splitlines()[0].split(",")
            used, util = float(first[0]), float(first[1])
        per = {}
        try:
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            for line in apps.splitlines():
                parts = line.split(",")
                if len(parts) == 2:
                    per[int(parts[0])] = float(parts[1])
        except (ValueError, subprocess.TimeoutExpired):
            pass
        return used, util, per
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None, None, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    args = ap.parse_args()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (place it after --)")

    start = time.time()
    proc = subprocess.Popen(cmd)
    samples = []
    stop = threading.Event()
    vmhwm_peak_kb = [0]

    def sampler():
        while not stop.is_set():
            pids = pid_tree(proc.pid)
            used, util, per = gpu_snapshot()
            proc_gpu = sum(v for pid, v in per.items() if pid in pids)
            # VmHWM must be read while the process is alive (it vanishes with
            # /proc/<pid> on exit, so wait() is too late).
            vmhwm_peak_kb[0] = max(vmhwm_peak_kb[0], peak_rss_kb(pids))
            sample = {
                "t": round(time.time() - start, 2),
                "gpu_used_mb": used,
                "proc_gpu_mb": proc_gpu or None,
                "gpu_util_pct": util,
                "rss_mb": round(rss_kb(pids) / 1024, 1),
            }
            samples.append(sample)
            if len(samples) >= args.max_samples:
                break
            stop.wait(args.interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    code = proc.wait()
    stop.set()
    thread.join(timeout=5)
    wall = time.time() - start

    def peak(key):
        vals = [s[key] for s in samples if s.get(key) is not None]
        return max(vals) if vals else None

    utils = [s["gpu_util_pct"] for s in samples if s.get("gpu_util_pct") is not None]
    record = {
        "label": args.label or " ".join(cmd[:4]),
        "cmd": cmd,
        "exit_code": code,
        "wall_s": round(wall, 2),
        "n_samples": len(samples),
        "peaks": {
            "gpu_used_mb": peak("gpu_used_mb"),
            "proc_gpu_mb": peak("proc_gpu_mb"),
            "rss_mb": peak("rss_mb"),
            "vmhwm_mb": round(vmhwm_peak_kb[0] / 1024, 1),
            "gpu_util_mean_pct": round(sum(utils) / len(utils), 1) if utils else None,
        },
        "timeline": samples,
    }
    with open(args.out, "w") as f:
        json.dump(record, f)
    print(f"MONITOR label={args.label} wall={wall:.1f}s exit={code} "
          f"gpu_peak={record['peaks']['gpu_used_mb']}MB proc_gpu_peak={record['peaks']['proc_gpu_mb']}MB "
          f"rss_peak={record['peaks']['rss_mb']}MB vmhwm={record['peaks']['vmhwm_mb']}MB")
    sys.exit(code)


if __name__ == "__main__":
    main()
