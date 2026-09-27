# -*- coding: utf-8 -*-
"""
CHB-MIT 分段下载器：利用 EDF 固定记录结构 + HTTP Range 请求，
只下载发作段（ictal）与远离发作的间期段（interictal）所需的字节。
EDF: header = 256 + 256*nchan bytes; 每个数据记录 = 1 秒,
记录内按通道顺序存放 256 个 int16 样本 => 23*256*2 = 11776 B/s。
可中断续跑：已存在的段文件自动跳过。
"""
import json, os, sys, time
import numpy as np
import urllib.request

BASE = "https://physionet.org/files/chbmit/1.0.0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segments")
os.makedirs(OUT, exist_ok=True)

HDR_BYTES = 256 + 256 * 23
REC_BYTES = 23 * 256 * 2  # 11776 bytes per second
CH_IDX = [1, 13]          # F7-T7, F8-T8

def http_range(url, start, end, retries=4):
    """下载 [start, end) 字节区间"""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end-1}"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start:
                return data
        except Exception as e:
            print(f"    retry {a+1}: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"failed: {url} {start}-{end}")

def get_header_scaling(sid, fn):
    """读 EDF 头，返回 (n_records, physical_min/max, digital_min/max) 每通道"""
    url = f"{BASE}/{sid}/{fn}"
    raw = http_range(url, 0, HDR_BYTES)
    hdr = raw
    n_records = int(hdr[236:244].decode('ascii').strip())
    ns = int(hdr[252:256].decode('ascii').strip())
    assert ns == 23
    # 各通道参数区（每通道字段连续存放）
    off = 256
    def field(width):
        nonlocal off
        vals = [hdr[off + i*width: off + (i+1)*width].decode('ascii').strip() for i in range(ns)]
        off += width * ns
        return vals
    labels = field(16); transducer = field(80); phys_dim = field(8)
    phys_min = [float(x) for x in field(8)]; phys_max = [float(x) for x in field(8)]
    dig_min = [float(x) for x in field(8)];  dig_max = [float(x) for x in field(8)]
    return n_records, np.array(phys_min), np.array(phys_max), np.array(dig_min), np.array(dig_max), labels

def decode_segment(raw, phys_min, phys_max, dig_min, dig_max):
    """把秒级字节块解码为 (2, n*256) 的 µV 浮点信号（仅 F7-T7, F8-T8）"""
    nsec = len(raw) // REC_BYTES
    x = np.frombuffer(raw[:nsec*REC_BYTES], dtype='<i2').reshape(nsec, 23, 256)
    x = x[:, CH_IDX, :].astype(np.float64)  # (nsec, 2, 256)
    for i, c in enumerate(CH_IDX):
        scale = (phys_max[c] - phys_min[c]) / (dig_max[c] - dig_min[c])
        x[:, i, :] = (x[:, i, :] - dig_min[c]) * scale + phys_min[c]
    n, c, s = x.shape
    return x.transpose(1, 0, 2).reshape(c, n * s)  # (2, nsec*256)

def plan_segments(sid, info, interictal_per_file=200, n_files_inter=3, min_gap=300):
    """生成下载计划: [(fn, t0, t1, kind)]"""
    segs = []
    for fn, seiz in info['files'].items():
        for s, e in seiz:
            segs.append((fn, max(0, s - 4), min(3600, e + 4), 'ictal'))
    # interictal: 从含发作的文件中取远离任何发作(min_gap秒以上)的连续块
    inter_files = list(info['files'].items())[:n_files_inter]
    for fn, seiz in inter_files:
        taken = 0
        t = 120
        while taken < interictal_per_file and t + interictal_per_file <= 3480:
            ok = all((t + interictal_per_file < s - min_gap) or (t > e + min_gap) for s, e in seiz)
            if ok:
                segs.append((fn, t, t + interictal_per_file, 'interictal'))
                taken += interictal_per_file
                break
            t += 200
    return segs

def main():
    ann = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotations.json')))
    todo = []
    for sid, info in ann.items():
        for fn, t0, t1, kind in plan_segments(sid, info):
            out = os.path.join(OUT, f"{sid}_{fn.replace('.edf','')}_{t0}_{t1}_{kind}.npy")
            if not os.path.exists(out):
                todo.append((sid, fn, t0, t1, kind, out))
    print(f"segments remaining: {len(todo)}", flush=True)
    hdr_cache = {}
    for sid, fn, t0, t1, kind, out in todo:
        url = f"{BASE}/{sid}/{fn}"
        if (sid, fn) not in hdr_cache:
            hdr_cache[(sid, fn)] = get_header_scaling(sid, fn)
        nrec, pmin, pmax, dmin, dmax, labels = hdr_cache[(sid, fn)]
        t1c = min(t1, nrec)
        print(f"GET {sid}/{fn} [{t0},{t1c}) {kind} ({(t1c-t0)*REC_BYTES/1e6:.1f} MB)", flush=True)
        raw = http_range(url, HDR_BYTES + t0*REC_BYTES, HDR_BYTES + t1c*REC_BYTES)
        sig = decode_segment(raw, pmin, pmax, dmin, dmax)
        np.save(out, sig.astype(np.float32))
    print("DONE", flush=True)

if __name__ == '__main__':
    main()
