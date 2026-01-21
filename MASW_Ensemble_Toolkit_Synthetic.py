import os, sys, math, json, random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

import os as _os
from pathlib import Path as _Path

def _running_in_colab():
    try:
        import google.colab
        return True
    except Exception:
        return False

IS_COLAB = _running_in_colab()

if IS_COLAB:
    ROOT = _Path("/content")
else:
    ROOT = _Path(__file__).resolve().parent

_default_data = ROOT / "data" / "masw_demo"
_default_out  = ROOT / "outputs"

DATA_DIR = _Path(_os.environ.get("MASW_DATA_DIR", str(_default_data)))
OUT_DIR  = _Path(_os.environ.get("MASW_OUT_DIR",  str(_default_out)))

if IS_COLAB and (str(DATA_DIR).startswith("/content/drive") or str(OUT_DIR).startswith("/content/drive")):
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "picked_curves").mkdir(parents=True, exist_ok=True)

print("[INFO] IS_COLAB =", IS_COLAB)
print("[INFO] DATA_DIR =", DATA_DIR)
print("[INFO] OUT_DIR  =", OUT_DIR)

FOLDER = DATA_DIR


F_HZ_MIN = 2.0
F_HZ_MAX = 30.0
C_MS_MIN = 80.0
C_MS_MAX = 600.0

SYNTHETIC_MODE = True
N_SYN_GRIDS = 60
SYN_SEED = 42

SYN_C0_RANGE = (120.0, 220.0)
SYN_C1_RANGE = (250.0, 520.0)
SYN_P_RANGE  = (0.25, 0.65)

SYN_SIGMA_C_RANGE = (10.0, 30.0)
SYN_BG_NOISE = 0.15
SYN_RIDGE_PEAK = 1.0
SYN_SPUR_BLOBS = 2

def _rng(seed=SYN_SEED):
    return np.random.default_rng(int(seed))

def synthetic_curve(f_hz, rng):
    c0 = rng.uniform(*SYN_C0_RANGE)
    c1 = rng.uniform(*SYN_C1_RANGE)
    p  = rng.uniform(*SYN_P_RANGE)
    f = np.asarray(f_hz, dtype=float)
    return c0 + c1 * np.power(np.maximum(f, 1e-6), -p)

def generate_synthetic_grid(f_axis, c_axis, rng):
    c_r = synthetic_curve(f_axis, rng)
    sigma_c = rng.uniform(*SYN_SIGMA_C_RANGE)

    F = f_axis[:, None]
    C = c_axis[None, :]
    CR = c_r[:, None]

    ridge = np.exp(-0.5 * ((C - CR) / sigma_c) ** 2)
    ridge = ridge / (np.max(ridge) + 1e-12) * SYN_RIDGE_PEAK

    blob = np.zeros_like(ridge)
    for _ in range(int(SYN_SPUR_BLOBS)):
        f0 = rng.uniform(float(f_axis.min()), float(f_axis.max()))
        c0 = rng.uniform(float(c_axis.min()), float(c_axis.max()))
        sf = rng.uniform(0.4, 2.0)
        sc = rng.uniform(20.0, 80.0)
        blob += np.exp(-0.5 * ((F - f0) / sf) ** 2 - 0.5 * ((C - c0) / sc) ** 2) * rng.uniform(0.15, 0.6)

    noise = rng.random(size=ridge.shape) * float(SYN_BG_NOISE)

    G = ridge + blob + noise

    G = G - np.min(G)
    mx = np.max(G)
    if mx > 0:
        G = G / mx
    return G


F_MIN = None
F_MAX = None
C_MIN = None
C_MAX = None

N_MEMBERS = 30
SMOOTH_WINS = [1, 3, 5, 7, 9, 11]

PICK_METHODS = ["track", "argmax", "weighted_centroid"]

TRACK_LAMBDAS = [0.001, 0.003, 0.01, 0.03]
TRACK_C_SCALES = [20.0, 30.0, 40.0]

F_WINDOWS = [(2.0, 30.0), (3.0, 28.0), (4.0, 26.0), (5.0, 25.0), (6.0, 24.0)]
BOOTSTRAP_FRAC = 0.75

K_DOI = 0.5

N_LAYERS = 6
H_MIN, H_MAX = 1.0, 20.0
VS_MIN, VS_MAX = 80.0, 600.0

ALPHA_DEPTH = 0.5
BETA_RAYLEIGH = 0.92
N_FWD_ITERS = 3

N_SAMPLES = 6000
N_REFINE_STEPS = 250
REFINE_STEP_H = 0.08
REFINE_STEP_VS = 0.08

Z_MAX = 60.0
DZ = 0.25
Z_ZOOM = 20.0

def list_dat_files(folder):
    files = sorted([p for p in Path(folder).glob("*.dat") if p.is_file()])
    if not files:
        raise FileNotFoundError("No .dat files found in: " + str(folder))
    return files

def is_probably_text(path, nbytes=4096):
    with open(path, "rb") as f:
        b = f.read(nbytes)
    if not b:
        return False
    printable = set(range(32, 127)) | {9, 10, 13}
    n_print = sum((x in printable) for x in b)
    return (n_print / float(len(b))) > 0.90

def read_numeric_tokens_text(path, max_lines=None):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.startswith("%") or s.startswith("!"):
                continue
            parts = s.replace(",", " ").split()
            nums = []
            ok = True
            for p in parts:
                try:
                    nums.append(float(p))
                except Exception:
                    ok = False
                    break
            if ok and len(nums) > 0:
                rows.append(nums)
    return rows

def try_parse_triplets_text(rows):
    if len(rows) < 50:
        return None
    if not all(len(r) >= 3 for r in rows[:200]):
        return None
    arr = np.array([r[:3] for r in rows if len(r) >= 3], dtype=float)
    if arr.shape[0] < 50:
        return None

    f = arr[:, 0]
    c = arr[:, 1]
    a = arr[:, 2]

    uf = np.unique(np.round(f, 8))
    uc = np.unique(np.round(c, 8))
    if uf.size < 10 or uc.size < 10:
        return None

    f_sorted = np.sort(uf)
    c_sorted = np.sort(uc)
    fi = {val: i for i, val in enumerate(f_sorted)}
    ci = {val: i for i, val in enumerate(c_sorted)}
    grid = np.zeros((f_sorted.size, c_sorted.size), dtype=float)

    for (ff, cc, aa) in arr:
        ff2 = float(np.round(ff, 8))
        cc2 = float(np.round(cc, 8))
        if ff2 in fi and cc2 in ci:
            grid[fi[ff2], ci[cc2]] += aa

    return f_sorted, c_sorted, grid

def factor_pairs(n):
    pairs = []
    r = int(math.sqrt(n))
    for a in range(1, r + 1):
        if n % a == 0:
            b = n // a
            pairs.append((a, b))
            if a != b:
                pairs.append((b, a))
    return pairs

def shape_score(nf, nc):
    targets = [(201, 256), (256, 201), (201, 512), (512, 201), (159, 647), (647, 159)]
    best = 1e30
    for (tf, tc) in targets:
        best = min(best, abs(nf - tf) + abs(nc - tc))
    aspect = max(nf, nc) / float(max(1, min(nf, nc)))
    pen = 0.0
    if aspect > 12.0:
        pen += 200.0 * (aspect - 12.0)
    return best + pen

def dtype_score(arr):
    if not np.isfinite(arr).all():
        return 1e30
    a = arr.astype(np.float64, copy=False)
    s = a.ravel()
    if s.size > 200000:
        idx = np.random.choice(s.size, size=200000, replace=False)
        s = s[idx]
    if np.all(s == s[0]):
        return 1e30
    q1, q50, q99 = np.quantile(s, [0.01, 0.50, 0.99])
    rng = float(q99 - q1)
    if rng <= 0.0:
        return 1e30
    return abs(q50) * 0.001 + 1.0 / (rng + 1e-12)

def try_parse_binary_grid(path):
    bsz = Path(path).stat().st_size
    header_candidates = [0, 16, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 256, 512, 1024, 2048]
    dtype_candidates = [
        np.dtype("<f8"), np.dtype("<f4"), np.dtype("<i2"),
        np.dtype(">f8"), np.dtype(">f4"), np.dtype(">i2"),
    ]

    best = None
    best_total = 1e30

    for h in header_candidates:
        if h >= bsz:
            continue
        rem = bsz - h

        for dt in dtype_candidates:
            item = dt.itemsize
            if rem % item != 0:
                continue
            n = rem // item
            if n < 1000:
                continue

            pairs = factor_pairs(n)
            plausible = []
            for (nf, nc) in pairs:
                if nf < 50 or nc < 50:
                    continue
                if nf > 3000 or nc > 9000:
                    continue
                plausible.append((nf, nc))
            if not plausible:
                continue

            plausible.sort(key=lambda sh: shape_score(sh[0], sh[1]))

            for (nf, nc) in plausible[:40]:
                try:
                    with open(path, "rb") as f:
                        f.seek(h)
                        arr = np.fromfile(f, dtype=dt, count=nf * nc)
                    if arr.size != nf * nc:
                        continue
                    G = arr.reshape((nf, nc))
                except Exception:
                    continue

                total = shape_score(nf, nc) * 10.0 + dtype_score(G)
                if total < best_total:
                    best_total = total
                    best = (h, dt, nf, nc)

    if best is None:
        raise ValueError("Could not infer binary layout for: " + str(path) + " size=" + str(bsz))

    return best

def read_binary_with_layout(path, header_bytes, dtype, nf, nc):
    dt = np.dtype(dtype)
    count = int(nf) * int(nc)
    with open(path, "rb") as f:
        f.seek(int(header_bytes))
        arr = np.fromfile(f, dtype=dt, count=count)
    if arr.size != count:
        raise ValueError("Not enough samples for locked layout")
    G = arr.reshape((int(nf), int(nc))).astype(np.float64, copy=False)
    f_axis = np.linspace(F_HZ_MIN, F_HZ_MAX, int(nf), dtype=float)
    c_axis = np.linspace(C_MS_MIN, C_MS_MAX, int(nc), dtype=float)
    meta = {"mode": "binary", "header_bytes": int(header_bytes), "dtype": str(dt), "shape": [int(nf), int(nc)]}
    return (f_axis, c_axis, G), meta

def parse_dat_to_grid(path):
    path = Path(path)
    if is_probably_text(path):
        rows = read_numeric_tokens_text(path)
        tri = try_parse_triplets_text(rows)
        if tri is None:
            raise ValueError("Text file but could not parse triplets: " + str(path))
        f, c, G = tri
        return (f, c, G), {"mode": "triplets"}
    else:
        h, dt, nf, nc = try_parse_binary_grid(path)
        return read_binary_with_layout(path, h, dt, nf, nc)

def apply_limits(f, c, grid, fmin=None, fmax=None, cmin=None, cmax=None):
    f = np.asarray(f, dtype=float)
    c = np.asarray(c, dtype=float)
    grid = np.asarray(grid, dtype=float)

    fi = np.ones_like(f, dtype=bool)
    ci = np.ones_like(c, dtype=bool)

    if fmin is not None:
        fi &= (f >= fmin)
    if fmax is not None:
        fi &= (f <= fmax)
    if cmin is not None:
        ci &= (c >= cmin)
    if cmax is not None:
        ci &= (c <= cmax)

    return f[fi], c[ci], grid[np.ix_(fi, ci)]

def stack_grids(grids):
    f0, c0, G0 = grids[0]
    acc = np.zeros_like(G0, dtype=float)
    for (f, c, G) in grids:
        if f.shape != f0.shape or c.shape != c0.shape or G.shape != G0.shape:
            raise ValueError("Grid axis mismatch inside stack")
        acc += G
    return f0, c0, acc

def smooth_1d(x, win=7):
    x = np.asarray(x, dtype=float)
    if win is None or int(win) <= 1:
        return x.copy()
    w = int(win)
    if w % 2 == 0:
        w += 1
    k = w // 2
    xp = np.pad(x, (k, k), mode="edge")
    ker = np.ones((w,), dtype=float) / float(w)
    y = np.convolve(xp, ker, mode="valid")
    return y

def pick_curve_track(f, c, G, smooth_lambda=0.01, smooth_win=7, c_scale=30.0):
    G2 = G.astype(np.float64, copy=False)
    G2 = G2 - np.min(G2)
    gmax = np.max(G2)
    if gmax > 0:
        G2 = G2 / gmax

    nf, nc = G2.shape
    dp = np.zeros((nf, nc), dtype=np.float64)
    back = np.zeros((nf, nc), dtype=np.int32)

    dp[0, :] = -G2[0, :]
    back[0, :] = -1

    max_jump = max(10, int(0.15 * nc))
    inv_cs = 1.0 / float(max(c_scale, 1e-9))

    for i in range(1, nf):
        for j in range(nc):
            j0 = max(0, j - max_jump)
            j1 = min(nc, j + max_jump + 1)
            prev = np.arange(j0, j1, dtype=np.int32)

            dc = (c[j] - c[prev]) * inv_cs
            pen = float(smooth_lambda) * (dc * dc)

            cand = dp[i - 1, prev] + pen
            k = int(np.argmin(cand))

            dp[i, j] = cand[k] - G2[i, j]
            back[i, j] = prev[k]

    idx = np.zeros((nf,), dtype=np.int32)
    idx[-1] = int(np.argmin(dp[-1, :]))
    for i in range(nf - 2, -1, -1):
        idx[i] = back[i + 1, idx[i + 1]]
        if idx[i] < 0:
            idx[i] = idx[i + 1]

    cp = c[idx]
    cp = smooth_1d(cp, smooth_win)
    return f.copy(), cp

def pick_curve(f, c, G, method="track", smooth_win=7, track_lambda=0.01, track_c_scale=30.0):
    if method == "track":
        return pick_curve_track(
            f, c, G,
            smooth_lambda=float(track_lambda),
            smooth_win=int(smooth_win),
            c_scale=float(track_c_scale),
        )

    if method == "argmax":
        idx = np.argmax(G, axis=1)
        cp = c[idx].astype(float)
    elif method == "weighted_centroid":
        W = G.astype(float) - float(np.min(G))
        W = np.maximum(W, 0.0)
        denom = np.sum(W, axis=1) + 1e-12
        cp = np.sum(W * c[None, :], axis=1) / denom
    else:
        raise ValueError("Unknown pick method: " + str(method))

    cp = smooth_1d(cp, smooth_win)
    return f.copy(), cp

def vs_time_averaged(vs_z, z):
    vs_z = np.asarray(vs_z, dtype=float)
    z = np.asarray(z, dtype=float)

    if vs_z.ndim != 1 or z.ndim != 1:
        raise ValueError("vs_time_averaged expects 1D arrays.")
    if vs_z.size != z.size:
        raise ValueError("vs_time_averaged: vs_z and z must have same length.")
    if z.size < 2:
        return vs_z.copy()

    if np.any(np.diff(z) < 0):
        idx = np.argsort(z)
        z = z[idx]
        vs_z = vs_z[idx]

    vs_safe = np.maximum(vs_z, 1e-9)
    s = 1.0 / vs_safe

    cum = np.zeros_like(z, dtype=float)
    for i in range(1, z.size):
        dz = z[i] - z[i - 1]
        cum[i] = cum[i - 1] + 0.5 * dz * (s[i] + s[i - 1])

    out = np.zeros_like(z, dtype=float)
    for i in range(z.size):
        if z[i] <= 0.0:
            out[i] = vs_safe[i]
        else:
            out[i] = z[i] / max(cum[i], 1e-12)
    return out

def nan_stats_1d(mat_2d):
    mu = np.nanmean(mat_2d, axis=0)
    sd = np.nanstd(mat_2d, axis=0)
    cov = sd / np.maximum(mu, 1e-9)
    med = np.nanmedian(mat_2d, axis=0)
    p10 = np.nanpercentile(mat_2d, 10, axis=0)
    p90 = np.nanpercentile(mat_2d, 90, axis=0)
    n_valid = np.sum(np.isfinite(mat_2d), axis=0).astype(float)
    return mu, sd, cov, med, p10, p90, n_valid

def doi_from_curve(f_hz, c_ms, k_doi=0.5):
    f_hz = np.asarray(f_hz, dtype=float)
    c_ms = np.asarray(c_ms, dtype=float)
    if f_hz.size == 0 or c_ms.size == 0:
        return 0.0
    idx = np.argsort(f_hz)
    f0 = float(f_hz[idx[0]])
    c0 = float(c_ms[idx[0]])
    if f0 <= 0.0:
        return 0.0
    lam = c0 / f0
    return float(k_doi) * float(lam)

def vs_profile_from_layers(hs, vs, zvec):
    out = np.empty_like(zvec, dtype=float)
    z0 = 0.0
    for i in range(N_LAYERS - 1):
        z1 = z0 + hs[i]
        mask = (zvec >= z0) & (zvec < z1)
        out[mask] = vs[i]
        z0 = z1
    out[zvec >= z0] = vs[-1]
    return out

def predict_dispersion(f_hz, hs, vs, alpha=ALPHA_DEPTH, beta=BETA_RAYLEIGH, n_iter=N_FWD_ITERS):
    f_hz = np.asarray(f_hz, dtype=float)
    c = np.full_like(f_hz, fill_value=float(np.mean(vs) * beta))

    zvec = np.arange(0.0, max(Z_MAX, float(np.sum(hs) + 5.0)) + 0.5, 0.1)
    vs_z = vs_profile_from_layers(hs, vs, zvec)

    for _ in range(int(n_iter)):
        lam = np.maximum(c / np.maximum(f_hz, 1e-9), 1e-6)
        z_eff = alpha * lam
        z_eff = np.clip(z_eff, 0.0, zvec[-1])
        vs_eff = np.interp(z_eff, zvec, vs_z)
        c = beta * vs_eff
    return c

def misfit_curve(f_hz, c_obs, hs, vs):
    c_pred = predict_dispersion(f_hz, hs, vs)
    r = c_pred - c_obs
    return float(np.mean(r * r))

def sample_models(n):
    hs = np.random.uniform(H_MIN, H_MAX, size=(n, N_LAYERS - 1))
    vs = np.random.uniform(VS_MIN, VS_MAX, size=(n, N_LAYERS))
    return hs, vs

def refine_model(f_hz, c_obs, hs0, vs0, steps=N_REFINE_STEPS):
    hs = hs0.copy()
    vs = vs0.copy()
    best = misfit_curve(f_hz, c_obs, hs, vs)

    for _ in range(int(steps)):
        hs_try = hs * (1.0 + np.random.randn(*hs.shape) * REFINE_STEP_H)
        vs_try = vs * (1.0 + np.random.randn(*vs.shape) * REFINE_STEP_VS)
        hs_try = np.clip(hs_try, H_MIN, H_MAX)
        vs_try = np.clip(vs_try, VS_MIN, VS_MAX)
        m = misfit_curve(f_hz, c_obs, hs_try, vs_try)
        if m < best:
            best = m
            hs = hs_try
            vs = vs_try

    return hs, vs, best

def invert_one_curve(f_hz, c_obs):
    hs_s, vs_s = sample_models(N_SAMPLES)
    best_m = 1e30
    best_hs = None
    best_vs = None

    bs = 300
    for i in range(0, N_SAMPLES, bs):
        j = min(N_SAMPLES, i + bs)
        for k in range(i, j):
            m = misfit_curve(f_hz, c_obs, hs_s[k], vs_s[k])
            if m < best_m:
                best_m = m
                best_hs = hs_s[k].copy()
                best_vs = vs_s[k].copy()

    hs_r, vs_r, m_r = refine_model(f_hz, c_obs, best_hs, best_vs)
    return hs_r, vs_r, m_r

print("Folder:", FOLDER)

if SYNTHETIC_MODE:
    print("[INFO] SYNTHETIC_MODE enabled. Generating synthetic dispersion grids.")
    rng = _rng(SYN_SEED)

    nf = 201
    nc = 256
    f_full = np.linspace(F_HZ_MIN, F_HZ_MAX, nf, dtype=float)
    c_full = np.linspace(C_MS_MIN, C_MS_MAX, nc, dtype=float)

    grids_all = []
    used_files = []
    skipped_files = []

    for i in range(int(N_SYN_GRIDS)):
        G = generate_synthetic_grid(f_full, c_full, rng)
        grids_all.append((f_full, c_full, G))
        used_files.append("synthetic_%03d" % i)

    files = used_files[:]
    print("Synthetic grids:", len(grids_all))

else:
    files = list_dat_files(FOLDER)
    print("Found .dat files:", len(files))
    print("First file:", files[0].name)

    layout_keys = []
    layout_meta = []
    for p in files:
        try:
            h, dt, nf, nc = try_parse_binary_grid(p)
            key = (int(h), str(np.dtype(dt)), int(nf), int(nc))
            layout_keys.append(key)
            layout_meta.append({"file": p.name, "key": key})
        except Exception:
            layout_keys.append(None)
            layout_meta.append({"file": p.name, "key": None})

    keys_clean = [k for k in layout_keys if k is not None]
    if len(keys_clean) == 0:
        raise ValueError("No valid binary layouts detected in folder")

    cnt = Counter(keys_clean)
    best_key, best_count = cnt.most_common(1)[0]
    print("Most common layout:", best_key, "count =", best_count)

    locked_h, locked_dt, locked_nf, locked_nc = best_key

    grids_all = []
    used_files = []
    skipped_files = []

    for p in files:
        try:
            h, dt, nf, nc = try_parse_binary_grid(p)
            key = (int(h), str(np.dtype(dt)), int(nf), int(nc))
            if key != best_key:
                skipped_files.append(p.name)
                continue
            (grid, meta) = read_binary_with_layout(p, locked_h, locked_dt, locked_nf, locked_nc)
            grids_all.append(grid)
            used_files.append(p.name)
        except Exception:
            skipped_files.append(p.name)

    print("Used files:", len(used_files))
    print("Skipped files:", len(skipped_files))
    if len(skipped_files) > 0:
        print("Skipped (first 20):", skipped_files[:20])


f_full, c_full, Gsum_full = stack_grids(grids_all)
fL0, cL0, GL0 = apply_limits(f_full, c_full, Gsum_full, F_MIN, F_MAX, C_MIN, C_MAX)

f_base, c_base = pick_curve(fL0, cL0, GL0, method="argmax", smooth_win=7)

plt.figure()
plt.imshow(GL0.T, aspect="auto", origin="lower",
           extent=[float(fL0.min()), float(fL0.max()), float(cL0.min()), float(cL0.max())])
plt.plot(f_base, c_base, "w-", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase velocity (m/s)")
plt.title("Stacked MASW energy + picked ridge (baseline)")
plt.tight_layout()
plt.savefig(OUT_DIR / "stacked_masw_pick.png", dpi=200)
plt.show()

plt.figure()
plt.plot(f_base, c_base, "o-")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase velocity (m/s)")
plt.title("Picked dispersion curve (baseline)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "picked_dispersion_curve.png", dpi=200)
plt.show()

print("Building dispersion ensemble with N_MEMBERS =", N_MEMBERS)

curves_f = []
curves_c = []
curve_meta = []

n_files = len(grids_all)
idx_all = np.arange(n_files)

for m in range(N_MEMBERS):
    n_take = max(5, int(round(BOOTSTRAP_FRAC * n_files)))
    sel = np.random.choice(idx_all, size=n_take, replace=True)
    subset = [grids_all[i] for i in sel]
    f_s, c_s, G_s = stack_grids(subset)

    smooth_win = random.choice(SMOOTH_WINS)
    method = random.choice(PICK_METHODS)

    if method == "track":
        track_lambda = float(random.choice(TRACK_LAMBDAS))
        c_scale = float(random.choice(TRACK_C_SCALES))
    else:
        track_lambda = 0.01
        c_scale = 30.0

    f_p, c_p = pick_curve(
        f_s, c_s, G_s,
        method=method,
        smooth_win=smooth_win,
        track_lambda=track_lambda,
        track_c_scale=c_scale,
    )

    f_lo, f_hi = random.choice(F_WINDOWS)
    mask = (f_p >= f_lo) & (f_p <= f_hi)
    f_p2 = f_p[mask]
    c_p2 = c_p[mask]

    if f_p2.size < 10:
        f_p2 = f_base.copy()
        c_p2 = c_base.copy()
        f_lo, f_hi = float(f_p2.min()), float(f_p2.max())

    csv_path = OUT_DIR / "picked_curves" / ("curve_%03d.csv" % m)
    np.savetxt(csv_path, np.column_stack([f_p2, c_p2]),
               delimiter=",", header="f_hz,c_ms", comments="")

    curves_f.append(f_p2)
    curves_c.append(c_p2)

    curve_meta.append({
        "member": int(m),
        "n_panels": int(n_take),
        "method": str(method),
        "smooth_win": int(smooth_win),
        "track_lambda": float(track_lambda),
        "track_c_scale": float(c_scale),
        "f_window": [float(f_lo), float(f_hi)],
        "curve_csv": str(csv_path),
    })

plt.figure()
plt.imshow(GL0.T, aspect="auto", origin="lower",
           extent=[float(fL0.min()), float(fL0.max()), float(cL0.min()), float(cL0.max())])
for m in range(N_MEMBERS):
    plt.plot(curves_f[m], curves_c[m], "-", alpha=0.25)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase velocity (m/s)")
plt.title("Dispersion ensemble (corrected)")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "dispersion_ensemble.png", dpi=200)
plt.show()

z = np.arange(0.0, Z_MAX + 1e-12, DZ)
vs_ens = np.zeros((N_MEMBERS, z.size), dtype=float)
vsz_ens = np.zeros((N_MEMBERS, z.size), dtype=float)
misfits = np.zeros((N_MEMBERS,), dtype=float)
z_doi_list = np.zeros((N_MEMBERS,), dtype=float)

print("Inverting each dispersion curve (random search + refinement)...")
for m in range(N_MEMBERS):
    f_obs = curves_f[m]
    c_obs = curves_c[m]

    hs, vs, mf = invert_one_curve(f_obs, c_obs)
    misfits[m] = mf

    vprof = vs_profile_from_layers(hs, vs, z)
    vs_ens[m, :] = vprof
    vsz_ens[m, :] = vs_time_averaged(vprof, z)

    z_doi_list[m] = doi_from_curve(f_obs, c_obs, k_doi=K_DOI)

    if (m + 1) % 5 == 0 or (m + 1) == N_MEMBERS:
        print("Done", m + 1, "/", N_MEMBERS, "| last misfit =", mf)

vs_ens_m = vs_ens.copy()
vsz_ens_m = vsz_ens.copy()
for m in range(N_MEMBERS):
    z_doi = float(z_doi_list[m])
    if z_doi <= 0.0:
        continue
    mask = z > z_doi
    vs_ens_m[m, mask] = np.nan
    vsz_ens_m[m, mask] = np.nan

vs_mean, vs_std, vs_cov, vs_med, vs_p10, vs_p90, n_prof = nan_stats_1d(vs_ens_m)
vsz_mean, vsz_std, vsz_cov, vsz_med, vsz_p10, vsz_p90, n_prof2 = nan_stats_1d(vsz_ens_m)

stats_path = OUT_DIR / "vs_ensemble_stats.csv"
table = np.column_stack([
    z,
    n_prof,
    vs_mean, vs_std, vs_cov, vs_med, vs_p10, vs_p90,
    vsz_mean, vsz_std, vsz_cov, vsz_med, vsz_p10, vsz_p90,
])
np.savetxt(
    stats_path, table, delimiter=",",
    header="z_m,n_profiles,vs_mean,vs_std,vs_cov,vs_median,vs_p10,vs_p90,vsz_mean,vsz_std,vsz_cov,vsz_median,vsz_p10,vsz_p90",
    comments=""
)
print("Saved stats:", stats_path)

def plot_ensemble_summary_4panel(
    title_prefix,
    z_plot,
    vs_ens,
    vs_p10,
    vs_p90,
    vs_med,
    vsz_ens,
    vsz_p10,
    vsz_p90,
    vsz_med,
    vs_cov,
    vsz_cov,
    n_prof,
):
    fig = plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 2, 1)
    n_members = int(vs_ens.shape[0])
    for m in range(n_members):
        ax1.plot(vs_ens[m, :], z_plot, "-", alpha=0.15)
    ax1.fill_betweenx(z_plot, vs_p10, vs_p90, alpha=0.25, label="P10-P90")
    ax1.plot(vs_med, z_plot, "-", linewidth=2.0, label="Median")
    ax1.invert_yaxis()
    ax1.set_xlabel("Vs (m/s)")
    ax1.set_ylabel("Depth z (m)")
    ax1.set_title("Vs(z) ensemble")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2 = plt.subplot(2, 2, 2)
    n_members2 = int(vsz_ens.shape[0])
    for m in range(n_members2):
        ax2.plot(vsz_ens[m, :], z_plot, "-", alpha=0.15)
    ax2.fill_betweenx(z_plot, vsz_p10, vsz_p90, alpha=0.25, label="P10-P90")
    ax2.plot(vsz_med, z_plot, "-", linewidth=2.0, label="Median")
    ax2.invert_yaxis()
    ax2.set_xlabel("Vs_z (m/s)")
    ax2.set_ylabel("Depth z (m)")
    ax2.set_title("Vs_z(z) ensemble (time-avg)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(vs_cov, z_plot, "-", label="CoV(Vs)")
    ax3.plot(vsz_cov, z_plot, "-", label="CoV(Vs_z)")
    ax3.invert_yaxis()
    ax3.set_xlabel("CoV (std/mean)")
    ax3.set_ylabel("Depth z (m)")
    ax3.set_title("Uncertainty summary (CoV)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(n_prof, z_plot, "-")
    ax4.invert_yaxis()
    ax4.set_xlabel("N profiles")
    ax4.set_ylabel("Depth z (m)")
    ax4.set_title("Coverage after DOI mask")
    ax4.grid(True, alpha=0.25)

    fig.suptitle(str(title_prefix), y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


title_full = "Synthetic MASW demo - ensemble summary (full depth)" if SYNTHETIC_MODE else "MASW demo - ensemble summary (full depth)"
fig_full = plot_ensemble_summary_4panel(
    title_prefix=title_full,
    z_plot=z,
    vs_ens=vs_ens_m,
    vs_p10=vs_p10,
    vs_p90=vs_p90,
    vs_med=vs_med,
    vsz_ens=vsz_ens_m,
    vsz_p10=vsz_p10,
    vsz_p90=vsz_p90,
    vsz_med=vsz_med,
    vs_cov=vs_cov,
    vsz_cov=vsz_cov,
    n_prof=n_prof,
)
fig_full_path = OUT_DIR / "ensemble_summary_4panel_full.png"
fig_full.savefig(fig_full_path, dpi=200)
plt.show()

maskz = z <= Z_ZOOM
zz = z[maskz]

title_zoom = "Synthetic MASW demo - ensemble summary (zoom)" if SYNTHETIC_MODE else "MASW demo - ensemble summary (zoom)"
fig_zoom = plot_ensemble_summary_4panel(
    title_prefix=title_zoom,
    z_plot=zz,
    vs_ens=vs_ens_m[:, maskz],
    vs_p10=vs_p10[maskz],
    vs_p90=vs_p90[maskz],
    vs_med=vs_med[maskz],
    vsz_ens=vsz_ens_m[:, maskz],
    vsz_p10=vsz_p10[maskz],
    vsz_p90=vsz_p90[maskz],
    vsz_med=vsz_med[maskz],
    vs_cov=vs_cov[maskz],
    vsz_cov=vsz_cov[maskz],
    n_prof=n_prof[maskz],
)
fig_zoom_path = OUT_DIR / "ensemble_summary_4panel_zoom.png"
fig_zoom.savefig(fig_zoom_path, dpi=200)
plt.show()
