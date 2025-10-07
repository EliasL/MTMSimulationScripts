import powerlaw
from matplotlib import pyplot as plt
import numpy as np
import ruptures as rpt


def _segment_with_ruptures(y, min_size=3, pen=None, model="l2"):
    y2 = np.asarray(y, float).reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=min_size).fit(y2)

    n = len(y)
    if pen is None:
        # conservative penalty for very small n
        mad = np.median(np.abs(y - np.median(y)))
        sigma = 1.4826 * mad + 1e-12
        pen = 4.0 * sigma**2 * np.log(max(n, 2))
    bkps = algo.predict(pen=pen)  # end-exclusive indices
    starts = [0] + bkps[:-1]
    return list(zip(starts, bkps)), bkps


def _seg_slope(x, y, i0, i1):
    xi, yi = x[i0:i1], y[i0:i1]
    if len(xi) < 2 or np.allclose(xi, xi[0]):
        return 0.0
    vx = np.var(xi)
    if vx == 0:
        return 0.0
    return np.cov(xi, yi, bias=True)[0, 1] / vx


def _seg_range(y, i0, i1):
    yi = y[i0:i1]
    return float(np.max(yi) - np.min(yi)) if len(yi) else np.inf


def active_plateau_search(
    x,
    y,
    f_eval,  # initial data + expensive function
    domain,  # (xmin, xmax)
    budget=8,  # max extra samples
    min_size=3,
    slope_tol=None,
    range_tol=None,
    probe_boundaries_every=3,  # occasionally test expansion
    verbose=False,
):
    """
    Iteratively: segment with ruptures -> pick flattest segment -> sample a midpoint
    (or boundary) -> repeat. Returns final plateau guess and augmented data.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # ensure sorted by x
    order = np.argsort(x)
    x, y = x[order], y[order]

    def adaptive_tols(xv, yv):
        nonlocal slope_tol, range_tol
        if slope_tol is None:
            if len(xv) >= 3:
                dx, dy = np.diff(xv), np.diff(yv)
                sl = dy[np.nonzero(dx)] / dx[np.nonzero(dx)]
                base = np.median(np.abs(sl)) if sl.size else 0.0
                slope = 3.0 * base if base > 0 else 1e-12
            else:
                slope = 1e-12
        else:
            slope = slope_tol
        if range_tol is None:
            m = np.median(np.abs(yv - np.median(yv)))
            rng = (3.0 * m) if m > 0 else 0.02 * max(1e-12, np.ptp(yv))
        else:
            rng = range_tol
        return slope, rng

    it = 0
    sample_log = []
    while budget > 0:
        segs, bkps = _segment_with_ruptures(y, min_size=min_size)

        st, rt = adaptive_tols(x, y)

        # pick “best plateau” segment: first try those passing both tests
        candidates = []
        for i0, i1 in segs:
            if i1 - i0 < min_size:
                continue
            s = abs(_seg_slope(x, y, i0, i1))
            r = _seg_range(y, i0, i1)
            if s <= st and r <= rt:
                candidates.append((i0, i1))

        if candidates:
            # prefer longest by x-span
            i0, i1 = max(candidates, key=lambda ij: x[ij[1] - 1] - x[ij[0]])
        else:
            # fall back to the **flattest** (smallest |slope|), then smallest range
            i0, i1 = min(
                segs,
                key=lambda ij: (
                    abs(_seg_slope(x, y, ij[0], ij[1])),
                    _seg_range(y, ij[0], ij[1]),
                ),
            )

        # choose next x to sample
        Xseg = x[i0:i1]
        if len(Xseg) >= 2:
            gaps = np.diff(Xseg)
            k = int(np.argmax(gaps)) if len(gaps) else 0
            xm = (
                (Xseg[k] + Xseg[k + 1]) / 2.0
                if len(gaps)
                else np.mean([Xseg[0], Xseg[-1]])
            )
        else:
            # only one point in segment: probe center of domain or neighbor
            xm = np.clip(Xseg[0], *domain) if len(Xseg) else np.mean(domain)

        # occasional boundary probe to see if the plateau expands
        if (it % probe_boundaries_every == 0) and len(Xseg) >= 2:
            left_gap = Xseg[0] - max(domain[0], x[i0 - 1] if i0 > 0 else domain[0])
            right_gap = min(domain[1], x[i1] if i1 < len(x) else domain[1]) - Xseg[-1]
            if right_gap > left_gap and right_gap > 0:
                xm = Xseg[-1] + 0.5 * right_gap
            elif left_gap > 0:
                xm = Xseg[0] - 0.5 * left_gap

        # avoid duplicates / near-duplicates
        if np.any(np.isclose(xm, x, rtol=0, atol=1e-12)):
            # jitter a tiny bit within domain
            xm = float(np.clip(xm + 1e-6 * max(1.0, np.ptp(x) or 1.0), *domain))

        if verbose:
            print(f"[iter {it}] proposing xm={xm:.6g}")

        # evaluate and insert
        ym = f_eval(xm)

        if verbose:
            print(f"[iter {it}]  f(xm)={ym:.6g}")
        sample_log.append((float(xm), float(ym)))

        x = np.insert(x, np.searchsorted(x, xm), xm)
        y = np.insert(y, np.searchsorted(x, xm), ym)

        budget -= 1
        it += 1

    # final plateau decision
    segs, _ = _segment_with_ruptures(y, min_size=min_size)
    st, rt = adaptive_tols(x, y)

    final = []
    for i0, i1 in segs:
        if i1 - i0 < min_size:
            continue
        s = abs(_seg_slope(x, y, i0, i1))
        r = _seg_range(y, i0, i1)
        if s <= st and r <= rt:
            final.append((i0, i1))

    if final:
        i0, i1 = max(final, key=lambda ij: x[ij[1] - 1] - x[ij[0]])
        interval = (x[i0], x[i1 - 1])
        idx_span = (i0, i1)
    else:
        interval, idx_span = None, None

    return {
        "plateau_interval": interval,
        "segment_indices": idx_span,
        "x": x,
        "y": y,
        "samples": np.array(sample_log),
    }


# expensive function (example)
def f(x):
    return 1.0 + 0.0005 * x + 0.02 * np.exp(-(((x - 50) / 5) ** 2))


x0 = np.linspace(0, 100, 100)
y0 = np.array([f(t) for t in x0])

res = active_plateau_search(
    x0,
    y0,
    f_eval=f,
    domain=(0.0, 100.0),
    budget=6,  # at most 6 extra evaluations
    min_size=3,
    slope_tol=1e-4,
    range_tol=5e-3,
    probe_boundaries_every=10,
    verbose=True,
)

print("Plateau:", res["plateau_interval"])
from matplotlib import pyplot as plt

plt.plot(res["x"], res["y"], label="piecewise path")
plt.scatter(x0, y0, s=30, label="initial")
if res.get("samples") is not None and len(res["samples"]) > 0:
    xs = res["samples"][:, 0]
    ys = res["samples"][:, 1]
    plt.scatter(xs, ys, marker="x", s=60, label="new samples")

if res["plateau_interval"] is not None:
    xl, xr = res["plateau_interval"]
    plt.axvspan(xl, xr, alpha=0.15, label="plateau")

plt.legend()
plt.tight_layout()
plt.show()
