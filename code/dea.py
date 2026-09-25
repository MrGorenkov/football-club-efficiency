"""DEA (output orientation, CRS/VRS) via scipy.optimize.linprog and the Simar-Wilson (1998) smoothed bootstrap.

Notation: X is (n, m) inputs, Y is (n, s) outputs. `farrell_output` returns the Farrell output measure
delta >= 1 (the factor by which all outputs could be expanded at the observed inputs); efficiency is 1 / delta.
"""
import numpy as np
from scipy.optimize import linprog


def farrell_output(X, Y, rts="vrs", Xref=None, Yref=None):
    """Output-oriented Farrell measure of each row of (X, Y) against the technology spanned by (Xref, Yref).

    LP for unit o:  max phi  s.t.  sum_j lam_j x_ij <= x_io,  sum_j lam_j y_rj >= phi y_ro,  lam >= 0,
    and sum_j lam_j = 1 under VRS.
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    Xref = X if Xref is None else np.atleast_2d(Xref)
    Yref = Y if Yref is None else np.atleast_2d(Yref)
    n, m = X.shape
    k = Xref.shape[0]
    s = Y.shape[1]
    c = np.zeros(k + 1)
    c[0] = -1.0
    a_in = np.hstack([np.zeros((m, 1)), Xref.T])
    out = np.empty(n)
    a_eq = np.hstack([[[0.0]], np.ones((1, k))]) if rts == "vrs" else None
    b_eq = [1.0] if rts == "vrs" else None
    for o in range(n):
        a_out = np.hstack([Y[o][:, None], -Yref.T])
        res = linprog(c, A_ub=np.vstack([a_in, a_out]), b_ub=np.concatenate([X[o], np.zeros(s)]),
                      A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        out[o] = -res.fun if res.status == 0 else np.nan
    return out


def farrell_output_1d(x, y, rts="vrs", xref=None, yref=None):
    """Closed form for one input and one output (used for speed in the bootstrap; checked against the LP in tests).

    VRS with free disposability: the frontier at x0 is the maximum of the concave hull of the reference points with
    input <= x0. CRS: the frontier is the ray through the reference point with the highest y/x.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    xref = x if xref is None else np.asarray(xref, float)
    yref = y if yref is None else np.asarray(yref, float)
    if rts == "crs":
        return (x * np.max(yref / xref)) / y
    order = np.lexsort((-yref, xref))
    xs, ys = xref[order], yref[order]
    # running max so the frontier is non-decreasing (free disposal of input)
    hull = []
    for px, py in zip(xs, ys):
        if hull and py <= hull[-1][1]:
            continue
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (px - x1) <= (py - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((px, py))
    hx = np.array([h[0] for h in hull])
    hy = np.array([h[1] for h in hull])
    front = np.interp(x, hx, hy, left=np.nan, right=hy[-1])
    return front / y


def _bandwidth(delta):
    """Normal-reference bandwidth on the reflected sample {delta, 2 - delta} (Silverman rule, as in Simar & Wilson 1998);
    efficient units (delta == 1) are dropped before computing it, as recommended by Simar & Wilson (2000)."""
    d = delta[delta > 1 + 1e-9]
    if len(d) < 3:
        d = delta
    r = np.concatenate([d, 2 - d])
    iqr = np.subtract(*np.percentile(r, [75, 25]))
    spread = min(r.std(ddof=1), iqr / 1.349) if iqr > 0 else r.std(ddof=1)
    return 0.9 * spread * len(r) ** (-0.2)


def sw98_bootstrap(X, Y, rts="vrs", B=2000, alpha=0.05, rng=None):
    """Simar-Wilson (1998) smoothed homogeneous bootstrap for output-oriented DEA.

    Returns dict with delta_hat, bias, delta_bc (bias-corrected), ci_low, ci_high (for delta), boot_sd, h.
    """
    rng = np.random.default_rng(rng)
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    n = X.shape[0]
    one_d = X.shape[1] == 1 and Y.shape[1] == 1
    score = (lambda x, y, xr, yr: farrell_output_1d(x[:, 0], y[:, 0], rts, xr[:, 0], yr[:, 0])) if one_d else \
        (lambda x, y, xr, yr: farrell_output(x, y, rts, xr, yr))
    d_hat = score(X, Y, X, Y)
    h = _bandwidth(d_hat)
    var = d_hat.var(ddof=1)
    boot = np.empty((B, n))
    for b in range(B):
        beta = rng.choice(d_hat, size=n, replace=True)
        t = beta + h * rng.standard_normal(n)
        t = np.where(t < 1, 2 - t, t)
        d_star = beta.mean() + (t - beta.mean()) / np.sqrt(1 + h ** 2 / var)
        d_star = np.maximum(d_star, 1.0)
        Y_star = Y * (d_hat / d_star)[:, None]
        boot[b] = score(X, Y, X, Y_star)
    bias = boot.mean(0) - d_hat
    diff = boot - d_hat
    lo_q, hi_q = np.quantile(diff, [alpha / 2, 1 - alpha / 2], axis=0)
    return {"delta_hat": d_hat, "bias": bias, "delta_bc": d_hat - bias, "boot_sd": boot.std(0, ddof=1),
            "ci_low": d_hat - hi_q, "ci_high": d_hat - lo_q, "h": h}
