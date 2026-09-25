"""Stochastic production frontiers estimated by maximum likelihood (scipy.optimize).

Model: y_i = x_i'b + v_i - u_i,  v ~ N(0, s_v^2),  u >= 0.
  - "halfnormal": u ~ |N(0, s_u^2)|  (Aigner, Lovell & Schmidt 1977); efficiency E[exp(-u)|e] (Battese & Coelli 1988)
  - "bc95": u ~ N+(z_i'd, s_u^2) truncated at zero (Battese & Coelli 1995), parametrised as in FRONTIER 4.1:
    sigma^2 = s_u^2 + s_v^2, gamma = s_u^2 / sigma^2.
True fixed effects (Greene 2005) = "halfnormal" with a full set of unit dummies in X.
"""
import numpy as np
from scipy import optimize, stats
from scipy.special import log_ndtr


def _unpack(theta, k, q):
    b = theta[:k]
    d = theta[k:k + q]
    sigma2 = np.exp(theta[k + q])
    gamma = 1 / (1 + np.exp(-theta[k + q + 1]))
    return b, d, sigma2, gamma


def loglik_obs(theta, y, X, Z, model):
    k, q = X.shape[1], (0 if Z is None else Z.shape[1])
    b, d, sigma2, gamma = _unpack(theta, k, q)
    e = y - X @ b
    su2, sv2 = gamma * sigma2, (1 - gamma) * sigma2
    s = np.sqrt(sigma2)
    mu = np.zeros_like(e) if model == "halfnormal" else Z @ d
    mu_star = (sv2 * mu - su2 * e) / sigma2
    s_star = np.sqrt(su2 * sv2 / sigma2)
    return (-0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((e + mu) / s) ** 2
            + log_ndtr(mu_star / s_star) - log_ndtr(mu / np.sqrt(su2)))


def grad_halfnormal(theta, y, X):
    """Analytic gradient of the half-normal log-likelihood in (b, log sigma^2, logit gamma); lambda = exp(t / 2)."""
    k = X.shape[1]
    b, s, t = theta[:k], theta[k], theta[k + 1]
    sigma = np.exp(s / 2)
    lam = np.exp(t / 2)
    e = y - X @ b
    a = -lam * e / sigma
    m = np.exp(stats.norm.logpdf(a) - log_ndtr(a))
    g_b = X.T @ (e / sigma ** 2 + m * lam / sigma)
    g_s = np.sum(-0.5 + 0.5 * (e / sigma) ** 2 - 0.5 * a * m)
    g_t = np.sum(0.5 * a * m)
    return np.concatenate([g_b, [g_s, g_t]])


def fit(y, X, Z=None, model="halfnormal", start=None, maxiter=20000):
    """ML fit. X and Z must contain their own constant columns. Returns a dict with estimates, SEs, efficiencies."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    Z = None if Z is None else np.asarray(Z, float)
    k, q = X.shape[1], (0 if Z is None else Z.shape[1])
    if start is None:
        b_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b_ols
        var = r.var()
        # method-of-moments shift of the intercept for a half-normal u with sigma_u^2 ~ var / 2
        start = np.concatenate([b_ols, np.zeros(q), [np.log(var)], [0.0]])
        if model == "bc95" and q:
            start[k] = 0.0
    f = lambda t: -loglik_obs(t, y, X, Z, model).sum()
    jac = (lambda t: -grad_halfnormal(t, y, X)) if model == "halfnormal" else None
    best = None
    methods = ("BFGS", "Nelder-Mead", "BFGS") if len(start) <= 30 else ("BFGS", "BFGS")
    for method in methods:
        res = optimize.minimize(f, start if best is None else best.x, method=method,
                                jac=jac if method == "BFGS" else None,
                                options={"maxiter": maxiter, "gtol": 1e-7} if method == "BFGS" else
                                {"maxiter": maxiter * 5, "xatol": 1e-9, "fatol": 1e-10, "adaptive": True})
        if best is None or res.fun < best.fun - 1e-10:
            best = res
    theta = best.x
    hess = _num_hessian_grad(jac, theta) if jac is not None else _num_hessian(f, theta)
    try:
        cov_t = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov_t = np.full((len(theta), len(theta)), np.nan)
    b, d, sigma2, gamma = _unpack(theta, k, q)
    # delta method from (log sigma2, logit gamma) to (sigma2, gamma)
    J = np.eye(len(theta))
    J[k + q, k + q] = sigma2
    J[k + q + 1, k + q + 1] = gamma * (1 - gamma)
    cov = J @ cov_t @ J.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    est = np.concatenate([b, d, [sigma2, gamma]])
    e = y - X @ b
    su2, sv2 = gamma * sigma2, (1 - gamma) * sigma2
    mu = np.zeros_like(e) if model == "halfnormal" else Z @ d
    mu_star = (sv2 * mu - su2 * e) / sigma2
    s_star = np.sqrt(su2 * sv2 / sigma2)
    te = np.exp(-mu_star + 0.5 * s_star ** 2) * stats.norm.cdf(mu_star / s_star - s_star) / stats.norm.cdf(mu_star / s_star)
    # JLMS conditional mean of u
    u_hat = mu_star + s_star * stats.norm.pdf(mu_star / s_star) / stats.norm.cdf(mu_star / s_star)
    return {"est": est, "se": se, "cov": cov, "loglik": -best.fun, "te": te, "u_hat": u_hat, "theta": theta,
            "converged": bool(best.success) or np.all(np.isfinite(se)), "k": k, "q": q, "resid": e}


def _num_hessian(f, x, eps=1e-4):
    n = len(x)
    h = eps * np.maximum(np.abs(x), 1.0)
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


def _num_hessian_grad(g, x, eps=1e-6):
    n = len(x)
    H = np.empty((n, n))
    for i in range(n):
        h = eps * max(abs(x[i]), 1.0)
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        H[i] = (g(xp) - g(xm)) / (2 * h)
    return 0.5 * (H + H.T)


def ols_loglik(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = (r ** 2).mean()
    return -0.5 * len(y) * (np.log(2 * np.pi * s2) + 1)


def resid_skewness(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return stats.skew(y - X @ b)
