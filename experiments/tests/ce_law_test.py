"""Tests the cross-entropy law's first-order facts on a synthetic linear readout, logits = U h. Synthetic data only (no model,
no outputs of the runs); exits on the first failed assert.
Cross-entropy's own measure is f = log p(y_s); experiments/scripts/readout_ce_law.py builds M_f from its gradients and v_f, M_f's top
eigenvector (Appendix C.2, "Cross-entropy ends near its own optimum"). Here grad f is compared with the gradient of the log odds
z = l_s - l_b in two regimes:
1. Two labels carry all the mass (vocabulary of two): grad f = (1 - p_s) grad z exactly, so M_f is M_z with row i weighted by
   (1 - p_s,i), and with equal weights v_f is M_z's top eigenvector.
2. A third token carries the mass and p_b is tiny: grad f is nearly grad l_s - grad l_other, so its component along grad l_b
   is p_b's size.
Usage: python -I experiments/tests/ce_law_test.py
"""
import torch

torch.manual_seed(0)
D = 16
# 1. two labels
U = torch.randn(2, D, dtype=torch.float64)            # rows: y_b (0), y_s (1)
for _ in range(5):
    h = torch.randn(D, dtype=torch.float64, requires_grad=True)
    lp = torch.log_softmax(U @ h, -1)
    gf, = torch.autograd.grad(lp[1], h)
    ps = lp[1].exp().item()
    gz = U[1] - U[0]
    assert torch.allclose(gf, (1 - ps) * gz, atol=1e-12), "1: grad f is not (1 - p_s) grad z"
# equal weights: v_f = v_z
n = 50
H = torch.randn(n, D, dtype=torch.float64); Dl = torch.randn(n, D, dtype=torch.float64)
Wz = (U[1] - U[0]).expand(n, D) + 0.3 * torch.randn(n, D, dtype=torch.float64)
w = 0.7
Mz = 0.5 * (Wz.T @ Dl + Dl.T @ Wz) / n; Mf = 0.5 * ((w * Wz).T @ Dl + Dl.T @ (w * Wz)) / n
vz, vf = torch.linalg.eigh(Mz)[1][:, -1], torch.linalg.eigh(Mf)[1][:, -1]
assert abs(abs(vz @ vf) - 1) < 1e-10, "1: equal weights change the top"
print("1 ok: with two labels grad f = (1 - p_s) grad z; equal weights leave the top unchanged")
# 2. a third token carries the mass
U3 = torch.randn(3, D, dtype=torch.float64)            # rows: y_b (0), y_s (1), other (2)
h = torch.randn(D, dtype=torch.float64)
bias = torch.tensor([-8.0, 0.0, 6.0], dtype=torch.float64)   # p_b tiny, the other token dominant
h = h.requires_grad_(True)
lg = U3 @ h + bias; lp = torch.log_softmax(lg, -1)
gf, = torch.autograd.grad(lp[1], h)
p = lp.exp().detach()
expect = (1 - p[1]) * U3[1] - p[0] * U3[0] - p[2] * U3[2]
assert torch.allclose(gf, expect, atol=1e-12)
coef_b = p[0].item()
assert coef_b < 1e-3, coef_b
print(f"2 ok: with the mass on a third token, grad f weighs grad l_b by p_b = {coef_b:.1e} (the log odds by 1)")
print("all cross-entropy-law checks passed")
