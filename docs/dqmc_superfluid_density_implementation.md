# DQMC implementation note: superfluid density, \(T_c\), and optional \(\sigma_{\rm dc}\)

This note summarizes the simulation observables needed to reproduce the
superfluid-density method used in
`/Users/cosdis/Desktop/projects/CE_GCE/reference/PhysRevB.105.184502.pdf`
for the 2D attractive Hubbard model.

Primary target:

\[
\rho_s(T)
\]

Then determine the KT/BKT superconducting transition temperature from

\[
\rho_s(T_c)=\frac{2T_c}{\pi}.
\]

No analytic continuation is required for \(\rho_s\) or for the KT crossing.

---

## 1. Current operator

For the attractive Hubbard model with nearest-neighbor hopping \(t\), define
the local \(x\)-bond current

\[
j_x(\mathbf r)
=
i t
\sum_\sigma
\left(
c^\dagger_{\mathbf r+\hat{x},\sigma}c_{\mathbf r,\sigma}
-
c^\dagger_{\mathbf r,\sigma}c_{\mathbf r+\hat{x},\sigma}
\right).
\]

The imaginary-time evolved current is

\[
j_x(\mathbf r,\tau)=e^{H\tau}j_x(\mathbf r)e^{-H\tau}.
\]

This is the same operator as Eq. (14) in the paper.

Define the Fourier current

\[
J_x(\mathbf q,\tau)
=
\sum_{\mathbf r}
e^{i\mathbf q\cdot \mathbf r}
j_x(\mathbf r,\tau).
\]

Use periodic boundary conditions, \(N=L^2\), and momenta

\[
q_\alpha = \frac{2\pi n_\alpha}{L}.
\]

---

## 2. Static current-current response

Use the per-site static response

\[
\Lambda_{xx}(\mathbf q,0)
=
\frac{1}{N}
\int_0^\beta d\tau\,
\left\langle
J_x(\mathbf q,\tau)J_x(-\mathbf q,0)
\right\rangle.
\]

With DQMC time step \(\Delta\tau\), \(\beta=M\Delta\tau\), approximate the
integral as

\[
\Lambda_{xx}(\mathbf q,0)
\approx
\frac{\Delta\tau}{N}
\sum_{m=0}^{M-1}
\left\langle
J_x(\mathbf q,\tau_m)J_x(-\mathbf q,0)
\right\rangle_{\rm MC}.
\]

For better statistics, average over all possible imaginary-time origins:

\[
\Lambda_{xx}(\mathbf q,0)
\approx
\frac{\Delta\tau}{N M}
\sum_{\tau_0}
\sum_{m=0}^{M-1}
\left\langle
J_x(\mathbf q,\tau_0+\tau_m)J_x(-\mathbf q,\tau_0)
\right\rangle_{\rm MC}.
\]

In a DQMC implementation, this is an unequal-time bilinear-bilinear correlator.
For a fixed Hubbard-Stratonovich field configuration the fermions are free, so
the estimator is evaluated by Wick contractions of unequal-time Green's
functions. Many DQMC codes already have this as a generic current-current or
bond-bond correlator.

---

## 3. Longitudinal and transverse responses

The paper defines

\[
\Lambda^L
=
\lim_{q_x\to 0}
\Lambda_{xx}(q_x,q_y=0,0),
\]

\[
\Lambda^T
=
\lim_{q_y\to 0}
\Lambda_{xx}(q_x=0,q_y,0).
\]

On a finite lattice use the smallest nonzero momentum

\[
q_{\min}=\frac{2\pi}{L}.
\]

Then estimate

\[
\Lambda^L_L
\approx
\Lambda_{xx}(q_{\min},0,0),
\]

\[
\Lambda^T_L
\approx
\Lambda_{xx}(0,q_{\min},0).
\]

The finite-size superfluid density in the paper's convention is

\[
\boxed{
\rho_s(L,T)
=
\frac{1}{4}
\left[
\Lambda_{xx}(q_{\min},0,0)
-
\Lambda_{xx}(0,q_{\min},0)
\right].
}
\]

This is Eq. (9) of the paper, with finite-size momenta replacing the
\(q\to0\) limits.

---

## 4. Equivalent diamagnetic-minus-paramagnetic form

The longitudinal response may also be replaced by the diamagnetic kinetic
energy in the \(x\) direction:

\[
K_x
=
-t
\sum_{\mathbf r,\sigma}
\left(
c^\dagger_{\mathbf r+\hat{x},\sigma}c_{\mathbf r,\sigma}
+
c^\dagger_{\mathbf r,\sigma}c_{\mathbf r+\hat{x},\sigma}
\right).
\]

With the same per-site normalization,

\[
\Lambda^L \rightarrow \frac{\langle -K_x\rangle}{N}.
\]

Thus

\[
\boxed{
\rho_s(L,T)
\approx
\frac{1}{4}
\left[
\frac{\langle -K_x\rangle}{N}
-
\Lambda_{xx}(0,q_{\min},0)
\right].
}
\]

This version is often numerically convenient. It is a useful implementation
check to compare

\[
\Lambda_{xx}(q_{\min},0,0)
\quad \text{against} \quad
\frac{\langle -K_x\rangle}{N}.
\]

They should approach each other as \(L\to\infty\), up to finite-size and
statistical errors.

---

## 5. KT/BKT transition temperature

For each temperature, compute \(\rho_s(L,T)\). Then plot both

\[
\rho_s(L,T)
\]

and

\[
\frac{2T}{\pi}.
\]

The crossing gives an estimate of \(T_c\):

\[
\boxed{
\rho_s(T_c)=\frac{2T_c}{\pi}
}
\]

or equivalently

\[
\boxed{
T_c=\frac{\pi}{2}\rho_s(T_c^-).
}
\]

In the paper this is Eq. (15), and the crossing method is shown in Fig. 3.

Recommended workflow:

1. Choose fixed \(U/t\) and target density \(\langle n\rangle\).
2. Tune \(\mu\) for each \(T\) if working at fixed density.
3. Run DQMC for several \(L\), e.g. \(L=10,12,14,16\).
4. Measure \(\Lambda_{xx}(q_{\min},0,0)\) and \(\Lambda_{xx}(0,q_{\min},0)\).
5. Compute \(\rho_s(L,T)\).
6. Locate the crossing with \(2T/\pi\).
7. Use bootstrap/binning to estimate the uncertainty in the crossing.

For high-precision finite-size scaling near the KT transition, one may include
the standard Weber-Minnhagen correction,

\[
\rho_s(L,T_c)
=
\frac{2T_c}{\pi}
\left[
1+\frac{1}{2\ln L+C}
\right],
\]

with \(C\) fitted. The paper mainly uses the weak \(L\)-dependence of
\(\rho_s\) and crossings on accessible lattice sizes.

---

## 6. Pseudocode

```text
input: L, beta, dtau, U, mu
N = L * L
M = beta / dtau
qmin = 2*pi / L

accumulate Lambda_L_bins
accumulate Lambda_T_bins
accumulate optional C_mid_bins

for each Monte Carlo bin:
    C_L = 0
    C_T = 0
    C_q0_tau = array length M  # optional, for sigma_dc estimator

    for each measured HS configuration in bin:
        # Build/obtain unequal-time Green functions.
        # Use Wick contractions to measure current-current correlators.

        for tau_origin in selected_time_origins:
            for m in 0 .. M-1:
                C_L += < Jx((qmin,0), tau_origin+m)
                         Jx((-qmin,0), tau_origin) >

                C_T += < Jx((0,qmin), tau_origin+m)
                         Jx((0,-qmin), tau_origin) >

                # Optional normal-state dc conductivity estimator:
                C_q0_tau[m] += < Jx((0,0), tau_origin+m)
                                 Jx((0,0), tau_origin) >

    Lambda_L_bin = dtau * C_L / (N * number_of_time_origins * number_of_configs)
    Lambda_T_bin = dtau * C_T / (N * number_of_time_origins * number_of_configs)

    rho_s_bin = 0.25 * (Lambda_L_bin - Lambda_T_bin)

    # Optional:
    C_mid_bin = C_q0_tau[M/2] / (N * number_of_time_origins * number_of_configs)
    sigma_dc_est_bin = beta^2 / pi * C_mid_bin

average rho_s_bin over bins
estimate error bar from bins/bootstrap
```

If \(M\) is odd, estimate the midpoint correlator by interpolating the two
time slices closest to \(\tau=\beta/2\).

---

## 7. Optional: midpoint estimator for \(\sigma_{\rm dc}\)

The true dc conductivity is a real-frequency limit,

\[
\sigma_{\rm dc}=\lim_{\omega\to0}\sigma'(\omega),
\]

so it formally requires analytic continuation. However, a common DQMC
approximation avoids full analytic continuation by using the midpoint
imaginary-time correlator:

\[
\boxed{
\sigma_{\rm dc}^{\rm est}
\approx
\frac{\beta^2}{\pi}
\Lambda_{xx}(\mathbf q=0,\tau=\beta/2).
}
\]

Here

\[
\Lambda_{xx}(\mathbf 0,\tau)
=
\frac{1}{N}
\left\langle
J_x(\mathbf 0,\tau)J_x(\mathbf 0,0)
\right\rangle.
\]

Reason: the spectral relation is

\[
\Lambda_{xx}(\beta/2)
=
\int_0^\infty
\frac{d\omega}{\pi}
\frac{\omega\,\sigma'(\omega)}
{\sinh(\beta\omega/2)}.
\]

If \(\sigma'(\omega)\) is smooth for \(|\omega|\lesssim T\), then

\[
\Lambda_{xx}(\beta/2)
\approx
\frac{\pi}{\beta^2}\sigma_{\rm dc}.
\]

Important limitations:

- This is only an estimator, not an exact dc conductivity.
- It is best used in the normal state, especially above \(T_c\).
- It can fail for a narrow Drude peak, a gap/pseudogap, strong finite-size
  effects, or a superconducting delta function.
- Below \(T_c\), the clean superconducting system has an infinite dc response
  associated with the zero-frequency delta function, so the midpoint estimator
  should not be interpreted as a normal-state dc conductivity.

---

## 8. Normalization and units checklist

Use a single convention throughout.

Common paper convention:

- \(t=1\)
- lattice spacing \(a=1\)
- \(k_B=1\)
- \(\hbar=1\)
- \(e=1\)
- \(N=L^2\)
- \(\rho_s\) has units of energy, i.e. units of \(t\)

Critical factor:

\[
\rho_s=\frac{1}{4}\left[\Lambda^L-\Lambda^T\right].
\]

If your code defines \(\Lambda\) without the per-site \(1/N\), divide by \(N\)
before using the formula above. If your code uses a different stiffness
convention without the \(1/4\), adjust the KT crossing criterion consistently.

---

## 9. Basic validation checks

1. **High temperature:** \(\rho_s\) should be near zero within errors.
2. **Low temperature away from half filling:** \(\rho_s\) should become positive.
3. **Finite size:** crossings of \(\rho_s(L,T)\) with \(2T/\pi\) should drift
   weakly for sufficiently large \(L\).
4. **Half filling in 2D attractive Hubbard:** finite-temperature \(T_c\) should
   vanish because CDW and superconducting correlations are degenerate and the
   Mermin-Wagner theorem forbids finite-\(T\) long-range order.
5. **Longitudinal check:** \(\Lambda_{xx}(q_{\min},0,0)\) should be compatible
   with \(\langle -K_x\rangle/N\) in the large-\(L\) limit.
6. **Paper benchmark:** for \(\langle n\rangle=0.5\), \(U/t=-5\), the paper
   finds \(T_c \approx 0.150\) from the \(\rho_s\) crossing and
   \(T_c \approx 0.152\) from KT finite-size scaling of the pair structure
   factor.

