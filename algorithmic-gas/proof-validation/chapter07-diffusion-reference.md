# Chapter 7 scalar diffusion rate experiment

This experiment evolves the specified spatial diffusion reference from `prop-halo-density` in Rust. It uses absorbing zero endpoints, a positive sampled sine initial density, and the actual explicit finite-difference update

\[
f_i^{n+1}=f_i^n+\tfrac14(f_{i-1}^n-2f_i^n+f_{i+1}^n),
\qquad \Delta t=\frac{\Delta x^2}{4D_0}.
\]

The experiment covers nine pairs of \(D_0\in\{0.01,1,3\}\) and \(L\in\{0.3,1,5\}\), each on 32, 64 and 128 cells. Every one of the 1,025 time states of each law is retained losslessly. The measured rate is \(-\log(m_{1024}/m_0)/(1024\Delta t)\), computed from evolved mass; the continuum prediction is \(D_0\pi^2/L^2\). No exponential output profile is substituted for time evolution.

The discretization allowance is explicit. Write \(\theta=\pi/J\), \(u=\theta/2\). The sine mode has the stencil multiplier \(1-\sin^2u=\cos^2u\). Consequently

\[
\frac{\lambda_{\mathrm{grid}}}{D_0\pi^2/L^2}
=\frac{-2\log\cos u}{u^2}.
\]

For \(0\le t\le u\), \(\tan t\ge t\), while \(\tan s=\int_0^s\sec^2v\,dv\le s\sec^2u\). Therefore

\[
0\le \tan t-t=\int_0^t\tan^2s\,ds
\le \frac{\sec^4u}{3}t^3,
\]

and integration gives

\[
1\le\frac{-2\log\cos u}{u^2}
\le1+\frac{\sec^4u}{6}u^2
=1+\frac{\theta^2}{24\cos^4(\theta/2)}.
\]

Since \(J\ge32\), this is bounded by \(1+\theta^2\), the deliberately conservative allowance used by the executable. The separate arithmetic tolerance is \(10^{-11}\) times the continuum rate. The experiment also checks the normalized profile's L1 change and the actual centered finite-difference source balance of the proposed parabola \(s x(L-x)/(2D_0)\) for \(s=0.2,2\).

These are reference diffusion updates, not Fractal Gas updates or a proof that an interacting swarm QSD is a sine density. The experiment adds 27,648 reference time updates and 108 comparisons; its source, compiler flags, executable SHA, compressed and decoded data SHAs are in `diffusion-reference-v1/provenance.json`.
