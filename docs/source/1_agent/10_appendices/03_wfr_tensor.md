(sec-appendix-c-wfr-stress-energy-tensor)=
# {ref}`Appendix C <sec-appendix-c-wfr-stress-energy-tensor>`: WFR Stress-Energy Tensor (Full Derivation)

## TLDR

- This appendix expands the full variational derivation of the WFR stress-energy tensor used in the geometry chapters.
- Use it as reference when implementing WFR-consistency losses or when auditing the variational principles.

This appendix provides the full derivation of Theorem {prf:ref}`thm-wfr-stress-energy-tensor-variational-form`.

(sec-appendix-c-setup)=
## C.1 Setup

Recall the {prf:ref}`def-the-wfr-action`:

$$
\mathcal{S}_{\mathrm{WFR}} = \frac12\int_0^T\int_{\mathcal{Z}} \rho\left(\|v\|_G^2+\lambda^2 r^2\right)\,d\mu_G\,ds,

$$
with the continuity equation enforced separately:

$$
\partial_s\rho+\nabla\!\cdot(\rho v)=\rho r.

$$
Define the Lagrangian density

$$
\mathcal{L}_{\mathrm{WFR}}:=\frac12\,\rho\left(\|v\|_G^2+\lambda^2 r^2\right).

$$
We vary the metric $G^{ij}$ while holding $(\rho, v, r)$ fixed as fields.

(sec-appendix-c-metric-variation)=
## C.2 Metric variation

Write the kinetic term using covariant components:

$$
\|v\|_G^2 = G_{ij} v^i v^j.

$$
Under a variation of the inverse metric, $\delta G^{ij}$, we have

$$
\delta G_{ij} = -G_{ia}G_{jb}\,\delta G^{ab},

$$
so

$$
\begin{aligned}
\delta\|v\|_G^2
&= v^i v^j\,\delta G_{ij} \\
&= -v_i v_j\,\delta G^{ij}.
\end{aligned}

$$
The volume form varies as

$$
\delta d\mu_G = -\frac12\,G_{ij}\,\delta G^{ij}\,d\mu_G.

$$
Combine these:

$$
\delta\left(\sqrt{|G|}\,\mathcal{L}_{\mathrm{WFR}}\right)
= \sqrt{|G|}\left[
-\frac12\,\rho v_i v_j\,\delta G^{ij}
-\frac12\,\mathcal{L}_{\mathrm{WFR}}\,G_{ij}\,\delta G^{ij}
\right].

$$
Therefore,

$$
\delta\mathcal{S}_{\mathrm{WFR}}
= -\frac12\int_0^T\int_{\mathcal{Z}}
\left(\rho v_i v_j + \mathcal{L}_{\mathrm{WFR}} G_{ij}\right)
\delta G^{ij}\,d\mu_G\,ds.

$$
By definition,

$$
T^{\mathrm{WFR}}_{ij}:=
-\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\mathrm{WFR}})}{\delta G^{ij}},

$$
so we identify

$$
T^{\mathrm{WFR}}_{ij}=\rho v_i v_j + \mathcal{L}_{\mathrm{WFR}} G_{ij}.

$$
(sec-appendix-c-perfect-fluid-form-and-pressure-split)=
## C.3 Perfect-fluid form and pressure split

Let

$$
P:=\mathcal{L}_{\mathrm{WFR}}
=\frac12\,\rho\left(\|v\|_G^2+\lambda^2 r^2\right).

$$
Then

$$
T^{\mathrm{WFR}}_{ij}=\rho v_i v_j + P G_{ij},

$$
which is the perfect-fluid form in Riemannian signature. The reaction contribution is

$$
P_{\mathrm{react}}=\frac12\,\lambda^2\rho r^2,

$$
and the transport contribution is $P_{\mathrm{trans}}=\tfrac12\rho\|v\|_G^2$.

(sec-appendix-c-relation-to-the-metric-law)=
## C.4 Relation to the metric law

This auxiliary tensor is not substituted into the capacity-constrained metric
law: that theorem uses the reward Risk Tensor from
{prf:ref}`def-extended-risk-tensor`. A combined source would require an explicit
coupling and a unit conversion. The metric law remains the separate
curvature--risk stationarity identity with source $T^{\mathrm{risk}}_{ij}$.
This completes the derivation of the auxiliary WFR tensor in Theorem
{prf:ref}`thm-wfr-stress-energy-tensor-variational-form`.
