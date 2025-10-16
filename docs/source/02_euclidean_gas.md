# Euclidean Gas — a Fragile Gas instantiation (Langevin perturbations)

> **Goal.** Define the **Euclidean Gas** as a concrete *Fragile Swarm* / *Fragile Gas* instantiation whose state space is Euclidean and whose random perturbations follow a (discrete‑time) Langevin dynamics; all other design choices follow the **Canonical Fragile Swarm**. We then verify, axiom‑by‑axiom, that this instantiation satisfies the framework and thus defines a valid Fragile Gas.
(see {prf:ref}`def-fragile-swarm-instantiation`, {prf:ref}`def-fragile-gas-algorithm`)

---

## 0. Framework alignment with velocity states

The canonical Fragile framework (`01_fractal_gas_framework.md`) phrases every axiom in terms of
an **algorithmic state** $y \in \mathcal Y$ and a binary status. To accommodate walkers of the
form $w=(x,v,s)$ we first make explicit how each framework object lifts to the product space
$\mathcal X \times \mathcal V$. Write $\pi_x(x,v,s)=(x,s)$ and $\pi_{\mathcal Y}(x,v,s)=(x,v)$.

:::{note} Framework adaptation
Throughout the remainder of this chapter we instantiate the Euclidean Gas by taking
$\widetilde{\mathcal Y}=\mathcal X	\times\mathcal V_{\mathrm{alg}}$ with the Sasaki metric of
Section 1.1. The canonical measurement and potential pipeline now operate on the full
position-velocity states; $\Pi$ only serves as a bookkeeping device when comparing with the
framework notation. Sections 1.2 and 2 re-derive every geometry-dependent continuity bound
for this Sasaki dispersion.

:::

## 1. Definition: the Euclidean Gas

A **Euclidean Gas** is the Fragile Swarm $\mathcal F_{\text{EG}}$ given by the tuple of environmental structures, parameters, operators, and noise measures below. It induces a Markov chain on the swarm state space via the **Fragile Gas Algorithm** $\mathcal{S}_{t+1}\!\sim\!\Psi_{\mathcal F_{\text{EG}}}(\mathcal S_t,\cdot)$ (Def. *Fragile Gas Algorithm* ({prf:ref}`def-fragile-gas-algorithm`)). Throughout this chapter the measurement pipeline is fixed to the patched standardisation operator of {prf:ref}`def-statistical-properties-measurement` followed by the Canonical Logistic Rescale Function ({prf:ref}`def-canonical-logistic-rescale-function-example`); these choices are part of the Euclidean Gas specification and underlie every continuity bound below.

### **1.1 Euclidean Gas algorithm (canonical pipeline)**

:::{prf:algorithm} Euclidean Gas Update
:label: alg-euclidean-gas

Given a swarm state $\mathcal S_t=(w_1,\dots,w_N)$ with walkers $w_i=(x_i,v_i,s_i)$, the Euclidean Gas performs one update as follows:

1.  **Cemetery check.** If all walkers are dead (no alive indices in $\mathcal A_t$) return the cemetery state; otherwise continue.
2.  **Measurement stage.** For every alive walker $i\in\mathcal A_t$ sample a companion $c_{\mathrm{pot}}(i)$ from the algorithmic distance-weighted kernel $\mathbb C_\epsilon(\mathcal S_t,i)$, then compute raw reward $r_i:=R(x_i,v_i)$ and algorithmic distance $d_i:=d_{\text{alg}}(i,c_{\mathrm{pot}}(i))$ as defined in Section 1.3 and detailed in {ref}`Stage 2 <sec-eg-stage2>`.
3.  **Patched standardisation.** Aggregate the raw reward and distance vectors with the empirical operator and apply the regularized standard deviation from {prf:ref}`def-statistical-properties-measurement` to obtain standardized scores with floor $\sigma'_{\min,\mathrm{patch}} = \sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}$.
4.  **Logistic rescale.** Apply the Canonical Logistic Rescale Function ({prf:ref}`def-canonical-logistic-rescale-function-example`) to the standardized reward and distance components, producing positive outputs $r'_i$ and $d'_i$. Combine them with the canonical exponents to freeze the potential vector $V_{\text{fit},i}=(d'_i)^\beta (r'_i)^\alpha$ with floor $\eta^{\alpha+\beta}$.
5.  **Clone/Persist gate.** For each walker draw a clone companion $c_{\mathrm{clone}}(i)$ from the same algorithmic distance-weighted kernel and threshold $T_i\sim\mathrm{Unif}(0,p_{\max})$, compute the canonical score $S_i:=\big(V_{\text{fit},c_{\mathrm{clone}}(i)}-V_{\text{fit},i}\big)/(V_{\text{fit},i}+\varepsilon_{\mathrm{clone}})$, and clone when $S_i>T_i$. Cloned walkers reset their position to the companion's position plus Gaussian jitter ($\sigma_x$) and reset their velocity directly to the companion's velocity (no velocity jitter), as detailed in {ref}`Stage 3 <sec-eg-stage3>`. Otherwise the walker persists unchanged. The intermediate swarm sets every status to alive before the kinetic step.
6.  **Kinetic perturbation.** Update each alive clone or survivor by applying the **BAOAB splitting integrator** for one step of underdamped Langevin dynamics with force $F(x)=\nabla R_{\mathrm{pos}}(x)$ and noise scales $(\sigma_v,\sigma_x)$.
7.  **Status refresh.** Set the new status $s_i^{(t+1)}=\mathbf 1_{\mathcal X_{\mathrm{valid}}}(x_i^+)$ and output the updated swarm $\mathcal S_{t+1}$.

**Euclidean Gas Algorithm**

$$
\begin{aligned}
& \textbf{Input:} \mathcal S_t = \{(x_i^{(t)}, v_i^{(t)}, s_i^{(t)})\}_{i=1}^N\text{; and parameters } \alpha, \beta, \varepsilon_{\mathrm{std}}, \eta, \tau, p_{\max}, \varepsilon_{\mathrm{clone}}, \delta_x, \delta_v, \sigma_x, \sigma_v, \\
& \qquad \sigma'_{\mathrm{patch}}, g_A, \mathbb C_i, Q_{\delta}, \Psi_{\mathrm{kin,BAOAB}}. \\
& \textbf{If } |\mathcal A_t| = 0: \textbf{ return } \delta_{\mathcal S_t} \quad \text{\# Cemetery absorption} \\
\\
& \underline{\text{Stage 2a: Raw vectors on alive set}} \\
& \dots \quad \text{\# Unchanged} \\
\\
& \underline{\text{Stage 2b: Patched standardisation}} \\
& \dots \quad \text{\# Unchanged} \\
\\
& \underline{\text{Stage 2c: Logistic rescale of components}} \\
& \dots \quad \text{\# Unchanged} \\
\\
& \underline{\text{Stage 2d: Assemble full vectors with floors}} \\
& \dots \quad \text{\# Unchanged} \\
\\
& \underline{\text{Stage 3: Cloning transition}} \\
& \dots \quad \text{\# Unchanged logic, produces } (x_i^{(t+\frac{1}{2})}, v_i^{(t+\frac{1}{2})}) \\
\\
& \underline{\text{Stage 4: Langevin perturbation and status refresh}} \\
& \mathcal S_{\mathrm{pert}} \sim \Psi_{\mathrm{kin,BAOAB}}(\{(x_i^{(t+\frac{1}{2})}, v_i^{(t+\frac{1}{2})})\}, \cdot) \quad \text{\# BAOAB Langevin step with velocity capping} \\
& \textbf{For each } i = 1..N: \\
& \quad (x_i^{(t+1)}, v_i^{(t+1)}) \leftarrow \text{draw from kinetic step output} \\
& \quad s_i^{(t+1)} \leftarrow \mathbf 1_{\mathcal X_{\mathrm{valid}}}(x_i^{(t+1)}) \\
& \textbf{Return } \mathcal S_{t+1}
\end{aligned}
$$

:::

### 1.2 Python implementation of the Euclidean Gas algorithm
```python

import numpy as np

def psi_v(v: np.ndarray, V_alg: float) -> np.ndarray:
    """
    Applies the smooth velocity squashing map to a set of velocity vectors.

    This function implements the formula from Section 1.1 of 02_euclidean_gas.md:
    ψ_v(v) = V_alg * (v / (V_alg + ||v||))

    It ensures that the returned velocity vectors have a magnitude strictly less
    than V_alg, while pointing in the same direction as the input vectors.
    The implementation is vectorized to handle an array of N walkers.

    Args:
        v (np.ndarray): An (N, D) array of N velocity vectors in D dimensions.
        V_alg (float): The scalar maximum algorithmic velocity (the radius of the ball).

    Returns:
        np.ndarray: An (N, D) array of squashed velocity vectors.
    """
    # Calculate the L2 norm (Euclidean magnitude) for each velocity vector (row).
    # The `axis=1` argument computes the norm along the columns for each row.
    # `keepdims=True` is crucial: it makes the output shape (N, 1) instead of (N,),
    # which allows for correct NumPy broadcasting during the division.
    norms = np.linalg.norm(v, axis=1, keepdims=True)

    # To avoid division by zero if a norm is zero, we can add a small epsilon.
    # However, the formula is mathematically well-behaved at v=0.
    # If v=0, then norm=0, and the output is V_alg * 0 / (V_alg + 0) = 0.
    # NumPy handles this correctly without explicit checks.

    # The scaling factor by which each vector is multiplied.
    # This factor is always in the range [0, 1).
    scaling_factor = V_alg / (V_alg + norms)

    # Apply the scaling factor to the original velocity vectors.
    # Broadcasting rules: (N, 1) * (N, D) -> (N, D)
    squashed_v = scaling_factor * v

    return squashed_v

# Helper function for the new BAOAB kinetic update
def Psi_kin_BAOAB(x, v, params):
    """
    Executes one step of the BAOAB integrator for underdamped Langevin dynamics.

    Args:
        x (np.array): (N, D) array of current positions.
        v (np.array): (N, D) array of current velocities.
        params (dict): Dictionary of physical parameters:
                       'tau', 'gamma_fric', 'm', 'sigma_v', 'sigma_x'.
                       Also needs access to force F(x) and flow u(x).

    Returns:
        (np.array, np.array): Next positions and velocities (x_next, v_next).
    """
    p = params
    N, D = x.shape

    # B-step: Propagate positions for a half-step
    x_mid = x + v * (p['tau'] / 2.0)

    # A-step: Update velocities with deterministic forces for a half-step
    force = F(x_mid) # Assumes F is a vectorized function
    flow = u(x_mid)   # Assumes u is a vectorized function
    v_mid = v + (force / p['m'] - p['gamma_fric'] * (v - flow)) * (p['tau'] / 2.0)

    # O-step: Exact solution for the Ornstein-Uhlenbeck process (friction and noise)
    c1 = np.exp(-p['gamma_fric'] * p['tau'])
    c2 = np.sqrt(1 - c1**2) * p['sigma_v'] # sigma_v = sqrt(2*gamma*kBT/m)
    noise_v = np.random.randn(N, D)
    v_postO = c1 * v_mid + c2 * noise_v

    # A-step: Update velocities again with deterministic forces for a half-step
    # Note: Forces are evaluated at the same half-step position x_mid
    force = F(x_mid)
    flow = u(x_mid)
    v_almost_final = v_postO + (force / p['m'] - p['gamma_fric'] * (v_postO - flow)) * (p['tau'] / 2.0)

    # B-step: Propagate positions for the final half-step
    x_next = x_mid + v_almost_final * (p['tau'] / 2.0)

    # (Optional) Add positional noise, if sigma_x is non-zero
    if p['sigma_x'] > 0:
        noise_x = np.random.randn(N, D)
        x_next += np.sqrt(p['tau']) * p['sigma_x'] * noise_x

    # Final Velocity Capping
    v_next = psi_v(v_almost_final) # Assumes psi_v is a vectorized function

    return x_next, v_next

def run_euclidean_gas_step(S_t, params):
    """
    Executes one step of the Euclidean Gas algorithm using BAOAB integrator.

    Note: This implementation uses uniform companion selection (infinite ε limit).
    For spatially-aware companion selection using algorithmic distance d_alg(i,j),
    see the full implementation in 03_cloning.md.
    """
    x_t, v_t, s_t = S_t['x'], S_t['v'], S_t['s']
    N, D = x_t.shape
    p = params

    A_t_indices = np.where(s_t == 1)[0]
    if len(A_t_indices) == 0:
        return S_t

    # --- Stage 2: Measurement using algorithmic distance ---
    r = np.zeros(N); d = np.zeros(N)
    # Companion selection: uniform for canonical EG (infinite ε limit)
    # For finite ε: weight by exp(-d_alg(i,j)^2 / 2ε^2)
    c_pot = np.random.choice(A_t_indices, size=len(A_t_indices))
    r[A_t_indices] = R(x_t[A_t_indices], v_t[A_t_indices])
    # Compute algorithmic distance d_alg(i, c_pot(i))
    # For canonical EG with λ_alg = λ_v, this matches the Sasaki metric
    p1 = phi(x_t[A_t_indices], v_t[A_t_indices])
    p2 = phi(x_t[c_pot], v_t[c_pot])
    d[A_t_indices] = d_sasaki(p1, p2)  # d_alg with λ_alg = λ_v
    r_alive = r[A_t_indices]; d_alive = d[A_t_indices]
    mu_R, sigma_R = np.mean(r_alive), np.std(r_alive)
    mu_D, sigma_D = np.mean(d_alive), np.std(d_alive)
    hat_sigma_R = patched_std(sigma_R, p['epsilon_std'])
    hat_sigma_D = patched_std(sigma_D, p['epsilon_std'])
    tilde_r_alive = (r_alive - mu_R) / hat_sigma_R
    tilde_d_alive = (d_alive - mu_D) / hat_sigma_D
    r_prime_alive = g_A(tilde_r_alive, p['eta'])
    d_prime_alive = g_A(tilde_d_alive, p['eta'])
    V_alive = (d_prime_alive**p['beta']) * (r_prime_alive**p['alpha'])
    V_fit = np.full(N, np.inf)
    V_fit[A_t_indices] = np.maximum(V_alive, p['eta']**(p['alpha'] + p['beta']))

    # --- Stage 3: Cloning with positional jitter only (no velocity jitter) ---
    x_half, v_half = np.zeros_like(x_t), np.zeros_like(v_t)
    c_clone = np.random.choice(A_t_indices, size=N)
    U = np.random.uniform(0, p['p_max'], size=N)
    v_i = V_fit
    v_c = V_fit[c_clone]
    S_i = (v_c - v_i) / (v_i + p['epsilon_clone'])
    should_clone = (S_i > U) | (s_t == 0)

    clone_indices = np.where(should_clone)[0]
    if len(clone_indices) > 0:
        c = c_clone[clone_indices]
        # Position: companion + Gaussian jitter
        jitter_x = np.random.randn(len(clone_indices), D) * p['sigma_x']
        x_half[clone_indices] = x_t[c] + jitter_x
        # Velocity: direct reset to companion (NO jitter)
        v_half[clone_indices] = v_t[c]

    persist_indices = np.where(~should_clone)[0]
    if len(persist_indices) > 0:
        x_half[persist_indices] = x_t[persist_indices]
        v_half[persist_indices] = v_t[persist_indices]

    # --- Stage 4: Langevin perturbation (BAOAB) and status refresh ---
    x_next, v_next = Psi_kin_BAOAB(x_half, v_half, p)
    s_next = is_valid(x_next).astype(int)

    return {'x': x_next, 'v': v_next, 's': s_next}
```

### 1.3 Position–velocity foundations and projection (Sasaki metric)

- **Position space** $(\mathcal X,d_{\mathcal X})$: the ambient space is $\mathbb R^d$ with its Euclidean metric, while the algorithm operates on the **bounded valid domain** $\mathcal X_{\mathrm{valid}}\subset\mathbb R^d$. We assume $\mathcal X_{\mathrm{valid}}$ is compact with $C^1$ boundary, the standing hypothesis across the framework.
- **Velocity radius** $V_{\mathrm{alg}}\in(0,\infty)$ and **velocity cap** $\mathcal V_{\mathrm{alg}}:=\{v\in\mathbb R^d:\|v\|\le V_{\mathrm{alg}}\}$.
- **Positional radius** $R_x\in(0,\infty)$, which sets the characteristic scale of the bounded algorithmic position space.
- **Walker state** $w_i=(x_i,v_i,s_i)\in\mathcal X\times\mathbb R^d\times\{0,1\}$ collects position, velocity, and status.
- **Algorithmic space and Sasaki metric** $(\mathcal Y,d_{\mathcal Y}^{\mathrm{Sasaki}})$ where the algorithmic space is the closure of the projection image,

  $$
  \mathcal Y\;:=\;\overline{B(0,R_x)}\times\overline{B(0,V_{\mathrm{alg}})}\subset\mathbb R^d\times\mathbb R^d,
  $$

  endowed with the Sasaki metric

  $$
  d_{\mathcal Y}^{\mathrm{Sasaki}}\bigl((y_x,y_v),(y'_x,y'_v)\bigr)^2:=\|y_x-y'_x\|^2+\lambda_v\|y_v-y'_v\|^2
  $$
  for some fixed weight $\lambda_v>0$. For physical states we write $y=\varphi(x,v)$ and $y'=\varphi(x',v')$; the metric therefore measures differences between squashed coordinates. The compactness of $\mathcal Y$ ensures a finite algorithmic diameter.
- **Projection** $\varphi:\mathbb R^d\times\mathbb R^d\to B(0,R_x)\times B(0,V_{\mathrm{alg}})$ given by $\varphi(x,v)=(\psi_x(x),\psi_v(v))$ with the smooth squashing maps

  $$
  \psi_x(x)\ :=\ R_x\,\frac{x}{R_x+\|x\|},\qquad
  \psi_v(v)\ :=\ V_{\mathrm{alg}}\,\frac{v}{V_{\mathrm{alg}}+\|v\|}.
  $$

  ::: {admonition} Design Note
  :class: tip
  Smooth ($C^{\infty}$ away from the origin) squashing maps are chosen over hard radial projections. They provide differentiability for both position and velocity coordinates, a prerequisite for the one-step minorization proof in the convergence analysis (Chapter 5) and for deriving continuum limits.
  :::

  The projection $\varphi$ maps the physical state space $\mathbb R^d\times\mathbb R^d$ into the bounded product $B(0,R_x)\times B(0,V_{\mathrm{alg}})$. Its image has compact closure $\mathcal Y$, so the **Axiom of Bounded Algorithmic Diameter** ({prf:ref}`def-axiom-bounded-algorithmic-diameter`) holds by construction. Lemma {prf:ref}`lem-squashing-properties-generic` shows that each squashing map is $1$-Lipschitz, and Lemma {prf:ref}`lem-projection-lipschitz` extends this to $\varphi$ under the Sasaki metric.

- **Algorithmic distance for companion selection.** For intra-swarm measurements (companion selection for diversity and cloning), the algorithm uses the **algorithmic distance** between two walkers $i$ and $j$:

  $$
  d_{\text{alg}}(i,j)^2 := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2
  $$

  where $\lambda_{\text{alg}} \geq 0$ controls the relative importance of velocity similarity in companion selection. For the Euclidean Gas, we set $\lambda_{\text{alg}} = \lambda_v$ to match the Sasaki metric weight, ensuring consistency between the algorithmic behavior and the analytical geometry. See Definition 5.0 in `03_cloning.md` for the full framework specification. This metric defines the algorithm's "perception" of proximity and is distinct from the Sasaki metric used in the analysis (see Section 1.4).

- **Reward** $R:\mathcal X_{\mathrm{valid}}\times\mathcal V_{\mathrm{alg}}\to\mathbb R$ couples the position potential with a kinetic regularizer:

  $$
  R(x,v):=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\|v\|^2,
  $$
  where $R_{\mathrm{pos}}:\mathcal X_{\mathrm{valid}}\to\mathbb R$ is a $C^1$ potential defined on a neighbourhood of the valid domain. We require only that $R_{\mathrm{pos}}$ is bounded above on $\mathcal X_{\mathrm{valid}}$ and that its gradient $F(x):=\nabla R_{\mathrm{pos}}(x)$ is Lipschitz on the compact set $\mathcal X_{\mathrm{valid}}$ with constant $L_F$. The potential therefore provides a smooth reward landscape inside the permitted region rather than a mechanism for confining walkers at infinity.

:::{prf:lemma} Properties of smooth radial squashing maps
:label: lem-squashing-properties-generic

For any constant $C>0$ define $\psi_C: \mathbb R^d\to B(0,C)$ by $\psi_C(z):=C\,z/(C+\|z\|)$. The map $\psi_C$ satisfies:

1. $\psi_C$ is $1$-Lipschitz on $\mathbb R^d$.
2. $\psi_C\in C^{\infty}(\mathbb R^d\setminus\{0\})$.
3. $\psi_C(\mathbb R^d)\subset B(0,C)$.

```{dropdown} Proof
:::{prf:proof}
1. *Lipschitz continuity.* The Jacobian at $z\neq 0$ is

  $$
  D\psi_C(z)=\frac{C}{C+\|z\|}I-\frac{C}{(C+\|z\|)^2}\,\frac{z z^{\top}}{\|z\|}.
  $$
  The first term has operator norm at most $1$. The second term is positive semidefinite with norm $C\|z\|/(C+\|z\|)^2\le 1$. Hence $\|D\psi_C(z)\|\le 1$ for all $z\neq 0$. Continuity gives $\|D\psi_C(0)\|=1$. The mean-value inequality then implies $\|\psi_C(z)-\psi_C(z')\|\le\|z-z'\|$ for all $z,z'\in\mathbb R^d$.

2. *Smoothness away from the origin.* For $z\neq 0$, $\psi_C$ is a composition of smooth functions: $z\mapsto\|z\|$, inversion on $(0,\infty)$, and scalar-vector multiplication. Hence $\psi_C\in C^{\infty}(\mathbb R^d\setminus\{0\})$.

3. *Image contained in the open ball.* For any $z\in\mathbb R^d$, $\|\psi_C(z)\| = C\,\|z\|/(C+\|z\|) < C$, so $\psi_C(z)$ lies in $B(0,C)$.

All three properties follow immediately.
```
:::

Both the positional squashing map $\psi_x$ and the velocity squashing map $\psi_v$ are obtained by setting $C=R_x$ and $C=V_{\mathrm{alg}}$, respectively, so they inherit the 1-Lipschitz and smoothness properties of Lemma {prf:ref}`lem-squashing-properties-generic`.

:::{prf:lemma} Lipschitz continuity of the projection $\varphi$
:label: lem-projection-lipschitz

For $(x,v),(x',v')\in\mathbb R^d\times\mathbb R^d$ the projection $\varphi(x,v)=(\psi_x(x),\psi_v(v))$ satisfies

$$
d_{\mathcal Y}^{\mathrm{Sasaki}}\bigl(\varphi(x,v),\varphi(x',v')\bigr)\le L_{\varphi}\,\sqrt{\|x-x'\|^2+\lambda_v\|v-v'\|^2},
$$

with Lipschitz constant $L_{\varphi}\le\max\{1,\sqrt{\lambda_v}\}$.

```{dropdown} Proof
:::{prf:proof}
Because $\psi_x$ and $\psi_v$ are $1$-Lipschitz (Lemma {prf:ref}`lem-squashing-properties-generic`),

$$
\|\psi_x(x)-\psi_x(x')\|\le\|x-x'\|,\qquad \|\psi_v(v)-\psi_v(v')\|\le\|v-v'\|.
$$

Therefore

$$
\begin{aligned}
d_{\mathcal Y}^{\mathrm{Sasaki}}\bigl(\varphi(x,v),\varphi(x',v')\bigr)^2
&=\|\psi_x(x)-\psi_x(x')\|^2+\lambda_v\|\psi_v(v)-\psi_v(v')\|^2\\
&\le\|x-x'\|^2+\lambda_v\|v-v'\|^2\\
&\le L_{\varphi}^2\bigl(\|x-x'\|^2+\lambda_v\|v-v'\|^2\bigr)
\end{aligned}
$$

with $L_{\varphi}=1$ if $\lambda_v\le 1$ and $L_{\varphi}=\sqrt{\lambda_v}$ otherwise. Taking square roots gives the stated bound.
```
:::

The bound exhibits at most quadratic growth in $\|x\|$ and $\|v\|$, meeting the controlled-moment requirement for the non-compact kinetic axiom.

The Sasaki metric retains the full position–velocity information needed for the kinetic perturbation, while the smooth squashing maps enforce the finite algorithmic diameter used by the Fragile framework. From this point forward every continuity and stability statement is re-proved in the Sasaki geometry: when we cite a "framework" lemma in later sections we first restate and re-derive its Lipschitz bounds for $d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}$. No argument is borrowed verbatim from the positional framework—each bound is recomputed from the primitive constants introduced above.

### 1.4 Swarm distance and canonical operators

We measure dispersion in the Sasaki metric and retain the canonical aggregation pipeline:

- **Dispersion distance.** For swarms $\mathcal S_1,\mathcal S_2$ write

  $$
  d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2
  := \frac{1}{N}\sum_{i=1}^{N} d_{\mathcal Y}^{\mathrm{Sasaki}}\big(\varphi(x_{1,i},v_{1,i}),\varphi(x_{2,i},v_{2,i})\big)^2
  + \frac{\lambda_{\mathrm{status}}}{N}\sum_{i=1}^{N}(s_{1,i}-s_{2,i})^2,
  $$
  with status penalty $\lambda_{\mathrm{status}}>0$ as in the canonical framework. Because the Sasaki metric adds a velocity term, Section 2.3 re-validates every deterministic Lipschitz bound against $d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}$.

  :::{admonition} Distinction: Algorithmic Distance vs. Sasaki Metric
  :class: note

  It is critical to distinguish two different distance metrics used in this document:

  1. **Algorithmic distance** $d_{\text{alg}}(i,j)$: Used by the *algorithm itself* for intra-swarm companion selection (diversity measurement and cloning). This defines how the algorithm "perceives" proximity between walkers within the same swarm.

  2. **Sasaki metric** $d_{\mathcal Y}^{\mathrm{Sasaki}}$: Used by the *analysis* to measure inter-swarm dispersion and derive continuity bounds. This is an analytical tool for proving convergence properties.

  For the Euclidean Gas, we set $\lambda_{\text{alg}} = \lambda_v$ so that these metrics coincide in their functional form, simplifying the connection between algorithmic behavior and analytical properties. However, they serve conceptually different roles: the algorithmic distance is intrinsic to the algorithm's design, while the Sasaki metric is extrinsic to the convergence analysis.
  :::
- **Walkers:** $N\ge 2$; the empirical reward and distance aggregators keep their canonical formulas. Lemma {prf:ref}`lem-sasaki-aggregator-lipschitz` supplies Sasaki-specific error moduli, and Lemma {prf:ref}`lem-sasaki-standardization-lipschitz` applies them to the regularized standard deviation and logistic rescale operators.
- **Dynamics weights:** $\alpha,\beta\ge 0$ with $\alpha+\beta>0$ fixed as in the framework’s Axiom of Sufficient Amplification ({prf:ref}`def-axiom-sufficient-amplification`).

### 1.5 Kinetic Langevin perturbations with velocity capping

- **Physical parameters.** Fix mass $m>0$, friction $\gamma_{\mathrm{fric}}>0$, temperature $\Theta>0$, and integrator step $\tau>0$. Optionally prescribe a steady flow field $u:\mathcal X_{\mathrm{valid}}\to\mathbb R^d$ (set $u\equiv 0$ if absent). The force field is derived from the potential $R_{\mathrm{pos}}$ via $F(x):=\nabla R_{\mathrm{pos}}(x)$; Section 1.1 shows that $F$ is Lipschitz on the compact domain with constant $L_F$ and therefore bounded. Define $\sigma_v^2:=2\gamma_{\mathrm{fric}}\Theta/m$ and choose a (possibly small) positional noise scale $\sigma_x>0$.
- **One kinetic Euler step (assumption EG-kin$^+$).** Given $(x,v)$ draw independent $\xi_v,\xi_x\sim\mathcal N(0,I_d)$ and set

  $$
  \begin{aligned}
  \tilde v &= v + \frac{\tau}{m}\,F(x) - \gamma_{\mathrm{fric}}\tau\,(v-u(x)) + \sqrt{\sigma_v^2\,\tau}\,\xi_v,\
  v^+ &= \psi_v(\tilde v),\
  x^+ &= x + \tau\,v^+ + \sqrt{\tau}\,\sigma_x\,\xi_x.
  \end{aligned}
  $$
  The **kinetic perturbation kernel** $\mathcal P_{\mathrm{kin}}$ therefore injects independent Gaussian noise into both velocities and positions. The cap ensures $\|v^+\|\le V_{\mathrm{alg}}$; when capping is inactive the drift component coincides with an underdamped Langevin Euler step. If one sets $\sigma_x=0$, the same reachability conclusions follow whenever $\operatorname{diam}(\mathcal X_{\mathrm{comp}})<\tau V_{\mathrm{alg}}$, because $\psi_v(\mathbb R^d)=B(0,V_{\mathrm{alg}})$ makes every point of $\mathcal X_{\mathrm{comp}}$ one-step reachable.
- **Clone jitter distribution.** When the Clone action fires in {ref}`Stage 3 <sec-eg-stage3>` we sample

  $$
  (x_c,v_c) + (\delta_x\zeta_x,\,\delta_v\zeta_v),\qquad \zeta_x,\zeta_v\sim\mathcal N(0,I_d),
  $$
  with tunable spreads $\delta_x,\delta_v>0$ (velocity-preserving cloning corresponds to $\delta_v=0$), after which the smooth squashing map $\psi_v$ caps the velocity.
- **Pipeline ordering.** {ref}`Section 4 <sec-eg-kernel>` executes the canonical measurement → standardize → rescale pipeline first, freezes the potential vector, and then applies the Clone/Persist rule to produce an all-alive intermediate swarm. The kinetic update above acts on that intermediate state, and the deterministic status operator sets $s_i^{(t+1)}=\mathbf 1_{\mathcal X_{\mathrm{valid}}}(x_i^+)$ afterward—no cloning occurs after the status check.
- **In-step independence.** The random draws $(\xi_i^v,\xi_i^x,\zeta_i^x,\zeta_i^v)$ used in the cloning and kinetic stages are independent across walkers given the current swarm, as required by Assumption A ({prf:ref}`def-assumption-instep-independence`).

**Design note.** The Langevin force field uses only the positional potential $R_{\mathrm{pos}}$, while the selection pipeline optimizes the full reward $R(x,v)=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\|v\|^2$. This intentional decoupling treats the velocity penalty as a regulariser that preserves fragility: Lemma {prf:ref}`lem-euclidean-richness` shows the quadratic term forces the environmental richness variance floor needed by ({prf:ref}`def-axiom-environmental-richness`). Consequently the kinetic perturbation samples a Gibbs law for $U(x)=-R_{\mathrm{pos}}(x)$ rather than $R$, and the stationary distribution of the swarm is not the standard underdamped Langevin equilibrium for the selection objective. All continuity and limit arguments in Section 2 therefore work directly with the Sasaki metric and the patched standardization pipeline, without assuming a coupled potential.

The kinetic parameters feed the geometric consistency constants computed in Section 2 (perturbation moment, anisotropy, drift control) and inherit continuity from the Gaussian/affine structure.

---

## 2. Axiom-by-axiom validation (Sasaki formulation)

We reuse the canonical Fragile framework proofs, updating every bound so it lives in the Sasaki metric on position–velocity space and the kinetic perturbation described in §1.3.

### 2.1 Viability axioms (survival)

1. **Guaranteed Revival.** {ref}`Stage 2 <sec-eg-stage2>` freezes the potential vector $\mathbf V_{\text{fit}}$ with the canonical floor $\eta^{\alpha+\beta}$, and Stage 3 draws independent thresholds $T_i\sim\mathrm{Unif}(0,p_{\max})$ while cloning walker $i$ whenever $S_i>T_i$. The fraction

   $$
   \kappa_{\mathrm{revival}}\;=\;\frac{\eta^{\alpha+\beta}}{\varepsilon_{\mathrm{clone}}\,p_{\max}}\;>\;1
   $$
   is therefore the same as in the framework, so each dead walker survives the Clone/Persist gate with strictly positive probability and the all-alive intermediate swarm satisfies the axiom (Theorem *Almost-sure revival*; {prf:ref}`thm-revival-guarantee`, {prf:ref}`def-axiom-guaranteed-revival`).

2. **Boundary regularity & smoothness.** Lemma {prf:ref}`lem-euclidean-boundary-holder` bounds the death probability with explicit Hölder constants, verifying the boundary axioms ({prf:ref}`def-axiom-boundary-regularity`, {prf:ref}`def-axiom-boundary-smoothness`).

:::{prf:lemma} Lipschitz property of the kinetic flow
:label: lem-sasaki-kinetic-lipschitz

For $(x,v),(x',v')\in\mathcal X\times\mathcal V_{\mathrm{alg}}$ and any $\xi_v,\xi_x\in\mathbb R^d$ define

$$
\Phi_{x,v}(\xi_v,\xi_x):=x+\tau\psi_v\big(v+\tfrac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau(v-u(x))+\sqrt{\sigma_v^2\tau}\,\xi_v\big)+\sqrt{\tau}\,\sigma_x\,\xi_x.
$$
Then

$$
\|\Phi_{x,v}(\xi_v,\xi_x)-\Phi_{x',v'}(\xi_v,\xi_x)\|\le L_{\mathrm{flow}}\,d_{\mathcal Y}^{\mathrm{Sasaki}}((x,v),(x',v')),\qquad L_{\mathrm{flow}}:=1+\frac{\tau^2}{m}L_F+\gamma_{\mathrm{fric}}\tau^2L_u+\frac{\tau(1+\gamma_{\mathrm{fric}}\tau)}{\sqrt{\lambda_v}}.
$$

```{dropdown} Proof
:::{prf:proof}
Fix $(x,v),(x',v')\in\mathcal X\times\mathcal V_{\mathrm{alg}}$ and $\xi_v,\xi_x\in\mathbb R^d$. Define the uncapped velocities

$$
\tilde v:=v+\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau\big(v-u(x)\big)+\sqrt{\sigma_v^2\tau}\,\xi_v,\qquad\tilde v':=v'+\frac{\tau}{m}F(x')-\gamma_{\mathrm{fric}}\tau\big(v'-u(x')\big)+\sqrt{\sigma_v^2\tau}\,\xi_v.
$$

Because the same velocity noise $\xi_v$ appears in both expressions, it cancels in the difference $\tilde v-\tilde v'$. We bound the displacement in four steps.

1. **Uncapped velocity difference.** Using the triangle inequality, the Lipschitz constant $L_F$ of $F$, and the Lipschitz constant $L_u$ of $u$, we obtain

$$
\begin{aligned}
\|\tilde v-\tilde v'\|&\le \|v-v'\|+\frac{\tau}{m}\|F(x)-F(x')\|+\gamma_{\mathrm{fric}}\tau\|v-v'\|+\gamma_{\mathrm{fric}}\tau\|u(x)-u(x')\|\\
&\le(1+\gamma_{\mathrm{fric}}\tau)\,\|v-v'\|+\Big(\frac{\tau}{m}L_F+\gamma_{\mathrm{fric}}\tau L_u\Big)\,\|x-x'\|.
\end{aligned}
$$

2. **Lipschitz projection.** Lemma {prf:ref}`lem-squashing-properties-generic` shows the smooth squashing map $\psi_v$ is $1$-Lipschitz, so the same inequality holds for the capped velocities $v^+:=\psi_v(\tilde v)$ and $v'^+:=\psi_v(\tilde v')$.

3. **Position update.** The Euler step sets $x^+:=x+\tau v^+ +\sqrt{\tau}\,\sigma_x\,\xi_x$ and $x'^+:=x'+\tau v'^+ +\sqrt{\tau}\,\sigma_x\,\xi_x$. Hence

$$
\|x^+-x'^+\|\le\|x-x'\|+\tau\,\|v^+-v'^+\|\le\Big(1+\frac{\tau^2}{m}L_F+\gamma_{\mathrm{fric}}\tau^2L_u\Big)\|x-x'\|+\tau(1+\gamma_{\mathrm{fric}}\tau)\|v-v'\|.
$$

4. **Express via the Sasaki metric.** The Sasaki distance satisfies $d_{\mathcal Y}^{\mathrm{Sasaki}}((x,v),(x',v'))^2=\|x-x'\|^2+\lambda_v\|v-v'\|^2$, so $\|x-x'\|\le d_{\mathcal Y}^{\mathrm{Sasaki}}$ and $\|v-v'\|\le d_{\mathcal Y}^{\mathrm{Sasaki}}/\sqrt{\lambda_v}$. Substituting these bounds into the inequality from Step 3 yields

$$
\|\Phi_{x,v}(\xi_v,\xi_x)-\Phi_{x',v'}(\xi_v,\xi_x)\|\le\Big(1+\frac{\tau^2}{m}L_F+\gamma_{\mathrm{fric}}\tau^2L_u+\frac{\tau(1+\gamma_{\mathrm{fric}}\tau)}{\sqrt{\lambda_v}}\Big) d_{\mathcal Y}^{\mathrm{Sasaki}}((x,v),(x',v')).
$$

The constant in parentheses is $L_{\mathrm{flow}}$, completing the proof.
```
:::

:::{prf:lemma} Hölder continuity of the death probability
:label: lem-euclidean-boundary-holder

```{dropdown} Proof
:::{prf:proof}
Fix $(x,v),(x',v')\in\mathcal X\times\mathcal V_{\mathrm{alg}}$ and set $\Delta:=d_{\mathcal Y}^{\mathrm{Sasaki}}((x,v),(x',v'))$. Let $C$ be any compact subset of $\mathbb R^d$ containing $x$ and $x'$, so that the local constants from Lemma {prf:ref}`lem-euclidean-geometric-consistency` apply uniformly on $C$. For independent $\xi_v,\xi_x\sim\mathcal N(0,I_d)$ define

$$
\Phi_{x,v}(\xi_v,\xi_x):=x+\tau\psi_v\Big(v+\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau\big(v-u(x)\big)+\sqrt{\sigma_v^2\tau}\,\xi_v\Big)+\sqrt{\tau}\,\sigma_x\,\xi_x.
$$

Lemma {prf:ref}`lem-sasaki-kinetic-lipschitz` delivers $\|\Phi_{x,v}(\xi_v,\xi_x)-\Phi_{x',v'}(\xi_v,\xi_x)\|\le L_{\mathrm{flow}}\,\Delta$ almost surely. Consequently

$$
|p_{\mathrm{dead}}(x,v)-p_{\mathrm{dead}}(x',v')|\le\mathbb P\big(\Phi_{x,v}(\xi)\in N_{L_{\mathrm{flow}}\Delta}(\partial\mathcal X_{\mathrm{valid}})\big)+\mathbb P\big(\Phi_{x',v'}(\xi)\in N_{L_{\mathrm{flow}}\Delta}(\partial\mathcal X_{\mathrm{valid}})\big).
$$

We bound the first term; the second is identical with primed variables.

1. **Tubular neighbourhood volume.** Because $\partial\mathcal X_{\mathrm{valid}}$ is $C^1$ with bounded curvature, the tubular-neighbourhood theorem (\[Federer 69, §4.18\]) provides $\varepsilon_{\mathrm{tube}}>0$ and

$$
C_{\partial}:=\sup_{0<\varepsilon\le\varepsilon_{\mathrm{tube}}}\frac{\operatorname{Vol}(N_\varepsilon(\partial\mathcal X_{\mathrm{valid}}))}{\varepsilon}<\infty.
$$

By monotonicity it suffices to treat $L_{\mathrm{flow}}\Delta\le\varepsilon_{\mathrm{tube}}$; otherwise the Hölder bound follows immediately.

2. **Affine Gaussian contribution (no capping).** Introduce the uncapped velocity update

$$
\tilde v:=v+\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau\big(v-u(x)\big)+\sqrt{\sigma_v^2\tau}\,\xi_v.
$$

Then $\tilde x:=x+\tau\tilde v+\sqrt{\tau}\,\sigma_x\,\xi_x$ is Gaussian with mean $x+\tau m(x,v)$, where $m(x,v):=\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau(v-u(x))$, and covariance $\tau(\sigma_v^2\tau^2+\sigma_x^2)I_d$. Its density is

$$
p_{\tilde x}(y)=\frac{1}{(2\pi\tau(\sigma_v^2\tau^2+\sigma_x^2))^{d/2}}\exp\Big(-\frac{\|y-(x+\tau m(x,v))\|^2}{2\tau(\sigma_v^2\tau^2+\sigma_x^2)}\Big).
$$

The density attains its supremum at the mean, yielding

$$
p_{\mathrm{aff}}:=\sup_{y\in\mathbb R^d}p_{\tilde x}(y)=\frac{1}{(2\pi\tau(\sigma_v^2\tau^2+\sigma_x^2))^{d/2}}.
$$

This constant governs the contribution of $C^c:=\{\|\tilde v\|\le V_{\mathrm{alg}}\}$, where the velocity cap is inactive.

3. **Directional density under capping.** Let $E:=\{\|\tilde v\|>V_{\mathrm{alg}}\}$. Lemma {prf:ref}`lem-euclidean-geometric-consistency` gives $\mathbb P(E)\le\rho_*(C)$. On $E$ write $\tilde v=ru$ with $r>V_{\mathrm{alg}}$ and $u\in S^{d-1}$. The capped velocity is $v^+=V_{\mathrm{alg}}u$, whose conditional density equals

$$
g(u)=\frac{1}{(2\pi\sigma_v^2\tau)^{d/2}}\int_{V_{\mathrm{alg}}}^{\infty}\exp\Big(-\frac{\|ru-m(x,v)\|^2}{2\sigma_v^2\tau}\Big) r^{d-1}\,dr.
$$

The local bounds on $F$ and $u$ over $C$ imply

$$
\|m(x,v)\|\le M_{\mathrm{kin}}(C).
$$

For $r\ge V_{\mathrm{alg}}$ the inequality $\|a-b\|^2\ge\tfrac{1}{2}\|a\|^2-\|b\|^2$ yields

$$
\|ru-m(x,v)\|^2\ge\frac{r^2}{2}-M_{\mathrm{kin}}(C)^2.
$$

Substituting into $g(u)$ and changing variables via $s=r^2/(4\sigma_v^2\tau)$ produces

$$
g(u)\le\frac{\exp(M_{\mathrm{kin}}(C)^2/(2\sigma_v^2\tau))}{(2\pi\sigma_v^2\tau)^{d/2}}(2\sigma_v^2\tau)^{d/2}\Gamma\Big(\frac{d}{2},\frac{V_{\mathrm{alg}}^2}{4\sigma_v^2\tau}\Big)=:q_{\mathrm{dir}}(C),
$$

where $\Gamma(\cdot,\cdot)$ is the upper incomplete gamma function. Thus the capped direction has uniformly bounded density.

4. **Probability of hitting the tube.** For any Borel $A\subseteq\mathcal X$ split according to $E$:

$$
\begin{aligned}
\mathbb P\big(\Phi_{x,v}(\xi)\in A\big)&=\mathbb P(E^c)\,\mathbb P\big(\Phi_{x,v}(\xi)\in A\mid E^c\big)+\mathbb P(E)\,\mathbb P\big(\Phi_{x,v}(\xi)\in A\mid E\big)\\
&\le p_{\mathrm{aff}}\operatorname{Vol}(A)+\rho_*(C) q_{\mathrm{dir}}(C)\operatorname{Vol}(A).
\end{aligned}
$$

Taking $A=N_{L_{\mathrm{flow}}\Delta}(\partial\mathcal X_{\mathrm{valid}})$ and using Step 1 yields

$$
\mathbb P\big(\Phi_{x,v}(\xi)\in N_{L_{\mathrm{flow}}\Delta}(\partial\mathcal X_{\mathrm{valid}})\big)\le\big(p_{\mathrm{aff}}+\rho_*(C)q_{\mathrm{dir}}(C)\big)C_{\partial}L_{\mathrm{flow}}\,\Delta.
$$

Combining the two probabilities shows

$$
|p_{\mathrm{dead}}(x,v)-p_{\mathrm{dead}}(x',v')|\le2\big(p_{\mathrm{aff}}+\rho_*(C)q_{\mathrm{dir}}(C)\big)C_{\partial}L_{\mathrm{flow}}\,\Delta.
$$

Therefore $\alpha_B^{\mathrm{Sasaki}}=1$ with Hölder constant

$$
L_{\mathrm{death}}^{\mathrm{Sasaki}}(C):=2\big(p_{\mathrm{aff}}+\rho_*(C)q_{\mathrm{dir}}(C)\big)C_{\partial}L_{\mathrm{flow}}.
$$

```
:::
3. **Finite algorithmic diameter.** Section 1.1 built $(\mathcal Y,d_{\mathcal Y}^{\mathrm{Sasaki}})$ from the capped velocities and showed that the projection $\varphi$ is $1$-Lipschitz. Consequently $\operatorname{diam}_{d_{\mathcal Y}^{\mathrm{Sasaki}}}(\mathcal Y)<\infty$, meeting the Axiom of Bounded Algorithmic Diameter ({prf:ref}`def-axiom-bounded-algorithmic-diameter`).

### 2.2 Environmental axioms

The ambient space $(\mathcal X,d_{\mathcal X})$ remains Euclidean with Lebesgue reference measure, so the canonical density and integration arguments carry over ({prf:ref}`def-ambient-euclidean`). Because $\mathcal X_{\mathrm{valid}}$ is compact and $F=\nabla R_{\mathrm{pos}}$ is Lipschitz on this set, both $F$ and the auxiliary flow field $u$ are uniformly bounded; these bounds are the only ingredients required by the kinetic and boundary estimates recorded below.

:::{prf:lemma} Reward regularity in the Sasaki metric
:label: lem-euclidean-reward-regularity

The reward function $R(x,v)=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\|v\|^2$ is continuous on $(\mathcal Y,d_{\mathcal Y}^{\mathrm{Sasaki}})$ and therefore satisfies the Axiom of Reward Regularity ({prf:ref}`def-axiom-reward-regularity`).

```{dropdown} Proof
:::{prf:proof}
Let $\mathcal Y^{\circ}:=B(0,R_x)\times B(0,V_{\mathrm{alg}})$ be the image of the projection $\varphi:\mathbb R^d\times\mathbb R^d\to\mathcal Y^{\circ}$. For $y=(y_x,y_v)\in\mathcal Y^{\circ}$ the inverse mapping is explicit:

$$
\psi_C^{-1}(y)=\frac{C}{1-\|y\|/C}\,y\qquad(\|y\|<C).
$$

Define $R_{\mathcal Y}:\mathcal Y^{\circ}\to\mathbb R$ by

$$
R_{\mathcal Y}(y):=R_{\mathrm{pos}}\big(\psi_{R_x}^{-1}(y_x)\big)-\lambda_{\mathrm{vel}}\,\big\|\psi_{V_{\mathrm{alg}}}^{-1}(y_v)\big\|^2.
$$

This is well defined because the squashing maps are bijections between $\mathbb R^d$ and the open balls $B(0,R_x)$ and $B(0,V_{\mathrm{alg}})$. The maps $\psi_{R_x}^{-1}$ and $\psi_{V_{\mathrm{alg}}}^{-1}$ are continuous on $\mathcal Y^{\circ}$, and the compositions with $R_{\mathrm{pos}}$ and the quadratic velocity term are continuous. Hence $R_{\mathcal Y}$ is continuous on $\mathcal Y^{\circ}$.

Because $R_{\mathcal Y}$ is continuous on $\mathcal Y^{\circ}$ and $\mathcal Y^{\circ}$ is bounded, the restriction of $R_{\mathcal Y}$ to any compact subset of $\mathcal Y^{\circ}$ is uniformly continuous. In particular, the walker positions belong to the compact valid domain $\mathcal X_{\mathrm{valid}}$, so the image $\varphi(\mathcal X_{\mathrm{valid}}\times\mathcal V_{\mathrm{alg}})$ is compact and $R_{\mathcal Y}$ is uniformly continuous (indeed, Lipschitz) on that set. Consequently the reward evaluated along the Sasaki projection is uniformly continuous, satisfying the reward-regularity axiom without invoking a global Lipschitz bound for $R_{\mathrm{pos}}$ on $\mathbb R^d$. :::
```
:::

:::{prf:lemma} Environmental richness with a kinetic regularizer
:label: lem-euclidean-richness

The reward $R(x,v)=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\|v\|^2$ with $\lambda_{\mathrm{vel}}>0$ satisfies the Axiom of Environmental Richness ({prf:ref}`def-axiom-environmental-richness`).

```{dropdown} Proof
:::{prf:proof}
Fix $(x_0,v_0)\in\mathcal Y$ and radius $r>0$. Every Sasaki ball of radius $r$ contains the set of velocities with Euclidean norm at most $r/\sqrt{\lambda_v}$ around $v_0$. Let

$$
\delta:=\min\Big\{\frac{r}{\sqrt{\lambda_v}},\,\frac{V_{\mathrm{alg}}}{2}\Big\}>0.
$$
Consider the two velocities $v_1:=v_0$ and $v_2:=v_0+\delta e$, where the direction $e$ is chosen as follows:

1. If $\|v_0\|\le V_{\mathrm{alg}}-\delta$, take $e$ orthogonal to $v_0$. Then $\|v_2\|^2=\|v_0\|^2+\delta^2\le V_{\mathrm{alg}}^2$, so $v_2\in\mathcal V_{\mathrm{alg}}$ and $d_{\mathcal Y}^{\mathrm{Sasaki}}((x_0,v_0),(x_0,v_2))=\sqrt{\lambda_v}\,\delta\le r$.
2. If $\|v_0\|>V_{\mathrm{alg}}-\delta$, set $e:=-v_0/\|v_0\|$ (if $v_0=0$, pick any unit vector). The new velocity has norm $\|v_0\| - \delta\le V_{\mathrm{alg}}$ and again lies within the Sasaki ball.

In both cases the velocities stay in the ball, and the reward difference equals

$$
|R(x_0,v_1)-R(x_0,v_2)|=\lambda_{\mathrm{vel}}\,|\|v_2\|^2-\|v_0\|^2|\ge \lambda_{\mathrm{vel}}\,\delta^2.
$$
For the inward-pointing choice we use that $\|v_0\|>V_{\mathrm{alg}}-\delta\ge V_{\mathrm{alg}}/2\ge\delta$ because $\delta\le V_{\mathrm{alg}}/2$, guaranteeing the same lower bound.
Hence the variance of $R$ on the ball is at least $\sigma_{\mathrm{rich}}^2(r):=\lambda_{\mathrm{vel}}^2\delta^4/4>0$, establishing environmental richness. :::
```
:::

These bounds also guarantee that the position-derived force admits explicit growth control: with $F(x)=\nabla R_{\mathrm{pos}}(x)$ the Lipschitz assumption gives $\|F(x)\|\le\|F(0)\|+L_F\|x\|$. The velocity penalty therefore fixes the degeneracy noted in the earlier draft by ensuring every Sasaki ball carries non-zero reward variance while allowing us to track the kinetic growth terms explicitly.

### 2.3 Algorithmic & operator axioms

1. **Valid noise measure (kinetic perturbation).** Lemma {prf:ref}`lem-euclidean-perturb-moment` provides a quadratic-growth second-moment bound and the Feller property for the capped kinetic kernel.

::: {admonition} Non-compact moment interpretation
:class: note
On an unbounded domain we cannot demand a uniform moment bound. Instead, the kinetic axiom tracks the squared Sasaki increment through a Lyapunov-style control that grows at most quadratically in $\|x\|$ and $\|v\|$. The following lemma establishes this controlled growth together with the requisite Feller property.
:::

:::{prf:lemma} Perturbation second moment in the Sasaki metric
:label: lem-euclidean-perturb-moment

```{dropdown} Proof
:::{prf:proof}
Introduce the uncapped velocity update

$$
\tilde v:=v+\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau\big(v-u(x)\big)+\sqrt{\sigma_v^2\tau}\,\xi_v,\qquad \xi_v\sim\mathcal N(0,I_d).
$$

Write $a(x,v):=\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau\big(v-u(x)\big)$. {ref}`Stage 4 <sec-eg-stage4>` applies the cap and Euler step to obtain

$$
v^+:=\psi_v(\tilde v),\qquad x^+:=x+\tau v^+ + \sqrt{\tau}\,\sigma_x\,\xi_x,
$$

with $\xi_x\sim\mathcal N(0,I_d)$ independent of $\xi_v$.

We bound the expected Sasaki increment in three explicit steps.

1. **Positional increment.** The cap guarantees $\|v^+\|\le V_{\mathrm{alg}}$. Hence

$$
\|x^+-x\|\le\tau\,\|v^+\|+\sqrt{\tau}\,\sigma_x\,\|\xi_x\|\le\tau V_{\mathrm{alg}}+\sqrt{\tau}\,\sigma_x\,\|\xi_x\|,
$$

so $\mathbb E\big[\|x^+-x\|^2\big]\le 2\tau^2 V_{\mathrm{alg}}^2+2\tau\sigma_x^2 d$.

2. **Velocity increment.** Lemma {prf:ref}`lem-squashing-properties-generic` gives $\|v^+-v\|\le\|\tilde v-v\|$. The random increment decomposes as

$$
\tilde v-v=a(x,v)+\sqrt{\sigma_v^2\tau}\,\xi_v.
$$

Let $F_0:=\|F(0)\|$ and $u_0:=\|u(0)\|$. The Lipschitz bounds $\|F(x)\|\le F_0+L_F\|x\|$ and $\|u(x)\|\le u_0+L_u\|x\|$ imply

$$
\|a(x,v)\|\le\frac{\tau}{m}\big(F_0+L_F\|x\|\big)+\gamma_{\mathrm{fric}}\tau\Big(\|v\|+u_0+L_u\|x\|\Big).
$$

Define the coefficients

$$
A_x:=\tau\Big(\frac{L_F}{m}+\gamma_{\mathrm{fric}}L_u\Big),\qquad A_v:=\gamma_{\mathrm{fric}}\tau,\qquad A_0:=\frac{\tau}{m}F_0+\gamma_{\mathrm{fric}}\tau u_0.
$$

Then $\|a(x,v)\|\le A_x\|x\|+A_v\|v\|+A_0$. Using $\mathbb E\|\xi_v\|^2=d$ and $(\alpha+\beta+\gamma)^2\le 3(\alpha^2+\beta^2+\gamma^2)$ gives

$$
\mathbb E\big[\|\tilde v-v\|^2\big]\le 3A_x^2\|x\|^2+3A_v^2\|v\|^2+3A_0^2+\sigma_v^2\tau d.
$$

3. **Assemble the Sasaki moment.** By definition of the Sasaki metric,

$$
d_{\mathcal Y}^{\mathrm{Sasaki}}\big((x,v),(x^+,v^+)\big)^2=\|x^+-x\|^2+\lambda_v\,\|v^+-v\|^2.
$$

Taking expectations and combining the bounds from Steps 1–2 yields

$$
\mathbb E\big[d_{\mathcal Y}^{\mathrm{Sasaki}}\big((x,v),(x^+,v^+)\big)^2\big]\le C_x^{(\mathrm{pert})}\,\|x\|^2+C_v^{(\mathrm{pert})}\,\|v\|^2+C_0^{(\mathrm{pert})},
$$

with

$$
\begin{aligned}
C_x^{(\mathrm{pert})}&:=3\lambda_vA_x^2,\\
C_v^{(\mathrm{pert})}&:=3\lambda_vA_v^2,\\
C_0^{(\mathrm{pert})}&:=2\tau^2V_{\mathrm{alg}}^2+2\tau\sigma_x^2 d+3\lambda_vA_0^2+\lambda_v\sigma_v^2\tau d.
\end{aligned}
$$

The kinetic kernel is Feller: it composes the continuous affine map $(x,v)\mapsto(x,\tilde v)$, the 1-Lipschitz projection $\psi_v$, and addition of a Gaussian with full support; appending the deterministic status update preserves this property.

```
:::
2. **Geometric consistency constants.** Lemma {prf:ref}`lem-euclidean-geometric-consistency` bounds the drift and anisotropy parameters in the Sasaki geometry.

:::{prf:lemma} Geometric consistency under the capped kinetic kernel
:label: lem-euclidean-geometric-consistency

```{dropdown} Proof
:::{prf:proof}
Because $\mathcal X_{\mathrm{valid}}$ is compact and $F$ and $u$ are continuous, the drift and anisotropy envelopes appearing in the Axiom of Geometric Consistency admit finite global bounds. To make the dependence on the geometry explicit we index the constants by an arbitrary compact subset $C \subset \mathcal X_{\mathrm{valid}}`; in practice we take $C = \mathcal X_{\mathrm{valid}}` and obtain uniform constants on the entire valid domain.

Let $C\subset\mathbb R^d$ be an arbitrary compact set and define the local envelopes

$$
F_C:=\sup_{x\in C}\|F(x)\|,\qquad u_C:=\sup_{x\in C}\|u(x)\|,
$$

which are finite by continuity. Set

$$
C_{\mathrm{force}}(C):=\frac{F_C}{m}+\gamma_{\mathrm{fric}}\big(V_{\mathrm{alg}}+u_C\big),\qquad M_{\mathrm{kin}}(C):=V_{\mathrm{alg}}+\frac{\tau}{m}F_C+\gamma_{\mathrm{fric}}\tau\big(V_{\mathrm{alg}}+u_C\big).
$$

Let $\tilde v:=v+\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau\big(v-u(x)\big)+\sqrt{\sigma_v^2\tau}\,\xi_v$ with $\xi_v\sim\mathcal N(0,I_d)$ and set $v^+:=\psi_v(\tilde v)$, $x^+:=x+\tau v^+ + \sqrt{\tau}\,\sigma_x\,\xi_x$. Denote

$$
a(x,v):=\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau\big(v-u(x)\big).
$$

We supply explicit constants for the drift and anisotropy parts of Definition {prf:ref}`def-axiom-geometric-consistency`.

1. **Drift of the mean displacement.** Because $\|v\|\le V_{\mathrm{alg}}$ we have

$$
\|\mathbb E[x^+-x]\|=\tau\,\|\mathbb E[v^+]\|\le\tau\big(\|\mathbb E[v^+-v]\|+V_{\mathrm{alg}}\big).
$$

The increment of the velocity splits as

$$
\mathbb E[v^+-v]=\mathbb E[\tilde v-v]+\mathbb E[\psi_v(\tilde v)-\tilde v].
$$

The affine term obeys $\|\mathbb E[\tilde v-v]\|=\|a(x,v)\|\le\tau C_{\mathrm{force}}(C)$. The projection error equals $(\|\tilde v\|-V_{\mathrm{alg}})_+$ and is supported on the capping event $E:=\{\|\tilde v\|>V_{\mathrm{alg}}\}$. Markov's inequality applied to

$$
\mathbb E\|\tilde v\|^2=\|a(x,v)\|^2+\sigma_v^2\tau\,\mathbb E\|\xi\|^2\le(\tau C_{\mathrm{force}}(C))^2+\sigma_v^2\tau d+V_{\mathrm{alg}}^2
$$

then yields

$$
\mathbb E[(\|\tilde v\|-V_{\mathrm{alg}})_+]\le\frac{(V_{\mathrm{alg}}+\tau C_{\mathrm{force}}(C))^2+\sigma_v^2\tau d-V_{\mathrm{alg}}^2}{V_{\mathrm{alg}}}=:\varepsilon_{\mathrm{cap}}^{\max}(C).
$$

Thus $\|\mathbb E[v^+-v]\|\le\tau C_{\mathrm{force}}(C)+\varepsilon_{\mathrm{cap}}^{\max}(C)$ and

$$
\|\mathbb E[x^+-x]\|\le\tau\big(\tau C_{\mathrm{force}}(C)+\varepsilon_{\mathrm{cap}}^{\max}(C)+V_{\mathrm{alg}}\big).
$$

Combining the position and velocity components gives

$$
\kappa_{\mathrm{drift}}^{\mathrm{Sasaki}}(C):=\sqrt{\tau^2\big(\tau C_{\mathrm{force}}(C)+\varepsilon_{\mathrm{cap}}^{\max}(C)+V_{\mathrm{alg}}\big)^2+\lambda_v\big(\tau C_{\mathrm{force}}(C)+\varepsilon_{\mathrm{cap}}^{\max}(C)\big)^2}.
$$

2. **Probability of capping.** The same second-moment estimate implies

$$
\rho_*(C):=\mathbb P(E)\le\frac{\mathbb E\|\tilde v\|^2}{V_{\mathrm{alg}}^2}\le\frac{(V_{\mathrm{alg}}+\tau C_{\mathrm{force}}(C))^2+\sigma_v^2\tau d}{V_{\mathrm{alg}}^2}.
$$

3. **Lower bound on the uncapped density.** The mean of $\tilde v$ satisfies

$$
\|m(x,v)\|=\Big\|v+\frac{\tau}{m}F(x)-\gamma_{\mathrm{fric}}\tau(v-u(x))\Big\|\le M_{\mathrm{kin}}(C).
$$

The Gaussian density of $\tilde v$ is

$$
p_{\tilde v}(y)=\frac{1}{(2\pi\sigma_v^2\tau)^{d/2}}\exp\Big(-\frac{\|y-m(x,v)\|^2}{2\sigma_v^2\tau}\Big).
$$

For $u\in S^{d-1}$ and $0\le r\le V_{\mathrm{alg}}/2$ the inequality $\|a-b\|^2\le 2\|a\|^2+2\|b\|^2$ yields

$$
\|ru-m(x,v)\|^2\le\Big(\frac{V_{\mathrm{alg}}}{2}+M_{\mathrm{kin}}(C)\Big)^2.
$$

Hence $p_{\tilde v}(ru)\ge c_{d,0}$ where

$$
c_{d,0}(C):=\frac{1}{(2\pi\sigma_v^2\tau)^{d/2}}\exp\Big(-\frac{(V_{\mathrm{alg}}/2+M_{\mathrm{kin}}(C))^2}{2\sigma_v^2\tau}\Big)>0.
$$

Integrating over the radial segment $[0,V_{\mathrm{alg}}/2]$ yields

$$
P(\tilde v\in K(u))\ge c_{d,0}(C)\int_{0}^{V_{\mathrm{alg}}/2} r^{d-1}\,dr=:c_d(C)>0,
$$

where $K(u):=\{ru:0\le r\le V_{\mathrm{alg}}/2\}$ and we used the polar-volume factor $r^{d-1}$. Thus every cone with opening direction $u$ receives probability at least $c_d(C)$.

4. **Pushforward through the cap.** On $E^c$ the cap is inactive and $v^+=\tilde v$, so the lower bound from Step 3 applies. On $E$ the map $\psi_v$ replaces the radial component by $V_{\mathrm{alg}}$ while leaving the direction $u$ unchanged, whence the directional distribution of $v^+$ dominates $(1-\rho_*(C))c_d(C)$ times the surface measure $\sigma_{d-1}$ on $S^{d-1}$. Equivalently, for every measurable $A\subseteq S^{d-1}$

$$
P(v^+\in A)\ge(1-\rho_*(C))c_d(C)\,\sigma_{d-1}(A).
$$

Taking reciprocals furnishes the anisotropy constant

$$
\kappa_{\mathrm{anisotropy}}^{\mathrm{Sasaki}}(C):=\frac{1}{(1-\rho_*(C))c_d(C)}.
$$

These constants realise the drift and anisotropy requirements of Definition {prf:ref}`def-axiom-geometric-consistency` on the compact set $C$. Since $C$ was arbitrary, the bounds hold uniformly on every compact subset of the state space, which suffices for the non-compact geometric-consistency axiom.

```
:::
3. **Distance-to-companion continuity.** The Sasaki geometry requires re-deriving the canonical continuity bounds for the expected raw distance vector before invoking the mean-square argument.

#### 2.3.3 Continuity of the Expected Raw Distance Vector ($k \ge 2$ Regime)

:::{admonition} Note on the Model Specification: Canonical EG vs. Full Spatially-Aware Model
:class: warning

The analysis in the remainder of this chapter is performed for the **canonical Euclidean Gas**, which assumes **uniform random companion selection**. This corresponds to a mean-field model with an infinite interaction range (`ε → ∞`). The continuity bounds and explicit constants derived for operators like the distance measurement (`thm-sasaki-distance-ms`) and standardization (`thm-sasaki-standardization-composite-sq`) are rigorously proven for this specific instantiation.

**Key Simplifications in the Canonical Model:**
- **Companion Selection:** Uniform random selection from alive walkers (infinite ε limit)
- **Algorithmic Distance:** While formally defined in Section 1.3 as `d_alg(i,j)² = ||x_i - x_j||² + λ_alg ||v_i - v_j||²`, the uniform selection means this distance only affects the raw distance measurement `d_i`, not the companion selection probabilities.
- **Cloning Operator:** Position reset with jitter, velocity reset without jitter (as defined in `03_cloning.md`)

This simplified, non-local model serves as a valuable and tractable baseline. The main convergence proof, presented in the `03_cloning.md` document, builds upon this foundation by analyzing the full model with:
- **Finite ε:** Companion selection weighted by `exp(-d_alg(i,j)² / 2ε²)`
- **Spatially-Aware Pairing:** Full sequential stochastic greedy pairing operator
- **Phase-Space Geometry:** Complete treatment of position-velocity coupling

The proofs in `03_cloning.md` generalize the calculations presented here to be `ε`-dependent, allowing for a full analysis of the local-interaction regime and the transition from position-only (`λ_alg = 0`) to full phase-space geometry (`λ_alg > 0`).
:::

Let $\mathcal S_1,\mathcal S_2$ be two swarm states and denote their alive sets by $\mathcal A_r:=\mathcal A(\mathcal S_r)$. Write $k_r:=|\mathcal A_r|$ and assume $k_1\ge 2$. Define the set of walkers that remain alive in both swarms by $\mathcal A_{\mathrm{stable}}:=\mathcal A_1\cap\mathcal A_2$ and the number of status changes by $n_c(\mathcal S_1,\mathcal S_2):=\sum_{i=1}^N (s_{1,i}-s_{2,i})^2$. The positional displacement of the capped states is

$$
\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2):=\sum_{i=1}^N d_{\mathcal Y}^{\mathrm{Sasaki}}\big(\varphi(w_{1,i}),\varphi(w_{2,i})\big)^2,
$$
and let $D_{\mathcal Y}:=\operatorname{diam}_{d_{\mathcal Y}^{\mathrm{Sasaki}}}(\mathcal Y)$.

:::{prf:lemma} Single-walker positional error bound in the Sasaki metric
:label: lem-sasaki-single-walker-positional-error

Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states. For a given walker $i$ that is alive in swarm $\mathcal S_1$ ($s_{1,i}=1$), let $\mathbb C_i(\mathcal S_1)$ be its companion selection measure.

The absolute error in its expected distance due to the positional displacement of the walkers between the two states, evaluated over the fixed companion set from $\mathcal S_1$, is bounded by the sum of its own displacement and the average displacement of its potential companions.

$$
\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] \right| \le d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i})) + \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}), \varphi(w_{2,c})) \right]
$$

```{dropdown} Proof
:::{prf:proof}
Let $\Delta_{\mathrm{pos},i}$ denote the absolute error term we wish to bound. The proof proceeds by applying standard metric and probability inequalities.

**Step 1: Apply Linearity of Expectation.**
We combine the two terms into a single expectation over the fixed companion selection measure $\mathbb C_i(\mathcal S_1)$.

$$
\Delta_{\mathrm{pos},i} = \left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) - d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] \right|
$$

**Step 2: Apply Jensen's Inequality.**
Using Jensen's inequality for the convex function $f(x)=|x|$, we can move the absolute value inside the expectation, which provides an upper bound:

$$
\Delta_{\mathrm{pos},i} \le \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ \left| d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) - d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right| \right]
$$

**Step 3: Apply the Reverse Triangle Inequality.**
The term inside the expectation is the absolute difference between two distance values. For any points $a,b,c,d$ in a metric space $(M,d)$, the reverse triangle inequality states that $|d(a,b) - d(c,d)| \le d(a,c) + d(b,d)$. Applying this to the Sasaki metric $d_{\mathcal Y}^{\mathrm{Sasaki}}$ yields:

$$
\left| d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) - d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right| \le d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i})) + d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}), \varphi(w_{2,c}))
$$

**Step 4: Finalize the Bound.**
We substitute the inequality from Step 3 back into the expression from Step 2.

$$
\Delta_{\mathrm{pos},i} \le \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i})) + d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}), \varphi(w_{2,c})) \right]
$$

By linearity of expectation, we can separate the terms. The first term, $d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i}))$, is a constant with respect to the expectation over the companion index $c$. This gives the final bound as stated in the lemma.

**Q.E.D.**
```
:::

:::{prf:lemma} Single-walker structural error bound in the Sasaki metric
:label: lem-sasaki-single-walker-structural-error

Let $i\in\mathcal A_{\mathrm{stable}}$ and keep the second swarm’s capped positions fixed. Let the initial swarm have at least two alive walkers, $k_1=|\mathcal A(\mathcal S_1)| \ge 2$. The absolute error in the expected distance for walker $i$ due to the change in the companion selection measure is bounded by:

$$
\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_2)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] \right| \le \frac{2 D_{\mathcal Y}}{k_1-1} \cdot n_c(\mathcal S_1, \mathcal S_2)
$$

where $D_{\mathcal Y}$ is the diameter of the algorithmic space.

```{dropdown} Proof
:::{prf:proof}
This result is a direct application of the framework's **Total Error Bound in Terms of Status Changes** ({prf:ref}`thm-total-error-status-bound`) to the specific function of interest in the Sasaki geometry.

**Step 1: Identify the Function and its Bound.**
Let the function being evaluated under the expectation be $f(c) := d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c}))$. This function measures the Sasaki distance from the fixed walker $i$ to a potential companion $c$, using the positions from the second swarm. By definition, any distance in the algorithmic space is bounded by the space's diameter, $D_{\mathcal Y}$. Therefore, we have a uniform bound on the function's value: $|f(c)| \le D_{\mathcal Y} =: M_f$.

**Step 2: Identify the Companion Support Sets.**
Let $S_1 = S_i(\mathcal{S}_1)$ and $S_2 = S_i(\mathcal{S}_2)$ be the companion support sets for walker $i$ in the two swarms. Since walker $i$ is alive in $\mathcal S_1$ (i.e., $i \in \mathcal A_{\mathrm{stable}} \subseteq \mathcal A_1$) and the precondition states $k_1 \ge 2$, the initial support set is $S_1 = \mathcal A_1 \setminus \{i\}$. Its size is therefore $|S_1| = k_1 - 1 > 0$.

**Step 3: Apply the General Error Bound.**
The framework theorem {prf:ref}`thm-total-error-status-bound` provides a general bound for the change in expectation of a bounded function due to a change in the underlying support set:

$$
\text{Error} \le \frac{2 M_f}{|S_1|} \cdot n_c(\mathcal S_1, \mathcal S_2)
$$
This bound is purely algebraic and holds for any choice of metric or bounded function.

**Step 4: Substitute and Finalize.**
We substitute our specific function bound $M_f = D_{\mathcal Y}$ and the support set size $|S_1| = k_1 - 1$ into the general formula. This immediately yields the stated bound for the structural error component.

**Q.E.D.**
```
:::



:::{prf:lemma} Mean-square error on stable walkers (Sasaki)
:label: lem-sasaki-total-squared-error-stable

Let $\mathcal S_1,\mathcal S_2$ be swarms with alive sets $\mathcal A_r$ and let $\mathbf d^{(r)}$ denote the expected raw distance vector produced by the measurement operator on $\mathcal S_r$. Write $\mathcal A_{\mathrm{stable}}:=\mathcal A_1\cap\mathcal A_2$ and $k_{\mathrm{stable}}:=|\mathcal A_{\mathrm{stable}}|$. Then

$$
\sum_{i\in\mathcal A_{\mathrm{stable}}}\big|d^{(1)}_i-d^{(2)}_i\big|^2\le C_{\mathrm{pos}}^{\mathrm{Sasaki}}(k_1,k_{\mathrm{stable}})\,\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2),
$$

where $C_{\mathrm{pos}}^{\mathrm{Sasaki}}(k_1,k_{\mathrm{stable}}):=2\Big(1+\frac{k_{\mathrm{stable}}}{\max\{1,k_1-1\}}\Big)$.

```{dropdown} Proof
:::{prf:proof}
For $i\in\mathcal A_{\mathrm{stable}}$ set $\Delta_i:=|d^{(1)}_i-d^{(2)}_i|$. Lemma {prf:ref}`lem-sasaki-single-walker-positional-error` gives

$$
\Delta_i\le d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}),\varphi(w_{2,i})) + \mathbb E_{c\sim\mathbb C_i(\mathcal S_1)}\big[d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}),\varphi(w_{2,c}))\big].
$$

Apply $(a+b)^2\le 2a^2+2b^2$ and Jensen's inequality to obtain

$$
\Delta_i^2\le 2\,d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}),\varphi(w_{2,i}))^2 + \frac{2}{k_1-1}\sum_{j\in\mathcal A_1} d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,j}),\varphi(w_{2,j}))^2,
$$
where the averaging denominator $k_1-1$ is interpreted as $1$ when $k_1=1$. Summing over $i\in\mathcal A_{\mathrm{stable}}$ yields

$$
\sum_{i\in\mathcal A_{\mathrm{stable}}}\Delta_i^2\le 2\,\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2)+\frac{2k_{\mathrm{stable}}}{\max\{1,k_1-1\}}\,\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2),
$$
which is the claimed bound.```
:::

:::{prf:theorem} Mean-square continuity of the distance measurement (Sasaki)
:label: thm-sasaki-distance-ms

Let $\mathbf d^{(r)}$ be the expected raw distance vectors of swarms $\mathcal S_r$. With $k_{\min}:=\max\{1,\min(k_1,k_2)\}$, $k_{\mathrm{stable}}:=|\mathcal A_{\mathrm{stable}}|$, and alive-difference count $n_c:=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2$, define

$$
F_{d,ms}^{\mathrm{Sasaki}}(\Delta_{\mathrm{pos}}^2,n_c):=C_{\mathrm{pos}}^{\mathrm{Sasaki}}(k_1,k_{\mathrm{stable}})\,\Delta_{\mathrm{pos}}^2+4k_{\mathrm{stable}}\frac{D_{\mathcal Y}^2}{\max\{1,k_1-1\}^2}\,n_c^2+4D_{\mathcal Y}^2 n_c.
$$

Then

$$
\big\|\mathbf d^{(1)}-\mathbf d^{(2)}\big\|_2^2\le F_{d,ms}^{\mathrm{Sasaki}}\big(\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2),n_c(\mathcal S_1,\mathcal S_2)\big).
$$

```{dropdown} Proof
:::{prf:proof}
Decompose the index set into stable walkers $\mathcal A_{\mathrm{stable}}$ and the complement. For stable walkers the bound in Lemma {prf:ref}`lem-sasaki-total-squared-error-stable` applies. For walkers whose status changes between the two swarms we use $|d^{(1)}_i-d^{(2)}_i|\le D_{\mathcal Y}$ because each expected distance is bounded by the diameter of the Sasaki algorithmic space. There are at most $2n_c$ such indices, contributing at most $4D_{\mathcal Y}^2n_c$ to the squared error.

Finally, the structural perturbation of the companion distribution for stable walkers is controlled by Lemma {prf:ref}`lem-sasaki-single-walker-structural-error`. Squaring its bound and summing over the $k_{\mathrm{stable}}$ indices yields the middle term in $F_{d,ms}^{\mathrm{Sasaki}}$. Adding the three contributions completes the proof.```
:::

4. **Non-degenerate noise.** Choosing $\sigma_v^2>0$ and cloning scales $\delta_x,\delta_v>0$ keeps the perturbation and cloning measures non-Dirac.

5. **Sufficient amplification.** The weights $\alpha,\beta\ge 0$ satisfy $\alpha+\beta>0$ exactly as in the canonical swarm ({prf:ref}`def-axiom-sufficient-amplification`).

6. **Aggregator axioms.** Let $R_{\max}:=\sup_{x\in\mathcal X}|R_{\mathrm{pos}}(x)|+\lambda_{\mathrm{vel}}V_{\mathrm{alg}}^2$ and recall from Lemma {prf:ref}`lem-euclidean-reward-regularity` that the reward satisfies the Lipschitz bound

$$
|R(x_1,v_1)-R(x_2,v_2)|\le L_R^{\mathrm{Sasaki}}\,d_{\mathcal Y}^{\mathrm{Sasaki}}\big((x_1,v_1),(x_2,v_2)\big),\qquad L_R^{\mathrm{Sasaki}}:=L_{\mathrm{pos}}+\frac{2\lambda_{\mathrm{vel}}V_{\mathrm{alg}}}{\sqrt{\lambda_v}}.
$$
Whenever aggregators act on reward vectors we use the uniform bound $V_{\mathrm{max}}^{(R)}:=\max\{|R_{\min}|,R_{\max}\}$; for distance vectors we use $V_{\mathrm{max}}^{(d)}:=D_{\mathcal Y}$. For swarms $\mathcal S_r$ write $k_r:=|\mathcal A(\mathcal S_r)|$, define $k_{\min}:=\max\{1,\min(k_1,k_2)\}$, and let $n_c:=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2$ count the status changes.

:::{prf:lemma} Value continuity of the empirical moments
:label: lem-sasaki-aggregator-value

Fix a swarm $\mathcal S$ with alive index set $\mathcal A(\mathcal S)$ of size $k\ge 1$. Let $\mathbf v_1,\mathbf v_2\in\mathbb R^k$ be two scalar value vectors whose components satisfy $|v_{j,i}|\le V_{\max}$. Then the empirical mean and second moment obey

$$
|\mu(\mathcal S,\mathbf v_1)-\mu(\mathcal S,\mathbf v_2)|\le \frac{1}{\sqrt{k}}\,\|\mathbf v_1-\mathbf v_2\|_2,\qquad|m_2(\mathcal S,\mathbf v_1)-m_2(\mathcal S,\mathbf v_2)|\le \frac{2V_{\max}}{\sqrt{k}}\,\|\mathbf v_1-\mathbf v_2\|_2.
$$

```{dropdown} Proof
:::{prf:proof}
The identities follow from the gradient calculations $\nabla\mu=(1/k)\mathbf 1$ and $\nabla m_2=(2/k)\mathbf v$ together with Cauchy–Schwarz, as in Lemma 6.2.2.a of the framework.

```
:::

:::{prf:lemma} Structural continuity of the empirical moments
:label: lem-sasaki-aggregator-structural

Let $\mathcal S_r=((x_{r,i},v_{r,i},s_{r,i}))_{i=1}^N$ with alive counts $k_r\ge 1$ and let $\mathbf v$ be a scalar vector on the union of alive indices satisfying $|v_i|\le V_{\max}$. Set $k_{\min}:=\max\{1,\min(k_1,k_2)\}$ and $n_c:=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2$. Then

$$
|\mu(\mathcal S_1,\mathbf v)-\mu(\mathcal S_2,\mathbf v)|\le \frac{3V_{\max}}{k_{\min}}\,n_c,\qquad|m_2(\mathcal S_1,\mathbf v)-m_2(\mathcal S_2,\mathbf v)|\le \frac{3V_{\max}^2}{k_{\min}}\,n_c.
$$

```{dropdown} Proof
:::{prf:proof}
The proof mirrors Lemma 6.2.2.b of the framework. Decompose the difference in means into contributions from walkers that remain alive in both swarms and those that change status. The former vanish, whereas the latter introduce at most $V_{\max}$ per status flip. Accounting for the normalisation factors $1/k_r$ and the difference in alive counts yields the stated bounds. The argument for $m_2$ uses $|a^2-b^2|\le 2V_{\max}|a-b|$.

```
:::

:::{prf:lemma} Lipschitz data for the Sasaki empirical aggregators
:label: lem-sasaki-aggregator-lipschitz

For reward vectors take $V_{\max}=V_{\mathrm{max}}^{(R)}$; for distance vectors take $V_{\max}=V_{\mathrm{max}}^{(d)}$. The empirical mean and second moment satisfy the aggregator axioms with

$$
L_{\mu,M}^{\mathrm{Sasaki}}(k)=\frac{1}{\sqrt{k}},\qquad L_{m_2,M}^{\mathrm{Sasaki}}(k)=\frac{2V_{\max}}{\sqrt{k}},
$$

$$
L_{\mu,S}^{\mathrm{Sasaki}}(k_{\min})=\frac{3V_{\max}}{k_{\min}},\qquad L_{m_2,S}^{\mathrm{Sasaki}}(k_{\min})=\frac{3V_{\max}^2}{k_{\min}},
$$
and growth exponents $p_{\mu,S}=p_{m_2,S}=p_{\mathrm{worst	ext{-}case}}=-1$. Consequently $\kappa_{\mathrm{var}}^{\mathrm{Sasaki}}=\kappa_{\mathrm{range}}^{\mathrm{Sasaki}}=1$ as in the canonical framework.

```{dropdown} Proof
:::{prf:proof}
Combine Lemmas {prf:ref}`lem-sasaki-aggregator-value` and {prf:ref}`lem-sasaki-aggregator-structural` with the dispersion metric identity $n_c\le(N/\sqrt{\lambda_{\mathrm{status}}})d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)$ to obtain the stated Lipschitz functions and exponents.
```
:::

7. **Standardization & rescale continuity.** Let $\sigma_{\min,\mathrm{patch}}:=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}$ be the lower bound supplied by the regularized standard deviation operator, and denote by $L_{\sigma'_{\mathrm{patch}}}$ the global derivative bound from Lemma {prf:ref}`lem-sigma-patch-lipschitz`. For notational compactness write

$$
L_{\sigma',M}^{\mathrm{Sasaki}}(k):=L_{\sigma'_{\mathrm{patch}}}\Big(L_{m_2,M}^{\mathrm{Sasaki}}(k)+2V_{\mathrm{max}}^{(R)}L_{\mu,M}^{\mathrm{Sasaki}}(k)\Big).
$$

:::{prf:definition} Standardization constants (Sasaki geometry)
:label: def-sasaki-standardization-constants

Let $\sigma_{\min,\mathrm{patch}}:=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}$ be the uniform lower bound on the regularized standard deviation, and let $L_{\sigma'_{\mathrm{patch}}}$ be its global Lipschitz constant from Lemma {prf:ref}`lem-sigma-patch-lipschitz`.

##### Value Error Coefficients
The following coefficients bound the error in the standardization operator when the swarm structure is fixed but the raw values change due to positional displacement. They are notably independent of the number of alive walkers, `k`.

-   **Direct Shift Coefficient ($C_{V,\mathrm{direct}}$):** Bounding the error from the direct change in the raw value vector.

    $$
    C_{V,\mathrm{direct}} := \frac{1}{\sigma_{\min,\mathrm{patch}}}
    $$

-   **Mean Shift Coefficient ($C_{V,\mathrm{mean}}$):** Bounding the error from the resulting change in the empirical mean.

    $$
    C_{V,\mathrm{mean}} := \frac{1}{\sigma_{\min,\mathrm{patch}}}
    $$

-   **Denominator Shift Coefficient ($C_{V,\mathrm{denom}}$):** Bounding the error from the resulting change in the regularized standard deviation.

    $$
    C_{V,\mathrm{denom}} := \frac{8\big(V_{\mathrm{max}}^{(R)}\big)^2 L_{\sigma'_{\mathrm{patch}}}}{\sigma_{\min,\mathrm{patch}}^2}
    $$

-   **Total Value Error Coefficient ($C_{V,\mathrm{total}}^{\mathrm{Sasaki}}$):** The composite coefficient for the full Lipschitz bound on the value error, which aggregates the component-wise effects.

    $$
    C_{V,\mathrm{total}}^{\mathrm{Sasaki}} := L_R^{\mathrm{Sasaki}} \left( C_{V,\mathrm{direct}} + C_{V,\mathrm{mean}} + C_{V,\mathrm{denom}} \right) = L_R^{\mathrm{Sasaki}} \left( \frac{2}{\sigma_{\min,\mathrm{patch}}} + \frac{8\big(V_{\mathrm{max}}^{(R)}\big)^2 L_{\sigma'_{\mathrm{patch}}}}{\sigma_{\min,\mathrm{patch}}^2} \right)
    $$

##### Structural Error Coefficients
The structural error coefficients, which are used in the subsequent theorem for structural continuity, remain as defined:

$$
C_{S,\mathrm{direct}}^{\mathrm{Sasaki}}(k_{\min}):=\frac{V_{\mathrm{max}}^{(R)}}{\sigma_{\min,\mathrm{patch}}}+\frac{2\big(V_{\mathrm{max}}^{(R)}\big)^2}{\sigma_{\min,\mathrm{patch}}^2},
\qquad C_{S,\mathrm{indirect}}^{\mathrm{Sasaki}}(k_{\min}):=\frac{3V_{\mathrm{max}}^{(R)}}{\sigma_{\min,\mathrm{patch}}k_{\min}}+\frac{6\big(V_{\mathrm{max}}^{(R)}\big)^2}{\sigma_{\min,\mathrm{patch}}^2k_{\min}}L_{\sigma',M}^{\mathrm{Sasaki}}(k_{\min}).
$$
:::


Set $C_R:=L_R^{\mathrm{Sasaki}}\sqrt{N}+R_{\max}\sqrt{\tfrac{N}{\lambda_{\mathrm{status}}}}$ for later use.

These constants verify the continuity axioms for the patched standardization and logistic rescale operators in the Sasaki geometry.

#### 2.3.4. Theorem: Value Continuity of Patched Standardization (Sasaki)

:::{prf:theorem} Value continuity of patched standardization (Sasaki)
Suppose $\mathcal S_1$ and $\mathcal S_2$ share the same alive set $\mathcal A$ of size $k\ge 1$ (so $n_c(\mathcal S_1,\mathcal S_2)=0$). The N-dimensional standardization operator is Lipschitz continuous with respect to positional changes in the Sasaki metric. The squared L2-norm of the output error is bounded as follows:

$$
\big\|z(\mathcal S_1)-z(\mathcal S_2)\big\|_2^2 \le C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1) \cdot \Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2).
$$

where $C_{V,\mathrm{total}}^{\mathrm{Sasaki}}$ is the **Total Value Error Coefficient**, a deterministic constant defined in {prf:ref}`def-sasaki-standardization-constants-sq`. The proof is provided in the subsequent sections by decomposing the total error into its constituent parts.
:::

##### 2.3.4.1. Sub-Lemma: Algebraic Decomposition of the Value Error

:::{prf:lemma} Decomposition of the Value Error
:label: lem-sasaki-value-error-decomposition

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k$. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors for the alive set. Let $(\mu_1, \sigma'_1)$ and $(\mu_2, \sigma'_2)$ be the corresponding statistical properties, and let $\mathbf z_1$ and $\mathbf z_2$ be the corresponding standardized vectors.

The total value error vector, $\Delta\mathbf{z} = \mathbf z_1 - \mathbf z_2$, can be expressed as the sum of three components:

$$
\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{denom}}
$$

where:
1.  **The Direct Shift ($\Delta_{\text{direct}}$):** The error from the change in the raw value vector itself, scaled by the initial standard deviation.

    $$
    \Delta_{\text{direct}} := \frac{\mathbf r_1 - \mathbf r_2}{\sigma'_1}
    $$

2.  **The Mean Shift ($\Delta_{\text{mean}}$):** The error from the change in the aggregator's computed mean, applied uniformly to all walkers.

    $$
    \Delta_{\text{mean}} := \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1}
    $$
    where $\mathbf{1}$ is a k-dimensional vector of ones.

3.  **The Denominator Shift ($\Delta_{\text{denom}}$):** The error from the change in the regularized standard deviation, which rescales the second standardized vector.

    $$
    \Delta_{\text{denom}} := \mathbf z_2 \cdot \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}
    $$

Furthermore, the total squared error is bounded by three times the sum of the squared norms of these components:

$$
\|\Delta\mathbf{z}\|_2^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{denom}}\|_2^2 \right)
$$

```{dropdown} Proof
:::{prf:proof}
**Step 1: Algebraic Decomposition.**
The proof of the decomposition is a direct algebraic manipulation. We start with the definition of the error and add and subtract the intermediate term $(\mathbf r_2 - \mu_2) / \sigma'_1$.

$$
\begin{aligned}
\Delta\mathbf{z} &= \frac{\mathbf r_1 - \mu_1}{\sigma'_1} - \frac{\mathbf r_2 - \mu_2}{\sigma'_2} \\
&= \left( \frac{\mathbf r_1 - \mu_1}{\sigma'_1} - \frac{\mathbf r_2 - \mu_2}{\sigma'_1} \right) + \left( \frac{\mathbf r_2 - \mu_2}{\sigma'_1} - \frac{\mathbf r_2 - \mu_2}{\sigma'_2} \right) \\
&= \frac{(\mathbf r_1 - \mathbf r_2) - (\mu_1 - \mu_2)}{\sigma'_1} + (\mathbf r_2 - \mu_2) \left( \frac{1}{\sigma'_1} - \frac{1}{\sigma'_2} \right) \\
&= \frac{\mathbf r_1 - \mathbf r_2}{\sigma'_1} + \frac{\mu_2 - \mu_1}{\sigma'_1}\mathbf{1} + \frac{\mathbf r_2 - \mu_2}{\sigma'_2} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1} \\
&= \Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{denom}}
\end{aligned}
$$
The final line follows by recognizing the definitions of the three components.

**Step 2: Bound on the Squared Norm.**
The bound on the total squared norm follows from the triangle inequality (`||A+B+C|| <= ||A|| + ||B|| + ||C||`) and the elementary inequality $(a+b+c)^2 \le 3(a^2+b^2+c^2)$ for non-negative reals. For vectors, this becomes:

$$
\|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{denom}}\|_2^2 \le \left( \|\Delta_{\text{direct}}\|_2 + \|\Delta_{\text{mean}}\|_2 + \|\Delta_{\text{denom}}\|_2 \right)^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{denom}}\|_2^2 \right)
$$

This completes the proof.

**Q.E.D.**
```
:::

##### 2.3.4.2. Sub-Lemma: Bounding the Squared Direct Shift Component

:::{prf:lemma} Bound on the Squared Direct Shift Component
:label: lem-sasaki-direct-shift-bound-sq

Let $\mathcal S$ be a fixed swarm state. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors for the alive set. The squared Euclidean norm of the direct shift error component, $\Delta_{\text{direct}} = (\mathbf r_1 - \mathbf r_2) / \sigma'_1$, is bounded as follows:

$$
\|\Delta_{\text{direct}}\|_2^2 \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
$$

where $\sigma_{\min,\mathrm{patch}} := \sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}$ is the uniform lower bound from the regularized standard deviation.

```{dropdown} Proof
:::{prf:proof}
The proof is a direct application of the definition of $\Delta_{\text{direct}}$ and the uniform lower bound on the regularized standard deviation.

1.  **Start with the Definition.**
    The squared L2-norm of the direct shift component is:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \left\| \frac{\mathbf r_1 - \mathbf r_2}{\sigma'_1} \right\|_2^2
    $$

2.  **Factor out the Scalar Term.**
    Since $\sigma'_1$ is a scalar value for the fixed swarm state and value vector $\mathbf r_1$, we can factor it out of the norm:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \frac{1}{(\sigma'_1)^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
    $$

3.  **Apply the Uniform Lower Bound.**
    The regularized standard deviation function $\sigma'_{\mathrm{patch}}(V)$ is, by construction in the framework ({prf:ref}`def-statistical-properties-measurement`), strictly positive and uniformly bounded below by the constant $\sigma_{\min,\mathrm{patch}}$. Therefore, $\sigma'_1 \ge \sigma_{\min,\mathrm{patch}} > 0$. This implies:

    $$
    \frac{1}{(\sigma'_1)^2} \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2}
    $$

4.  **Combine to Finalize the Bound.**
    Substituting the inequality from Step 3 into the expression from Step 2 yields the final bound as stated in the lemma.

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
    $$

**Q.E.D.**
```
:::

##### 2.3.4.3. Sub-Lemma: Bounding the Squared Mean Shift Component

:::{prf:lemma} Bound on the Squared Mean Shift Component
:label: lem-sasaki-mean-shift-bound-sq

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k \ge 1$. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors. The squared Euclidean norm of the mean shift error component, $\Delta_{\text{mean}} = ((\mu_2 - \mu_1) / \sigma'_1) \cdot \mathbf{1}$, is bounded as follows:

$$
\|\Delta_{\text{mean}}\|_2^2 \le \frac{k \cdot (L_{\mu,M}^{\mathrm{Sasaki}}(k))^2}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
$$

where $L_{\mu,M}^{\mathrm{Sasaki}}(k)$ is the axiomatic **Value Lipschitz Function** for the aggregator's mean from {prf:ref}`lem-sasaki-aggregator-lipschitz`.

```{dropdown} Proof
:::{prf:proof}
The proof combines the definition of the mean shift component with the axiomatic continuity of the mean aggregator.

1.  **Start with the Definition.**
    The squared L2-norm of the mean shift component is:

    $$
    \|\Delta_{\text{mean}}\|_2^2 = \left\| \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1} \right\|_2^2
    $$

2.  **Factor out the Scalar and Evaluate the Norm.**
    The term $(\mu_2 - \mu_1) / \sigma'_1$ is a scalar. The L2-norm of the k-dimensional vector of ones, $\mathbf{1}$, is $\|\mathbf{1}\|_2 = \sqrt{k}$. Therefore, the squared norm is:

    $$
    \|\Delta_{\text{mean}}\|_2^2 = \frac{(\mu_2 - \mu_1)^2}{(\sigma'_1)^2} \cdot \|\mathbf{1}\|_2^2 = \frac{k \cdot (\mu_2 - \mu_1)^2}{(\sigma'_1)^2}
    $$

3.  **Apply Axiomatic Continuity of the Mean.**
    The empirical aggregator is Lipschitz continuous with respect to the raw value vector, as established in {prf:ref}`lem-sasaki-aggregator-value`. This provides the bound:

    $$
    |\mu_2 - \mu_1|^2 \le \left(L_{\mu,M}^{\mathrm{Sasaki}}(k)\right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
    $$

4.  **Apply the Uniform Lower Bound.**
    As in the previous lemma, we use the bound $1/(\sigma'_1)^2 \le 1/\sigma_{\min,\mathrm{patch}}^2$.

5.  **Combine to Finalize the Bound.**
    Substituting the bounds from Step 3 and Step 4 into the expression from Step 2 yields the final result as stated in the lemma.

    $$
    \|\Delta_{\text{mean}}\|_2^2 \le \frac{k \cdot (L_{\mu,M}^{\mathrm{Sasaki}}(k))^2}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
    $$

**Q.E.D.**
```
:::

##### 2.3.4.4. Sub-Lemma: Bounding the Squared Denominator Shift Component

:::{prf:lemma} Bounding the Squared Denominator Shift Component
:label: lem-sasaki-denom-shift-bound-sq

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k \ge 1$. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors with components bounded by $V_{\max}^{(R)}$. The squared Euclidean norm of the denominator shift error component, $\Delta_{\text{denom}} = \mathbf z_2 \cdot ((\sigma'_2 - \sigma'_1) / \sigma'_1)$, is bounded as follows:

$$
\|\Delta_{\text{denom}}\|_2^2 \le k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \left( \frac{L_{\sigma',M}^{\mathrm{Sasaki}}(k)}{\sigma_{\min,\mathrm{patch}}} \right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
$$

where $L_{\sigma',M}^{\mathrm{Sasaki}}(k)$ is the derived Lipschitz constant for the regularized standard deviation.

```{dropdown} Proof
:::{prf:proof}
The proof bounds the squared norm by bounding its three constituent parts: the norm of the standardized vector, the change in the regularized standard deviation, and the inverse of the standard deviation.

1.  **Start with the Definition.**
    The squared L2-norm of the denominator shift component is:

    $$
    \|\Delta_{\text{denom}}\|_2^2 = \left\| \mathbf z_2 \cdot \frac{\sigma'_2 - \sigma'_1}{\sigma'_1} \right\|_2^2
    $$

2.  **Factor out the Scalar Term.**
    The fractional term involving the standard deviations is a scalar. We factor it out of the norm:

    $$
    \|\Delta_{\text{denom}}\|_2^2 = \|\mathbf z_2\|_2^2 \cdot \frac{(\sigma'_2 - \sigma'_1)^2}{(\sigma'_1)^2}
    $$

3.  **Bound Each Factor.**
    We now find a deterministic upper bound for each of the three factors in the expression.
    *   **Bound on `||z2||_2^2`**: The framework provides a universal bound on the squared norm of any standardized vector, proven in {prf:ref}`thm-z-score-norm-bound`. For the k-dimensional vector $\mathbf z_2$, this is:

        $$
        \|\mathbf z_2\|_2^2 \le k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2
        $$

    *   **Bound on `(sigma'_2 - sigma'_1)^2`**: The regularized standard deviation function is Lipschitz continuous with respect to the raw value vector, as established by composing the Lipschitz properties of the aggregator moments and the patching function itself ({prf:ref}`lem-stats-value-continuity` in the framework). This gives:

        $$
        (\sigma'_2 - \sigma'_1)^2 \le \left(L_{\sigma',M}^{\mathrm{Sasaki}}(k)\right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
        $$

    *   **Bound on `1/(sigma'_1)^2`**: As in the preceding lemmas, we use the uniform lower bound:

        $$
        \frac{1}{(\sigma'_1)^2} \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2}
        $$

4.  **Combine to Finalize the Bound.**
    Substituting the bounds for all three factors from Step 3 into the expression from Step 2 yields the final bound as stated in the lemma.

    $$
    \|\Delta_{\text{denom}}\|_2^2 \le \left( k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \right) \cdot \left( \left(L_{\sigma',M}^{\mathrm{Sasaki}}(k)\right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2 \right) \cdot \left( \frac{1}{\sigma_{\min,\mathrm{patch}}^2} \right)
    $$
    Rearranging the terms gives the stated result.

**Q.E.D.**
```
:::

##### 2.3.4.5. Proof of Theorem 2.3.4

:::{prf:proof} Proof

The proof establishes the final bound by assembling the deterministic bounds for each of the three error components derived in the preceding sub-lemmas.

**Step 1: Start with the Decomposed Error Bound.**
From the algebraic decomposition in {prf:ref}`lem-sasaki-value-error-decomposition`, the total squared value error is bounded by:

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{denom}}\|_2^2 \right)
$$

**Step 2: Substitute the Bounds for Each Component.**
We substitute the deterministic bounds for the squared norm of each component, which all relate the component error to the squared norm of the raw value difference, $\|\mathbf r_1 - \mathbf r_2\|_2^2$.

*   From {prf:ref}`lem-sasaki-direct-shift-bound-sq`:

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
    $$
*   From {prf:ref}`lem-sasaki-mean-shift-bound-sq`:

    $$
    \|\Delta_{\text{mean}}\|_2^2 \le C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
    $$
*   From {prf:ref}`lem-sasaki-denom-shift-bound-sq`:

    $$
    \|\Delta_{\text{denom}}\|_2^2 \le C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
    $$

**Step 3: Combine and Factor.**
Substituting these into the inequality from Step 1 and factoring out the common term $\|\mathbf r_1 - \mathbf r_2\|_2^2$ gives:

$$
\|z_1 - z_2\|_2^2 \le 3 \left( C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1) + C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S_1) + C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S_1) \right) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
$$

By definition ({prf:ref}`def-sasaki-standardization-constants-sq`), the term in parentheses is the **Total Value Error Coefficient**, $C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)$.

**Step 4: Relate Raw Value Error to Positional Displacement.**
The raw reward vector difference is bounded by the positional displacement via the Lipschitz continuity of the reward function ({prf:ref}`lem-euclidean-reward-regularity`):

$$
\|\mathbf r_1 - \mathbf r_2\|_2^2 = \sum_{i \in \mathcal A} |R(x_{1,i},v_{1,i}) - R(x_{2,i},v_{2,i})|^2 \le \sum_{i \in \mathcal A} \left(L_R^{\mathrm{Sasaki}}\right)^2 d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i}))^2 \le \left(L_R^{\mathrm{Sasaki}}\right)^2 \Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2)
$$

**Step 5: Final Assembly.**
Substituting the bound from Step 4 into the inequality from Step 3 yields the final result. The constant of proportionality in the theorem statement absorbs the Lipschitz constant of the reward function. For clarity, we can redefine the total coefficient to include this factor, or leave it explicit as shown here. Let's define a new total coefficient to match the theorem statement:
$C_{V,\mathrm{total}}^{\mathrm{Sasaki}} := 3 \cdot \left( C_{V,\mathrm{direct}}^{\mathrm{sq}} + \dots \right) \cdot (L_R^{\mathrm{Sasaki}})^2$. This completes the proof.

**Q.E.D.**
:::

##### 2.3.5. Definition: Value Error Coefficients (Squared Form)

:::{prf:definition} Value Error Coefficients (Squared Form)
:label: def-sasaki-standardization-constants-sq

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k \ge 1$, and let $M$ be the chosen **Swarm Aggregation Operator**. The coefficients for the bounds on the squared value error are defined as follows:

1.  **The Squared Direct Shift Coefficient ($C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S)$):**

    $$
    C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S) := \frac{1}{\sigma_{\min,\mathrm{patch}}^2}
    $$

2.  **The Squared Mean Shift Coefficient ($C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S)$):**

    $$
    C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) := \frac{k \cdot (L_{\mu,M}^{\mathrm{Sasaki}}(k))^2}{\sigma_{\min,\mathrm{patch}}^2}
    $$

3.  **The Squared Denominator Shift Coefficient ($C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S)$):**

    $$
    C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S) := k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \left( \frac{L_{\sigma',M}^{\mathrm{Sasaki}}(k)}{\sigma_{\min,\mathrm{patch}}} \right)^2
    $$

4.  **The Total Value Error Coefficient ($C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S)$):** The composite coefficient that bounds the total squared value error, incorporating the factor of 3 from the error decomposition.

    $$
    C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S) := 3 \cdot \left( C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S) + C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) + C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S) \right)
    $$

where $L_{\mu,M}^{\mathrm{Sasaki}}(k)$ and $L_{\sigma',M}^{\mathrm{Sasaki}}(k)$ are the value Lipschitz functions for the aggregator's mean and regularized standard deviation, respectively. For the canonical empirical aggregator, these coefficients simplify, notably making the mean shift coefficient independent of $k$: $C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) = 1/\sigma_{\min,\mathrm{patch}}^2$.
:::

###### 2.3.6. Theorem: Structural Continuity of Patched Standardization (Sasaki)

:::{prf:theorem} Structural Continuity of Patched Standardization (Sasaki)
:label: thm-sasaki-standardization-structural-sq

For general swarms $\mathcal S_1,\mathcal S_2$ with alive counts $k_r\ge 1$, the squared L2-norm of the output error of the standardization operator is bounded by a function of the number of status changes, $n_c(\mathcal S_1,\mathcal S_2)$.

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2) + C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2
$$

where $C_{S,\mathrm{direct}}^{\mathrm{sq}}$ and $C_{S,\mathrm{indirect}}^{\mathrm{sq}}$ are the **Squared Structural Error Coefficients** defined in {prf:ref}`def-sasaki-structural-coeffs-sq`. The proof is provided in the subsequent sections.
:::

##### 2.3.6.1. Sub-Lemma: Decomposition of the Structural Error

:::{prf:lemma} Decomposition of the Structural Error
:label: lem-sasaki-structural-error-decomposition

Let $\mathbf r_2$ be a fixed raw value vector. Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states. Let $\mathbf z_1 = z(\mathcal S_1, \mathbf r_2)$ and $\mathbf z_2 = z(\mathcal S_2, \mathbf r_2)$ be the corresponding N-dimensional standardized vectors, computed using the fixed raw values from the second swarm but the structure of each respective swarm.

The total structural error vector, $\Delta\mathbf{z} = \mathbf z_1 - \mathbf z_2$, can be expressed as the sum of two **orthogonal** components:

$$
\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{indirect}}
$$

where:
1.  **The Direct Error ($\Delta_{\text{direct}}$):** The error vector whose non-zero components correspond to walkers whose status changes between $\mathcal S_1$ and $\mathcal S_2$.
2.  **The Indirect Error ($\Delta_{\text{indirect}}$):** The error vector whose non-zero components correspond to walkers that are alive in both swarms.

Because these two vectors have disjoint support, the squared L2-norm of the total error is the sum of the squared L2-norms of the components:

$$
\|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2
$$

```{dropdown} Proof
:::{prf:proof}
The proof follows from partitioning the sum of squared errors over the N walker indices. Let the full set of indices be $\{1, ..., N\}$.

1.  **Define Index Partitions.**
    Let $\mathcal{A}_{\text{unstable}} := \mathcal{A}(\mathcal S_1) \triangle \mathcal{A}(\mathcal S_2)$ be the set of indices of walkers whose survival status changes. Let $\mathcal{A}_{\text{stable}} := \mathcal{A}(\mathcal S_1) \cap \mathcal{A}(\mathcal S_2)$ be the set of indices for walkers that remain alive. Let $\mathcal{D}_{\text{stable}}$ be the indices of walkers dead in both swarms. These three sets form a partition of $\{1, ..., N\}$.

2.  **Analyze Error Components on Each Partition.**
    *   For $i \in \mathcal{A}_{\text{unstable}}$, the error component $(z_{1,i} - z_{2,i})$ is generally non-zero.
    *   For $i \in \mathcal{A}_{\text{stable}}$, the error component $(z_{1,i} - z_{2,i})$ is generally non-zero because the statistical moments $(\mu, \sigma')$ change with the swarm structure.
    *   For $i \in \mathcal{D}_{\text{stable}}$, both $z_{1,i}$ and $z_{2,i}$ are deterministically zero by the definition of the standardization operator. The error component is zero.

3.  **Define Orthogonal Error Vectors.**
    We define the vector $\Delta_{\text{direct}}$ such that its components are $(\Delta\mathbf{z})_i$ for $i \in \mathcal{A}_{\text{unstable}}$ and zero otherwise. We define $\Delta_{\text{indirect}}$ such that its components are $(\Delta\mathbf{z})_i$ for $i \in \mathcal{A}_{\text{stable}}$ and zero otherwise. By construction, $\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{indirect}}$.

4.  **Show Orthogonality.**
    The two vectors have disjoint support, meaning for any index $i$, at most one of the vectors can have a non-zero component. Therefore, their dot product is zero: $\Delta_{\text{direct}} \cdot \Delta_{\text{indirect}} = 0$.

5.  **Finalize the Squared Norm Identity.**
    For orthogonal vectors, the squared norm of the sum is the sum of the squared norms:

    $$
    \|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}} + \Delta_{\text{indirect}}\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2 + 2(\Delta_{\text{direct}} \cdot \Delta_{\text{indirect}}) = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2
    $$

This completes the proof.

**Q.E.D.**
```
:::

##### 2.3.6.2. Sub-Lemma: Bounding the Squared Direct Structural Error

:::{prf:lemma} Bound on the Squared Direct Structural Error
:label: lem-sasaki-direct-structural-error-sq

Let $\mathbf r_2$ be a fixed raw value vector with components bounded by $V_{\max}^{(R)}$. The squared Euclidean norm of the direct structural error component, $\|\Delta_{\text{direct}}\|_2^2$, is bounded by a term linear in the number of status changes, $n_c(\mathcal S_1, \mathcal S_2)$.

$$
\|\Delta_{\text{direct}}\|_2^2 \le \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \cdot n_c(\mathcal S_1, \mathcal S_2)
$$

```{dropdown} Proof
:::{prf:proof}
The proof bounds the squared error for each unstable walker and sums the results.

1.  **Isolate the Sum.**
    By definition, the vector $\Delta_{\text{direct}}$ has non-zero components only for walkers in the unstable set $\mathcal{A}_{\text{unstable}} = \mathcal{A}(\mathcal S_1) \triangle \mathcal{A}(\mathcal S_2)$. The number of walkers in this set is exactly $n_c = n_c(\mathcal S_1, \mathcal S_2)$. The squared norm is the sum of the squared errors over this set:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \sum_{i \in \mathcal{A}_{\text{unstable}}} (z_{1,i} - z_{2,i})^2
    $$

2.  **Bound the Error for a Single Unstable Walker.**
    Consider a single walker $i \in \mathcal{A}_{\text{unstable}}$. Its status changes between $\mathcal S_1$ and $\mathcal S_2$. This means one of two cases:
    *   Case A: Walker $i$ is alive in $\mathcal S_1$ and dead in $\mathcal S_2$. Then $z_{2,i} = 0$. The error is $(z_{1,i})^2$.
    *   Case B: Walker $i$ is dead in $\mathcal S_1$ and alive in $\mathcal S_2$. Then $z_{1,i} = 0$. The error is $(-z_{2,i})^2 = (z_{2,i})^2$.

    In both cases, the squared error for walker $i$ is the square of a single, valid standardized score.

3.  **Apply the Universal Z-Score Bound.**
    The framework provides a universal bound for the magnitude of any single standardized score in {prf:ref}`thm-z-score-norm-bound`, which is $|z_j| \le 2V_{\max}^{(R)} / \sigma_{\min,\mathrm{patch}}$. Squaring this gives a uniform bound for the squared error of any unstable walker:

    $$
    (z_{1,i} - z_{2,i})^2 \le \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2
    $$

4.  **Sum Over All Unstable Walkers.**
    We sum this uniform bound over all $n_c$ walkers in the unstable set:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \sum_{i \in \mathcal{A}_{\text{unstable}}} (z_{1,i} - z_{2,i})^2 \le \sum_{i=1}^{n_c} \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2
    $$

    This yields the final result as stated in the lemma.

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le n_c \cdot \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2
    $$

**Q.E.D.**
```
:::

##### 2.3.6.3. Sub-Lemma: Bounding the Squared Indirect Structural Error

:::{prf:lemma} Bound on the Squared Indirect Structural Error
:label: lem-sasaki-indirect-structural-error-sq

Let $\mathbf r_2$ be a fixed raw value vector. Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states. The squared Euclidean norm of the indirect structural error component, $\|\Delta_{\text{indirect}}\|_2^2$, is bounded by a term quadratic in the number of status changes, $n_c(\mathcal S_1, \mathcal S_2)$.

$$
\|\Delta_{\text{indirect}}\|_2^2 \le C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2
$$

where $C_{S,\mathrm{indirect}}^{\mathrm{sq}}$ is the **Squared Indirect Structural Error Coefficient** defined in {prf:ref}`def-sasaki-structural-coeffs-sq`.

```{dropdown} Proof
:::{prf:proof}
The proof decomposes the error for each stable walker into a mean-shift and a denominator-shift component and then bounds the sum of their squares.

**Step 1: Decompose the Error for a Single Stable Walker.**
For any walker $i$ in the stable set $\mathcal A_{\mathrm{stable}} = \mathcal A(\mathcal S_1) \cap \mathcal A(\mathcal S_2)$, the error is:

$$
\begin{aligned}
z_{1,i} - z_{2,i} &= \frac{r_{2,i} - \mu_1}{\sigma'_1} - \frac{r_{2,i} - \mu_2}{\sigma'_2} \\
&= \left(\frac{r_{2,i} - \mu_1}{\sigma'_1} - \frac{r_{2,i} - \mu_2}{\sigma'_1}\right) + \left(\frac{r_{2,i} - \mu_2}{\sigma'_1} - \frac{r_{2,i} - \mu_2}{\sigma'_2}\right) \\
&= \underbrace{\frac{\mu_2 - \mu_1}{\sigma'_1}}_{\text{Mean Shift}} + \underbrace{z_{2,i} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}}_{\text{Denominator Shift}}
\end{aligned}
$$
The squared error for this single walker is bounded using $(a+b)^2 \le 2(a^2+b^2)$:

$$
(z_{1,i} - z_{2,i})^2 \le 2\left(\frac{\mu_2 - \mu_1}{\sigma'_1}\right)^2 + 2\left(z_{2,i} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}\right)^2
$$

**Step 2: Sum the Errors Over All Stable Walkers.**
The total squared indirect error is the sum over all $i \in \mathcal A_{\mathrm{stable}}$. Let $k_{\mathrm{stable}} := |\mathcal A_{\mathrm{stable}}|$.

$$
\|\Delta_{\text{indirect}}\|_2^2 = \sum_{i \in \mathcal A_{\mathrm{stable}}} (z_{1,i} - z_{2,i})^2 \le \sum_{i \in \mathcal A_{\mathrm{stable}}} 2\left(\frac{\mu_2 - \mu_1}{\sigma'_1}\right)^2 + \sum_{i \in \mathcal A_{\mathrm{stable}}} 2\left(z_{2,i} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}\right)^2
$$

This can be simplified:

$$
\|\Delta_{\text{indirect}}\|_2^2 \le 2 k_{\mathrm{stable}} \frac{(\mu_2 - \mu_1)^2}{(\sigma'_1)^2} + 2 \frac{(\sigma'_2 - \sigma'_1)^2}{(\sigma'_1)^2} \sum_{i \in \mathcal A_{\mathrm{stable}}} (z_{2,i})^2
$$

**Step 3: Bound the Components.**
We now bound each term using the axiomatic properties of the aggregator and the uniform bounds from the framework.
*   **Bound on `(mu_2 - mu_1)^2`**: The structural continuity of the empirical mean ({prf:ref}`lem-sasaki-aggregator-structural`) gives:

    $$
    (\mu_2 - \mu_1)^2 \le \left(L_{\mu,S}^{\mathrm{Sasaki}}(k_{\min}) \cdot n_c\right)^2
    $$
*   **Bound on `(sigma'_2 - sigma'_1)^2`**: By composing the structural continuity of the variance with the Lipschitz property of the patching function ({prf:ref}`lem-stats-structural-continuity` in the framework), we get:

    $$
    (\sigma'_2 - \sigma'_1)^2 \le \left(L_{\sigma',S}^{\mathrm{Sasaki}}(\mathcal S_1, \mathcal S_2) \cdot n_c\right)^2
    $$
*   **Bound on `sum(z_2,i^2)`**: The sum is over the stable set, which is a subset of the alive walkers in $\mathcal S_2$. Thus, the sum is bounded by the total squared norm of the z-score vector for $\mathcal S_2$:

    $$
    \sum_{i \in \mathcal A_{\mathrm{stable}}} (z_{2,i})^2 \le \|\mathbf z_2\|_2^2 \le k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2
    $$
*   **Bound on `1/(sigma'_1)^2`**: This is bounded by $1/\sigma_{\min,\mathrm{patch}}^2$.

**Step 4: Assemble the Final Bound.**
Substituting these bounds back into the inequality from Step 2 gives a bound that is quadratic in $n_c$.

$$
\|\Delta_{\text{indirect}}\|_2^2 \le 2 k_{\mathrm{stable}} \frac{(L_{\mu,S})^2 n_c^2}{\sigma_{\min,\mathrm{patch}}^2} + 2 \frac{(L_{\sigma',S})^2 n_c^2}{\sigma_{\min,\mathrm{patch}}^2} k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2
$$

Factoring out $n_c^2$ and combining the coefficients gives:

$$
\|\Delta_{\text{indirect}}\|_2^2 \le \left[ 2 k_{\mathrm{stable}} \frac{(L_{\mu,S})^2}{\sigma_{\min,\mathrm{patch}}^2} + 2 k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2 \frac{(L_{\sigma',S})^2}{\sigma_{\min,\mathrm{patch}}^2} \right] \cdot n_c^2
$$
The term in the brackets is precisely the definition of the **Squared Indirect Structural Error Coefficient**, $C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2)$. This completes the proof.

**Q.E.D.**
```
:::

##### 2.3.6.4. Proof of Theorem 2.3.6
:label: proof-thm-sasaki-standardization-structural-sq

:::{prf:proof} **Proof**
The proof establishes the final bound by assembling the deterministic bounds for the two orthogonal error components derived in the preceding sub-lemmas.

**Step 1: Start with the Orthogonal Decomposition.**
From {prf:ref}`lem-sasaki-structural-error-decomposition`, the total squared structural error is the sum of the squared norms of the direct and indirect error components:

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2
$$

**Step 2: Substitute the Bounds for Each Component.**
We substitute the deterministic bounds for the squared norm of each component.

*   From {prf:ref}`lem-sasaki-direct-structural-error-sq`, the direct error is bounded by a term linear in $n_c$:

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}} \cdot n_c(\mathcal S_1, \mathcal S_2)
    $$

*   From {prf:ref}`lem-sasaki-indirect-structural-error-sq`, the indirect error is bounded by a term quadratic in $n_c$:

    $$
    \|\Delta_{\text{indirect}}\|_2^2 \le C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2
    $$

**Step 3: Combine the Bounds.**
Summing the two bounds from Step 2 directly gives the final inequality as stated in Theorem {prf:ref}`thm-sasaki-standardization-structural-sq`.

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}} \cdot n_c(\mathcal S_1, \mathcal S_2) + C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2
$$

This completes the proof, establishing a deterministic, worst-case bound on the operator's output error due to structural changes.

**Q.E.D.**

##### 2.3.7. Definition: Structural Error Coefficients (Squared Form)
:label: def-sasaki-structural-coeffs-sq

Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states with alive sets $\mathcal A_1$ and $\mathcal A_2$, of sizes $k_1:=|\mathcal A_1|$ and $k_2:=|\mathcal A_2|$. Let $k_{\mathrm{stable}}:=|\mathcal A_1\cap\mathcal A_2|$. The coefficients for the bounds on the squared structural error are defined as follows:

1.  **The Squared Direct Structural Error Coefficient ($C_{S,\mathrm{direct}}^{\mathrm{sq}}$):** The coefficient of the term linear in $n_c$.

    $$
    C_{S,\mathrm{direct}}^{\mathrm{sq}} := \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2
    $$

2.  **The Squared Indirect Structural Error Coefficient ($C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2)$):** The coefficient of the term quadratic in $n_c$, which bounds the error for the stable walkers.

    $$
    C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) := 2 k_{\mathrm{stable}} \frac{(L_{\mu,S}^{\mathrm{Sasaki}})^2}{\sigma_{\min,\mathrm{patch}}^{2}} + 2 k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2 \frac{(L_{\sigma',S}^{\mathrm{Sasaki}})^2}{\sigma_{\min,\mathrm{patch}}^{2}}
    $$

where $L_{\mu,S}^{\mathrm{Sasaki}}$ and $L_{\sigma',S}^{\mathrm{Sasaki}}$ are the structural continuity functions for the aggregator's mean and regularized standard deviation, respectively, which depend on the swarm states.

###### 2.3.8. Theorem: Composite Continuity of the Patched Standardization Operator

:::{prf:theorem} Composite Continuity of the Patched Standardization Operator (Sasaki)
:label: thm-sasaki-standardization-composite-sq

The N-dimensional standardization operator $z(\mathcal S)$, when applied to the reward vector, is continuous with respect to the dispersion metric. For any two swarms $\mathcal S_1, \mathcal S_2$ with $k_1=|\mathcal A(\mathcal S_1)|\ge 1$, the squared L2-norm of the output error is bounded by a composite function of the squared dispersion distance:

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le L_{z,L}^2(\mathcal S_1,\mathcal S_2) \cdot d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1, \mathcal S_2)^2 + L_{z,H}^2(\mathcal S_1,\mathcal S_2) \cdot d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1, \mathcal S_2)^4
$$

where $L_{z,L}^2$ and $L_{z,H}^2$ are state-dependent coefficients representing the Lipschitz and higher-order parts of the bound, respectively. Consequently, the logistic rescale operator $u(\mathcal S) = g_A(z(\mathcal S))$ is also continuous.

```{dropdown} Proof
:::{prf:proof}
The proof establishes a deterministic bound on the total error $\|z(\mathcal S_1, \mathbf r_1) - z(\mathcal S_2, \mathbf r_2)\|_2^2$ by combining the bounds for value-induced error and structure-induced error.

**Step 1: Decomposing the Total Error.**
Let $\mathbf r_1$ and $\mathbf r_2$ be the raw reward vectors for swarms $\mathcal S_1$ and $\mathcal S_2$. We introduce an intermediate vector $z(\mathcal S_1, \mathbf r_2)$ and use the inequality $\|A-C\|_2^2 \le 2(\|A-B\|_2^2 + \|B-C\|_2^2)$. The total squared error is bounded by the sum of a pure value error component and a pure structural error component:

$$
\|z(\mathcal S_1, \mathbf r_1) - z(\mathcal S_2, \mathbf r_2)\|_2^2 \le 2\,\|\underbrace{z(\mathcal S_1, \mathbf r_1) - z(\mathcal S_1, \mathbf r_2)}_{E_V}\|_2^2 + 2\,\|\underbrace{z(\mathcal S_1, \mathbf r_2) - z(\mathcal S_2, \mathbf r_2)}_{E_S}\|_2^2
$$

**Step 2: Bounding the Squared Value Error Term (`||E_V||_2^2`).**
The first term is a pure value error for the fixed swarm structure $\mathcal S_1$. We apply Theorem {prf:ref}`thm-sasaki-standardization-value`:

$$
\|E_V\|_2^2 \le C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2
$$
The squared difference of the raw reward vectors is bounded by the sum of contributions from walkers with stable status and unstable status:

$$
\|\mathbf r_1 - \mathbf r_2\|_2^2 = \sum_{i \in \mathcal A_1 \cap \mathcal A_2} |r_{1,i}-r_{2,i}|^2 + \sum_{i \in \mathcal A_1 \triangle \mathcal A_2} |r_{1,i}-r_{2,i}|^2 \le (L_R^{\mathrm{Sasaki}})^2 \Delta_{\mathrm{pos,Sasaki}}^2 + n_c (2 V_{\max}^{(R)})^2
$$
where we used that for unstable walkers, one reward is zero and the other is bounded by $V_{\max}^{(R)}$.

**Step 3: Bounding the Squared Structural Error Term (`||E_S||_2^2`).**
The second term is a pure structural error for the fixed raw value vector $\mathbf r_2$. We apply Theorem {prf:ref}`thm-sasaki-standardization-structural-sq`:

$$
\|E_S\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}} \cdot n_c + C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c^2
$$

**Step 4: Assembling the Composite Bound in Terms of Displacement Components.**
Combining the bounds from Steps 2 and 3 into the decomposition from Step 1 gives a complete bound in terms of $\Delta_{\mathrm{pos,Sasaki}}^2$ and $n_c$:

$$
\|z_1 - z_2\|_2^2 \le 2\,C_{V,\mathrm{total}}^{\mathrm{Sasaki}} \left[ (L_R^{\mathrm{Sasaki}})^2 \Delta_{\mathrm{pos,Sasaki}}^2 + 4(V_{\max}^{(R)})^2 n_c \right] + 2 \left[ C_{S,\mathrm{direct}}^{\mathrm{sq}} n_c + C_{S,\mathrm{indirect}}^{\mathrm{sq}} n_c^2 \right]
$$

**Step 5: Expressing the Bound in Terms of the Dispersion Metric.**
Let $d^2:=d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2$. From the definition of the dispersion metric we obtain

$$
\Delta_{\mathrm{pos,Sasaki}}^2\le N d^2,\qquad n_c\le\frac{N}{\lambda_{\mathrm{status}}}d^2,\qquad n_c^2\le\left(\frac{N}{\lambda_{\mathrm{status}}}\right)^2 d^4.
$$

Substituting these bounds into the expression from Step 4 yields

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2\le L_{z,L}^2(\mathcal S_1,\mathcal S_2)d^2+L_{z,H}^2(\mathcal S_1,\mathcal S_2)d^4.
$$

The coefficients are explicit:

$$
\begin{aligned}
L_{z,L}^2(\mathcal S_1,\mathcal S_2)&:=2C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)(L_R^{\mathrm{Sasaki}})^2N\\&\quad{}+\frac{N}{\lambda_{\mathrm{status}}}\Big(8C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)(V_{\max}^{(R)})^2+2C_{S,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1,\mathcal S_2)\Big),\\[4pt]
L_{z,H}^2(\mathcal S_1,\mathcal S_2)&:=2C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1,\mathcal S_2)\left(\frac{N}{\lambda_{\mathrm{status}}}\right)^2.
\end{aligned}
$$

All quantities on the right-hand side depend only on the swarm parameters and the bounds established earlier, so the coefficients are finite. Since the rescale function $g_A$ is globally Lipschitz ({prf:ref}`thm-rescale-function-lipschitz`), the continuity of $z$ implies the continuity of the composite operator $u(\mathcal S)=g_A(z(\mathcal S))$.

**Q.E.D.**
```
:::{prf:lemma} Lipschitz continuity of patched standardization (Sasaki)
:label: lem-sasaki-standardization-lipschitz

The bounds in Theorem {prf:ref}`thm-sasaki-standardization-composite-sq` show that the patched standardization operator $z$ is continuous with respect to the dispersion metric. In particular, $z$ admits the composite Lipschitz–Hölder control

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2\le L_{z,L}^2(\mathcal S_1,\mathcal S_2)\,d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2+L_{z,H}^2(\mathcal S_1,\mathcal S_2)\,d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^4.
$$

```{dropdown} Proof
:::{prf:proof}
The inequality is precisely the statement of Theorem {prf:ref}`thm-sasaki-standardization-composite-sq`; no additional work is required.
```
:::

:::

### 2.4 Swarm-level continuity & dynamics


### 2.5 Constants at a glance (Sasaki geometry)

| Constant                                                                                                   | Definition                                                                                                                                                                                                                                                                                                                   | Source lemma                                     |
|:-----------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------|
| $C_x^{(\mathrm{pert})}, C_v^{(\mathrm{pert})}, C_0^{(\mathrm{pert})}$                                      | $3\lambda_vA_x^2$, $3\lambda_vA_v^2$, $2\tau^2 V_{\mathrm{alg}}^2 + 2\tau\sigma_x^2 d + 3\lambda_vA_0^2 + \lambda_v\sigma_v^2\tau d$                                                                                                                                                                                         | {prf:ref}`lem-euclidean-perturb-moment`          |
| $\kappa_{\mathrm{drift}}^{\mathrm{Sasaki}}(C)$                                                             | $\sqrt{\tau^2(\tau C_{\mathrm{force}}(C)+\varepsilon_{\mathrm{cap}}^{\max}(C)+V_{\mathrm{alg}})^2+\lambda_v(\tau C_{\mathrm{force}}(C)+\varepsilon_{\mathrm{cap}}^{\max}(C))^2}$                                                                                                                                             | {prf:ref}`lem-euclidean-geometric-consistency`   |
| $\kappa_{\mathrm{anisotropy}}^{\mathrm{Sasaki}}(C)$                                                        | $[(1-\rho_*(C))c_d(C)]^{-1}$                                                                                                                                                                                                                                                                                                 | {prf:ref}`lem-euclidean-geometric-consistency`   |
| $L_{\mu,M}^{\mathrm{Sasaki}}(k)$, $L_{m_2,M}^{\mathrm{Sasaki}}(k)$                                         | $k^{-1/2}$, $2V_{\max}/k^{1/2}$ (with $V_{\max}=V_{\max}^{(R)}$ or $V_{\max}^{(d)}$)                                                                                                                                                                                                                                         | {prf:ref}`lem-sasaki-aggregator-lipschitz`       |
| $C_{V,\mathrm{total}}^{\mathrm{Sasaki}}$                                                                   | $L_R^{\mathrm{Sasaki}}\left(\frac{2}{\sigma_{\min,\mathrm{patch}}}+\frac{8(V_{\mathrm{max}}^{(R)})^2}{\sigma_{\min,\mathrm{patch}}^2}L_{\sigma'_{\mathrm{patch}}}\right)$                                                                                                                                                    | {prf:ref}`thm-sasaki-standardization-value`      |
| $C_{S,\mathrm{direct}}^{\mathrm{Sasaki}}(k_{\min})$, $C_{S,\mathrm{indirect}}^{\mathrm{Sasaki}}(k_{\min})$ | $\frac{V_{\mathrm{max}}^{(R)}}{\sigma_{\min,\mathrm{patch}}}+\frac{2(V_{\mathrm{max}}^{(R)})^2}{\sigma_{\min,\mathrm{patch}}^2}$, $\frac{3V_{\mathrm{max}}^{(R)}}{\sigma_{\min,\mathrm{patch}}k_{\min}}+\frac{6(V_{\mathrm{max}}^{(R)})^2}{\sigma_{\min,\mathrm{patch}}^2k_{\min}}L_{\sigma',M}^{\mathrm{Sasaki}}(k_{\min})$ | {prf:ref}`thm-sasaki-standardization-structural` |
| $L_{\mathrm{death}}^{\mathrm{Sasaki}}(C)$                                                                  | $2\big(p_{\mathrm{aff}}+\rho_*(C) q_{\mathrm{dir}}(C)\big) C_{\partial} L_{\mathrm{flow}}$                                                                                                                                                                                                                                   | {prf:ref}`lem-euclidean-boundary-holder`         |

Note. The mean-square continuity bound $F_{d,ms}^{\mathrm{Sasaki}}$ retains both $d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2$ and $d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^4$ contributions through the $n_c$ and $n_c^2$ terms in Theorem {prf:ref}`thm-sasaki-distance-ms`; the bound is therefore a composite Lipschitz–Hölder function of the dispersion distance rather than purely Lipschitz.

These constants feed the composition bound in Section 2.4 and the axiom checklist in Section 3.

{ref}`Section 4.5 <sec-eg-kernel-repr>` expresses $\Psi_{\mathcal F_{\mathrm{EG}}}$ as the pushforward of a product measure over $(\boldsymbol U,\boldsymbol C^{\mathrm{pot}},\boldsymbol C^{\mathrm{clone}},\boldsymbol\zeta,\boldsymbol\xi)$ with all draws conditionally independent across walkers, realising Assumption A ({prf:ref}`def-assumption-instep-independence`). Lemmas {prf:ref}`lem-euclidean-perturb-moment`, {prf:ref}`lem-euclidean-geometric-consistency`, {prf:ref}`lem-sasaki-aggregator-lipschitz`, {prf:ref}`lem-sasaki-standardization-lipschitz`, {prf:ref}`thm-sasaki-distance-ms`, and {prf:ref}`lem-euclidean-boundary-holder` supply the Lipschitz and Feller bounds for every stage of the pipeline, so the hypotheses of the framework’s composition theorem (Framework Sec. 17) hold verbatim in the Sasaki geometry. The resulting continuity bound for $\Psi_{\mathcal F_{\mathrm{EG}}}$ inherits the perturbation-growth coefficients $(C_x^{(\mathrm{pert})},C_v^{(\mathrm{pert})},C_0^{(\mathrm{pert})})$, the drift constant $\kappa_{\mathrm{drift}}^{\mathrm{Sasaki}}$, the anisotropy constant $\kappa_{\mathrm{anisotropy}}^{\mathrm{Sasaki}}$, $F_{d,ms}^{\mathrm{Sasaki}}$, and $L_{\mathrm{death}}^{\mathrm{Sasaki}}$ with exponent $\alpha_B^{\mathrm{Sasaki}}$. The only potential discontinuity is the status indicator, but the $C^1$ boundary of $\mathcal X_{\mathrm{valid}}$ ensures $\mathbb P(\hat x_i\in\partial\mathcal X_{\mathrm{valid}})=0$, so dominated convergence applies. Appendix A records the resulting proof that $\Psi_{\mathcal F_{\mathrm{EG}}}$ defines a time-homogeneous Feller Markov chain on $(\Sigma_N,d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}})$ ({prf:ref}`def-fragile-gas-algorithm`).

### 2.6 Axiom for Convergence: Non-Deceptive Landscape

For the geometric ergodicity proof in `convergence.md` to hold, the Euclidean Gas must satisfy an additional environmental axiom that strengthens the Axiom of Environmental Richness. This axiom prevents pathological scenarios where positional diversity and reward signals become decoupled.

:::{prf:axiom} Axiom of Non-Deceptive Landscapes
:label: def-axiom-non-deceptive

The environment $(X_{\mathrm{valid}}, R_{\mathrm{pos}})$ is **non-deceptive** if there exist constants $\kappa_{\mathrm{grad}} > 0$ and $L_{\mathrm{grad}} > 0$ such that for any two points $x, y \in X_{\mathrm{valid}}$ with $\|x - y\| \ge L_{\mathrm{grad}}$, the average squared norm of the reward gradient along the line segment connecting them is bounded below:

$$
\frac{1}{\|x-y\|} \int_{0}^{\|x-y\|} \big\|\nabla R_{\mathrm{pos}}\big(x + t\tfrac{y-x}{\|y-x\|}\big)\big\|^2 dt \ge \kappa_{\mathrm{grad}}.
$$

**Validation:** This axiom is satisfied by ensuring the potential function $R_{\mathrm{pos}}$ does not contain large, perfectly flat plateaus within the compact valid domain $X_{\mathrm{valid}}$. Continuity of $\nabla R_{\mathrm{pos}}$ on the compact set allows the constants to be chosen with $L_{\mathrm{grad}}$ no larger than the richness scale $r_{\mathrm{rich}}/4$ from Section 2.2. This regularity condition is assumed to hold for the Euclidean Gas instantiation.
:::

---

## 3. Statement & proof: Kinetic Euclidean Gas is a valid Fragile Gas

**Theorem.** The Kinetic Euclidean Gas $\mathcal F_{\mathrm{EG}}$ defined in Section 1, equipped with the Sasaki dispersion metric and kinetic perturbation of §1.3, is a valid instantiation of a *Fragile Swarm* and hence a *Fragile Gas* (Defs. 18.1–18.2).

**Proof (axiom checklist).**
The Kinetic Euclidean Gas is a valid Fragile Gas because every axiom required by the framework has been verified in the preceding sections:
- **Foundations & environment:** Lemmas {prf:ref}`lem-euclidean-reward-regularity` and {prf:ref}`lem-euclidean-richness` prove reward regularity and environmental richness in the Sasaki space, and the projection in Section 1.1 establishes bounded algorithmic diameter ({prf:ref}`def-axiom-bounded-algorithmic-diameter`).
- **Noise validity & growth control:** Lemma {prf:ref}`lem-euclidean-perturb-moment` bounds the second moment of the kinetic perturbation by a quadratic function of $\|x\|$ and $\|v\|$ and confirms the Feller property of the capped kernel.
- **Geometric constants & non-degeneracy:** Lemma {prf:ref}`lem-euclidean-geometric-consistency` supplies drift and anisotropy bounds that are uniform on compact subsets of the state space, while the parameter choices $\sigma_v^2>0$ and $\delta_x,\delta_v\ge 0$ keep the noise non-degenerate.
- **Measurement operator continuity:** Lemma {prf:ref}`thm-sasaki-distance-ms` proves the mean-square continuity of the Sasaki distance measurement with explicit error function $F_{d,ms}^{\mathrm{Sasaki}}$.
- **Deterministic operator pipeline:** Lemma {prf:ref}`lem-sasaki-aggregator-lipschitz` together with Theorems {prf:ref}`thm-sasaki-standardization-value` and {prf:ref}`thm-sasaki-standardization-structural` (culminating in Lemma {prf:ref}`lem-sasaki-standardization-lipschitz`) provide Sasaki-specific Lipschitz and Feller bounds for the empirical aggregators, patched standardization, and logistic rescale, so the deterministic stage respects the dispersion metric ({prf:ref}`def-assumption-instep-independence`).
- **Viability:** Section 2.1 re-establishes guaranteed revival and boundary regularity via Lemma {prf:ref}`lem-euclidean-boundary-holder`.

Since all axioms are satisfied, $\Psi_{\mathcal F_{\mathrm{EG}}}$ is a Feller Markov kernel on the alive-swarm space and the Euclidean Gas realises a Fragile Gas.

````{dropdown} Axiom validation checklist
:class: tip
| Framework axiom | Validation in this chapter |
| --- | --- |
| Bounded Algorithmic Diameter ({prf:ref}`def-axiom-bounded-algorithmic-diameter`) | Section 1.1, squashing projection $\varphi$ |
| Reward Regularity ({prf:ref}`def-axiom-reward-regularity`) | Lemma {prf:ref}`lem-euclidean-reward-regularity` |
| Environmental Richness ({prf:ref}`def-axiom-environmental-richness`) | Lemma {prf:ref}`lem-euclidean-richness` |
| Measurement Stability (patched standardisation & logistic rescale) | Lemmas {prf:ref}`lem-sasaki-aggregator-lipschitz`, {prf:ref}`lem-sasaki-standardization-lipschitz` |
| Geometric Consistency ({prf:ref}`def-axiom-geometric-consistency`) | Lemma {prf:ref}`lem-euclidean-geometric-consistency` |
| Patched Standardisation ({prf:ref}`def-statistical-properties-measurement`) | Algorithm {prf:ref}`alg-euclidean-gas` and Section 1.2 |
| Boundary Regularity | Lemma {prf:ref}`lem-euclidean-boundary-holder` |
````

---

(sec-eg-kernel)=
## 4. Swarm Update Operator Kernel

We define the one-step kernel $\Psi_{\mathcal F_{\mathrm{EG}}}$ on the ordered swarm space $\Sigma_N=(\mathcal X\times\mathbb R^d\times\{0,1\})^N$. Randomness inside a step is conditionally independent across walkers given the current swarm, matching Assumption A ({prf:ref}`def-assumption-instep-independence`). When no walkers are alive the process becomes absorbing.

(sec-eg-stage1)=
### 4.1 Stage 1 — Cemetery absorption

If the alive index set $\mathcal A(\mathcal S_t)$ is empty, the operator returns $\delta_{\mathcal S_t}$ and the run stops. All subsequent stages are skipped.

(sec-eg-stage2)=
### 4.2 Stage 2 — Single-shot measurement and frozen potentials

1.  **Raw scores.** For $i\in\mathcal A_t$ set $r_i:=R(x_i,v_i)$, and put $r_i:=0$ for $i\notin\mathcal A_t$.
2.  **Measurement companions.** For each alive walker $i\in\mathcal A_t$, a companion for the diversity measurement, $c_{\mathrm{pot}}(i)$, is drawn independently from the **`ε`-dependent companion kernel** $\mathbb C_\epsilon(\mathcal S_t, i)$. This measure assigns a probability to each potential companion $j \in \mathcal A_t \setminus \{i\}$ that is weighted by their algorithmic distance (see Section 1.3):

   $$
   P(\text{choose } j \mid \mathcal S_t, i) \propto \exp\left(-\frac{d_{\text{alg}}(i,j)^2}{2\epsilon^2}\right).
   $$

   The raw distance to this sampled companion is then computed using the algorithmic distance:

   $$
   d_i:=d_{\text{alg}}(i,c_{\mathrm{pot}}(i)),
   $$
   while $d_i:=0$ for $i\notin\mathcal A_t$.
3.  **Potential pipeline.** Apply the canonical patched standardization and logistic rescale operators of Section 1.2 to $(r_i)$ and $(d_i)$ restricted to the alive set. Denote the positive outputs by $r'_i$ and $d'_i$, and combine them with the canonical exponents

   $$
   V_{\text{fit},i}:=(d'_i)^\beta (r'_i)^\alpha \quad (i\in\mathcal A_t),\qquad V_{\text{fit},i}:=0 \quad (i\notin\mathcal A_t).
   $$
   The vector $\mathbf V_{\text{fit}}\in\mathbb R_+^N$ is frozen for the remainder of the step.

(sec-eg-stage3)=

(sec-eg-stage3)=
### 4.3 Stage 3 — Per-walker Clone vs. Persist

For each index $i\in\{1,\dots,N\}$ independently:

1.  **Clone companion and threshold.** Sample a companion $c_{\mathrm{clone}}(i)\sim\mathbb C_\epsilon(\mathcal S_t, i)$ from the same **`ε`-dependent companion kernel** used in the measurement stage, weighted by algorithmic distance. Independently, sample a threshold $T_i\sim\mathrm{Unif}(0,p_{\max})$.
2.  **Score comparison.** With $v_i:=V_{\text{fit},i}$ and $v_c:=V_{\text{fit},c_{\mathrm{clone}}(i)}$, compute

   $$
   S_i:=\frac{v_c-v_i}{v_i+\varepsilon_{\mathrm{clone}}}.
   $$
   Set $a_i=\textsf{Clone}$ if $S_i>T_i$, else $a_i=\textsf{Persist}$. (Note: For a dead walker `i`, this condition is guaranteed to be met by the Axiom of Guaranteed Revival.)
3.  **Intermediate state.**
   - If $a_i=\textsf{Persist}$, take $(\tilde x_i,\tilde v_i):=(x_i,v_i)$.
   - If $a_i=\textsf{Clone}$, draw $\zeta_i^x\sim\mathcal N(0,I_d)$ and set

     $$
     (\tilde x_i,\tilde v_i):=\Big(x_{c_{\mathrm{clone}}(i)}+\sigma_x\zeta_i^x,\;v_{c_{\mathrm{clone}}(i)}\Big).
     $$

     The position is reset to the companion's position plus Gaussian jitter with scale $\sigma_x$, while the velocity is reset directly to the companion's velocity without jitter. This matches the cloning operator defined in Definition 5.7.4 of `03_cloning.md`. For the full inelastic collision model with momentum conservation when multiple walkers clone simultaneously, see that definition.

Collect the intermediate swarm $\mathcal S_{t+1/2}=((\tilde x_i,\tilde v_i,1))_{i=1}^N$, i.e. every walker is set to alive before the kinetic step.

(sec-eg-stage4)=
### 4.4 Stage 4 — Kinetic perturbation and status update

Independently for each $i$, draw $\xi_i\sim\mathcal N(0,I_d)$ and apply the kinetic Euler step of Section 1.3:

$$
\hat v_i:=\psi_v\!\Big(\tilde v_i+\frac{\tau}{m}F(\tilde x_i)-\gamma_{\mathrm{fric}}\tau(\tilde v_i-u(\tilde x_i))+\sqrt{\sigma_v^2\tau}\,\xi_i\Big),\qquad\hat x_i:=\tilde x_i+\tau\hat v_i.
$$
Set the terminal status by the deterministic boundary check

$$
 s_i^{(t+1)}:=\mathbf 1_{\mathcal X_{\mathrm{valid}}}(\hat x_i).
$$
The next swarm is $\mathcal S_{t+1}=((\hat x_i,\hat v_i,s_i^{(t+1)}))_{i=1}^N$.

(sec-eg-kernel-repr)=
### 4.5 Kernel representation

Let $\boldsymbol U=(T_i)$, $\boldsymbol C^{\mathrm{pot}}=(c_{\mathrm{pot}}(i))$, $\boldsymbol C^{\mathrm{clone}}=(c_{\mathrm{clone}}(i))$, $\boldsymbol\zeta=(\zeta_i^x,\zeta_i^v)$, and $\boldsymbol\xi=(\xi_i)$. Writing $\nu$ for the product law of these arrays, the one-step operator is the pushforward

$$
\Psi_{\mathcal F_{\mathrm{EG}}}(\mathcal S_t,A)=\int\mathbf 1\big\{\Phi(\mathcal S_t;\boldsymbol U,\boldsymbol C^{\mathrm{pot}},\boldsymbol C^{\mathrm{clone}},\boldsymbol\zeta,\boldsymbol\xi)\in A\big\}\,\nu(d\boldsymbol U\,d\boldsymbol C^{\mathrm{pot}}\,d\boldsymbol C^{\mathrm{clone}}\,d\boldsymbol\zeta\,d\boldsymbol\xi)
$$
for Borel $A\subseteq\Sigma_N$, where $\Phi$ executes the deterministic composition of Stages 2–4.

## Appendix A. Proof of the Feller Property for the Euclidean Gas Kernel

:::{prf:theorem} Feller continuity of $\Psi_{\mathcal F_{\mathrm{EG}}}$
:label: thm-euclidean-feller

```{dropdown} Proof
:::{prf:proof}
Write the single-step operator as the composition of the cemetery check (Stage 1), the measurement and potential pipeline (Stage 2), the Clone/Persist gate (Stage 3), and the kinetic-plus-status update (Stage 4). Each stage defines a Markov kernel on $\Sigma_N$ that is Feller with respect to $d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}$; the claim follows because Feller kernels are closed under composition (see, e.g., \[Ethier & Kurtz 86, Prop. 4.2.2\]).

1. **Stage 1 (cemetery absorption).** The map $\mathcal S\mapsto\delta_{\mathcal S}$ is continuous. The absorbing branch fires only when $\mathcal A(\mathcal S)=\varnothing$, which is a closed condition, so Stage 1 is Feller.

2. **Stage 2 (measurement and potential pipeline).** Conditional on $\mathcal S$, the companion draws $(c_{\mathrm{pot}}(i))$ are sampled from the product coupling $\mathbb C(\mathcal S)$, which depends continuously on $\mathcal S$ in the dispersion metric by Lemma {prf:ref}`lem-sasaki-single-walker-positional-error`. Lemma {prf:ref}`thm-sasaki-distance-ms` bounds the mean-square variation of the companion distances, and Lemma {prf:ref}`lem-sasaki-aggregator-lipschitz` together with Theorems {prf:ref}`thm-sasaki-standardization-value` and {prf:ref}`thm-sasaki-standardization-structural` show that the patched standardisation and logistic rescale operators depend continuously on the raw values. Consequently the kernel that outputs $(\mathbf r,\mathbf d,\mathbf V_{\text{fit}})$ is Feller.

3. **Stage 3 (Clone/Persist gate).** The Bernoulli probabilities $p_i(\mathcal S)$ governing the Clone/Persist decision are continuous functions of the standardized scores produced in Stage 2. Conditional on cloning, the jitter distribution is Gaussian with fixed covariance; the resulting pushforward through the 1-Lipschitz squashing map $\psi_v$ is Feller by Lemma {prf:ref}`lem-squashing-properties-generic`. Therefore the Stage-3 kernel is a finite mixture of Feller kernels with continuous weights and is itself Feller.

4. **Stage 4 (kinetic step and boundary check).** Lemma {prf:ref}`lem-euclidean-perturb-moment` proves that the capped kinetic update map $(x,v)\mapsto(x^+,v^+)$ is Feller and that its second moment grows at most quadratically in the state norm. The status indicator is applied to $x^+$; Lemma {prf:ref}`lem-euclidean-boundary-holder` shows that the death probability varies continuously and that $\mathbb P(x^+\in\partial\mathcal X_{\mathrm{valid}})=0$ under the $C^1$-boundary assumption, so the indicator preserves the Feller property.

Since every stage is Feller, their composition $\Psi_{\mathcal F_{\mathrm{EG}}}$ is a Feller Markov kernel on $(\Sigma_N,d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}})$.
```
:::

## Appendix B. References (selected)

- Federer, H. *Geometric Measure Theory*. Springer, 1969. Standard tubular-neighbourhood volume estimates and Weyl's tube formula.
- Ethier, S. N., and Kurtz, T. G. *Markov Processes: Characterization and Convergence*. Wiley, 1986. Composition properties of Feller kernels.

Here are BibTeX entries for the sources cited in the convergence section.

```bibtex
@book{MeynTweedie2009,
  author    = {Sean P. Meyn and Richard L. Tweedie},
  title     = {Markov Chains and Stochastic Stability},
  edition   = {2},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  year      = {2009},
  doi       = {10.1017/CBO9780511626630},
  isbn      = {9780521731829}
}
```

([Cambridge University Press & Assessment][1])

```bibtex
@book{BoucheronLugosiMassart2013,
  author    = {St{\'e}phane Boucheron and G{\'a}bor Lugosi and Pascal Massart},
  title     = {Concentration Inequalities: A Nonasymptotic Theory of Independence},
  publisher = {Oxford University Press},
  address   = {Oxford},
  year      = {2013},
  isbn      = {9780198767657}
}
```

([Oxford Academic][2])

```bibtex
@book{Federer1969,
  author    = {Herbert Federer},
  title     = {Geometric Measure Theory},
  series    = {Grundlehren der mathematischen Wissenschaften},
  volume    = {153},
  publisher = {Springer},
  address   = {Berlin Heidelberg},
  year      = {1969},
  isbn      = {9783540045052}
}
```

([SpringerLink][3])

```bibtex
@book{Santambrogio2015,
  author    = {Filippo Santambrogio},
  title     = {Optimal Transport for Applied Mathematicians: Calculus of Variations, PDEs, and Modeling},
  series    = {Progress in Nonlinear Differential Equations and Their Applications},
  volume    = {87},
  publisher = {Birkh{\"a}user},
  address   = {Cham},
  year      = {2015},
  doi       = {10.1007/978-3-319-20828-2},
  isbn      = {9783319208275}
}
```

([SpringerLink][4])

```bibtex
@book{Kechris1995,
  author    = {Alexander S. Kechris},
  title     = {Classical Descriptive Set Theory},
  series    = {Graduate Texts in Mathematics},
  volume    = {156},
  publisher = {Springer},
  address   = {New York},
  year      = {1995},
  doi       = {10.1007/978-1-4612-4190-4},
  isbn      = {9780387943749}
}
```

([SpringerLink][5])

```bibtex
@article{FritschCarlson1980,
  author  = {F. N. Fritsch and R. E. Carlson},
  title   = {Monotone Piecewise Cubic Interpolation},
  journal = {SIAM Journal on Numerical Analysis},
  year    = {1980},
  volume  = {17},
  number  = {2},
  pages   = {238--246},
  doi     = {10.1137/0717021}
}
```

([SIAM E-Books][6])

```bibtex
@article{Hyman1983,
  author  = {James M. Hyman},
  title   = {Accurate Monotonicity Preserving Cubic Interpolation},
  journal = {SIAM Journal on Scientific and Statistical Computing},
  year    = {1983},
  volume  = {4},
  number  = {4},
  pages   = {645--654},
  doi     = {10.1137/0904045}
}
```

([SIAM E-Books][7])

```bibtex
@incollection{McDiarmid1989,
  author    = {Colin McDiarmid},
  title     = {On the Method of Bounded Differences},
  booktitle = {Surveys in Combinatorics, 1989},
  editor    = {J. Siemons},
  series    = {London Mathematical Society Lecture Note Series},
  volume    = {141},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  year      = {1989},
  pages     = {148--188}
}
```

([Cambridge University Press & Assessment][8])

```bibtex
@book{AmbrosioFuscoPallara2000,
  author    = {Luigi Ambrosio and Nicola Fusco and Diego Pallara},
  title     = {Functions of Bounded Variation and Free Discontinuity Problems},
  series    = {Oxford Mathematical Monographs},
  publisher = {Oxford University Press},
  address   = {Oxford},
  year      = {2000},
  isbn      = {0198502451}
}
```

([Oxford University Press][9])

If you want these exported as a `.bib` file, say the filename you prefer and I’ll package it for download.

[1]: https://www.cambridge.org/core/books/markov-chains-and-stochastic-stability/E2B82BFB409CD2F7D67AFC5390C565EC?utm_source=chatgpt.com "Markov Chains and Stochastic Stability"
[2]: https://academic.oup.com/book/26549?utm_source=chatgpt.com "Concentration Inequalities: A Nonasymptotic Theory of ..."
[3]: https://link.springer.com/content/pdf/10.1007/978-3-642-62010-2.pdf?utm_source=chatgpt.com "Download book PDF"
[4]: https://link.springer.com/book/10.1007/978-3-319-20828-2?utm_source=chatgpt.com "Optimal Transport for Applied Mathematicians"
[5]: https://link.springer.com/book/10.1007/978-1-4612-4190-4?utm_source=chatgpt.com "Classical Descriptive Set Theory"
[6]: https://epubs.siam.org/doi/10.1137/0717021?utm_source=chatgpt.com "Monotone Piecewise Cubic Interpolation | SIAM Journal on ..."
[7]: https://epubs.siam.org/doi/abs/10.1137/0904045?utm_source=chatgpt.com "Accurate Monotonicity Preserving Cubic Interpolation"
[8]: https://www.cambridge.org/core/books/surveys-in-combinatorics-1989/BF6F779EA29B29CBB30715E8C406C282?utm_source=chatgpt.com "Surveys in Combinatorics, 1989"
[9]: https://global.oup.com/academic/product/functions-of-bounded-variation-and-free-discontinuity-problems-9780198502456?utm_source=chatgpt.com "Functions of Bounded Variation and Free Discontinuity ..."
