# Formalization of the General Algorithmic Search (GAS) Algorithm

## Introduction

The **General Algorithmic Search (GAS)** is a metaheuristic algorithm for global optimization proposed by Hernández, Durán, and Amigó. It is a **stochastic, population-based search** method that evolves a swarm of agents (called *walkers*) cooperatively to locate a global extremum (minimum or maximum) of a given objective function. GAS draws inspiration from collective intelligence and incorporates concepts from Tabu Search and evolutionary algorithms. Notably, GAS introduces an internal state called **flow** for each walker and a mechanism of **cloning** (replicating states between agents) to enhance exploration. Empirical benchmarks have shown that GAS can outperform classic metaheuristics like Basin Hopping (BH), Cuckoo Search (CS), and Differential Evolution (DE), especially when multiple independent runs are executed in parallel (concurrent optimization). In this report, we formalize the GAS algorithm with rigorous mathematical notation and then analyze its theoretical properties, including convergence behavior and connections to principles of *Fractal AI* theory (multi-scale search and entropy-driven optimization).

## Problem Definition and Notation

GAS addresses the **unconstrained global optimization** problem for a real-valued function. Formally, let \$f: \mathbb{R}^d \to \mathbb{R}\$ be a continuous objective function. We seek a global minimizer \$x^\* \in D\$ (within a compact search domain \$D \subset \mathbb{R}^d\$) such that:

$x^* \;=\; \arg\min_{x \in D} f(x). \tag{2.1}$

*(For maximization problems, a similar formulation applies by considering \$-f\$ as the objective.)* The compactness of \$D\$ (closed and bounded set) ensures that a global minimum \$x^*\$ exists. We denote \$f^* = f(x^*)\$ as the optimal function value. The goal of GAS is to find or approximate \$x^*\$ through iterative randomized search.

**Swarm state:** GAS maintains a population (swarm) of \$N\$ candidate solutions called **walkers**. At iteration (time) \$t\$, each walker \$i \in {1,2,\dots,N}\$ has:

* A **position** \$x\_i(t) \in D\$ in the search space. We write \$x\_i = (x\_{i}^{(1)}, ..., x\_{i}^{(d)})\$ for the coordinates of walker \$i\$.
* An **internal flow value** \$F\_i(t) > 0\$ which influences its behavior (defined below).

Additionally, GAS maintains a **tabu memory list** \$T(t) = {,t\_1(t), t\_2(t), \dots, t\_N(t),}\$ of size \$N\$. This memory stores positions of local minima found during search (serving a role analogous to Tabu Search memory). Initially, all entries of \$T\$ may be set to some found local optimum (see Initialization step). The algorithm also tracks a current **best solution** found, denoted \$\textbf{BEST}(t)\$, which is the lowest-\$f\$ point seen so far.

To formalize the dynamic, we will describe one full iteration of the GAS algorithm, which consists of several phases: (1) Flow computation and walker cloning, (2) Local search and tabu memory update, (3) (Optional) Halt check, and (4) Walker position update. These steps repeat until a stopping criterion is met.

## GAS Algorithm Definition

### Initialization

**Step 0:** At \$t=0\$, initialize the swarm and memory:

* **0.1 Initial positions:** For each walker \$i\$, sample an initial position \$x\_i(0)\$ randomly from the domain \$D\$ (e.g. uniformly at random). This provides a diverse starting population.
* **0.2 Evaluate fitness:** Compute \$f(x\_i(0))\$ for each \$i\$, and identify the index \$i\_{\min}\$ of the best (lowest) objective value in the swarm: \$i\_{\min} := \arg\min\_{1\le i\le N} f(x\_i(0))\$. Let \$x\_{\min}(0) = x\_{i\_{\min}}(0)\$ and \$f\_{\min}(0) = f(x\_{\min}(0))\$.
* **0.3 Local refinement:** Use a local optimization method (the authors use **L-BFGS-B** \[10], a quasi-Newton bounded-memory solver) initialized at \$x\_{\min}(0)\$ to find a nearby local minimum. Let \$x^{}\_{{\rm loc}}\$ denote the resulting local minimizer (this step improves the best candidate).
* **0.4 Initialize tabu list:** Set every entry of the tabu memory list \$T(0)\$ to this found local minimum: \$t\_r(0) := x^{}\_{{\rm loc}}\$ for all \$r = 1,\dots,N\$. (This effectively flags that position as “tabu” for the swarm initially, preventing walkers from lingering there in the first iteration.)
* **0.5 Record best:** Set \$\textbf{BEST}(0) := x^{}*{{\rm loc}}\$ as the best solution found so far. Also store \$f^{}*{{\rm BEST}}(0) := f(x^{}\_{{\rm loc}})\$.

At this point, the algorithm enters the main loop (iterative steps) to evolve the swarm and search for the global optimum.

### Walker Flow Computation and Cloning (Exploration Step)

**Step 1:** In each iteration \$t \ge 1\$, GAS first computes the **flow values** \$F\_i\$ for walkers and performs a cloning operation to allow walkers to jump to promising regions. This step introduces directed **exploration** by probabilistic information sharing among walkers. The procedure is as follows:

1. **Compute relative fitness scale:** Determine the current best and worst objective values in the swarm: \$f\_{\min}(t) = \min\_i f(x\_i(t))\$ and \$f\_{\max}(t) = \max\_i f(x\_i(t))\$. For each walker \$i\$, define a normalized *fitness level* (for a minimization problem)
   $\displaystyle \phi_i(t) \;=\; \frac{\,f(x_i(t)) \;-\; f_{\min}(t)\,}{\,f_{\max}(t) \;-\; f_{\min}(t)\,}\,, \tag{2.2}$
   so that \$\phi\_i(t) \in \[0,1]\$. Here \$\phi\_i=0\$ indicates the current best walker and \$\phi\_i=1\$ would correspond to the worst in the swarm. This scaling \$\phi\_i\$ measures how far \$x\_i\$ is from the swarm’s best in terms of objective value. (If \$f\_{\max} = f\_{\min}\$, *i.e.* all walkers have identical \$f\$, one may set all \$\phi\_i=1\$ for this step to avoid division by zero.)

2. **Random pair selection:** Each walker \$i\$ randomly selects **one other** distinct walker \$j \neq i\$ uniformly from the population. Let this random choice be denoted \$j = \mathcal{J}(i,t)\$. Compute the squared Euclidean **distance** between their positions:
   $d_{i,j}^2(t) \;=\; \|\,x_i(t) - x_j(t)\,\|^2 \;=\; \sum_{n=1}^d \Big(x_i^{(n)}(t) - x_j^{(n)}(t)\Big)^2. \tag{2.3}$
   Intuitively, \$d\_{i,j}^2\$ represents how far walker \$i\$ is from another randomly chosen walker \$j\$ in the solution space. This encourages **diffusion**: interactions can occur between distant parts of the swarm.

3. **Random tabu memory selection:** Each walker \$i\$ also randomly selects an index \$r \in {1,\dots,N}\$ from the tabu memory list. Denote this choice \$r = \mathcal{M}(i,t)\$. Compute the squared distance from walker \$i\$ to that memory location:
   $\delta_{i,r}^2(t) \;=\; \|\,x_i(t) - t_r(t)\,\|^2. \tag{2.4}$
   If by chance the memory \$t\_r(t)\$ equals the walker’s own position \$x\_i(t)\$ (i.e. the walker is exactly at a tabu memory point), we define \$\delta\_{i,r}^2(t) := 1\$ as a floor value. (This convention avoids a degenerate case of zero distance, and effectively treats being at a known tabu point as a *moderate* distance – ensuring such a walker’s flow isn’t artificially low.)

4. **Flow value calculation:** Compute the **flow** \$F\_i(t)\$ for each walker \$i\$. The flow is defined as:
   $\displaystyle F_i(t) \;=\; \Big(\phi_i(t) + 1\Big)^2 \;\cdot\; d_{i,\mathcal{J}(i,t)}^2(t) \;\cdot\; \delta_{i,\mathcal{M}(i,t)}^2(t)\,. \tag{2.5}$
   This formula encapsulates three factors:

   * \$\phi\_i(t) + 1 \in \[1,2]\$ increases with the walker’s relative objective value. A walker with higher \$f(x)\$ (worse solution) has larger \$\phi\_i\$, hence larger \$(\phi\_i+1)^2\$, giving it higher flow. Lower-\$f\$ walkers (nearer current best) have \$\phi\_i \approx 0\$ and thus smaller flow.
   * \$d\_{i,j}^2\$ (distance to a random walker) encourages **diversity**: a walker far from another randomly chosen walker has a larger flow. This term favors exploration by giving isolated or outlying walkers a tendency to move (or be cloned from).
   * \$\delta\_{i,r}^2\$ (distance to a random tabu memory) encourages walkers to escape known local optima: if a walker is far from a known tabu point, \$\delta^2\$ is large (increasing flow); if it is actually at a tabu point, we set \$\delta^2=1\$ (moderate), preventing flow from dropping to zero. In essence, being near a tabu (previously found) local minimum does *not* grant a walker low flow – so it won’t just sit there.

   **Interpretation:** The flow \$F\_i\$ can be seen as a measure of how “exploratory” or “restless” walker \$i\$ should be. High flow occurs for walkers that have poor objective value **and** are distant from others and from known optima, indicating a candidate in a potentially unpromising and isolated region – such a walker is a prime candidate to **jump elsewhere**. Conversely, a walker with low \$F\_i\$ is one that is either near the current best or near other walkers (in a densely explored region), suggesting it might exploit its area more carefully.

5. **Cloning decision:** Now each walker has a chance to **clone** (copy) the state of another walker. For each \$i=1,\dots,N\$, independently do:

   * Randomly choose another walker \$k \neq i\$ uniformly from \${1,\dots,N}\setminus{i}\$. (This \$k\$ can be different from the \$j\$ used in step 1.2; let’s denote this choice \$k = \mathcal{K}(i,t)\$.)
   * Define the **cloning probability** \$P\_{i \leftarrow k}(t)\$ (probability that walker \$i\$ clones walker \$k\$) by:

     $$$P_{i \leftarrow k}(t) ;=; \begin{cases}
       \displaystyle \min\!\Big\{\,1,\;\frac{\,F_i(t) - F_k(t)\,}{\,F_i(t)\,}\Big\} \;, & \text{if } F_k(t) \le F_i(t),\\[2ex]
       0 \;, & \text{if } F_k(t) > F_i(t)~,
     \end{cases}$$
     as long as $F_i(t)>0$. (If $F_i=0$ which is unlikely except edge cases, one could set $P_{i\leftarrow k}=0$ by convention.) This formula:contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17} ensures that only a walker with **higher flow** can potentially clone (copy) the state of a walker with **lower or equal flow**. If walker $i$ has the higher flow of the pair, then $0 < (F_i - F_k)/F_i \le 1$ and so $P_{i\leftarrow k} > 0$. If instead $i$’s flow is lower (meaning $i$ is already in a relatively good/settled state compared to $k$), then $P_{i\leftarrow k}=0$ – a lower-flow walker will **never** copy a higher-flow (worse-off) walker’s state.
     $$$
   * Draw a uniform random number \$\rho \sim U(0,1)\$. If \$\rho < P\_{i \leftarrow k}(t)\$, then **clone**: set \$x\_i(t) := x\_k(t)\$ (and optionally \$F\_i(t) := F\_k(t)\$, though \$F\_i\$ will be recomputed next loop anyway). In words, walker \$i\$ jumps to the position of walker \$k\$. If \$\rho \ge P\_{i \leftarrow k}\$, no cloning occurs and \$x\_i\$ remains unchanged in this step.

Through this cloning mechanism, poorly performing walkers tend to **leap to the locations of better walkers**. The probability is higher when the disparity \$F\_i - F\_k\$ is larger (i.e. \$i\$ is much more exploratory than \$k\$). Cloning promotes **convergence of the swarm** by quickly spreading good solutions: if one walker finds a region of low \$f\$, other high-flow walkers are likely to copy it, flocking towards promising areas. Crucially, because \$d\$ and \$\delta\$ factors influence \$F\$, a walker in a far-flung or novel part of the space may have large \$F\$ and thus readily clone to a more central/better position, preventing exhaustive search of unproductive regions. This is reminiscent of *selection* in evolutionary algorithms, where worse candidates are replaced by better ones, but here it occurs in a decentralized, probabilistic manner.

### Local Search and Tabu Memory Update (Exploitation Step)

**Step 2:** After the exploratory cloning phase, GAS performs local intensification and updates its memory of discovered optima. This phase improves solutions through gradient-based search and inserts new tabu points to avoid revisiting found local minima. The process is:

1. **Center of mass calculation:** Compute a **weighted center-of-mass** of the swarm positions, weighted by their normalized fitness \$\phi\_i\$. Specifically:
   $x_{cm}(t) \;=\; \sum_{i=1}^N \phi_i(t)\; x_i(t)\,. \tag{2.6}$
   Here \$\phi\_i\$ plays the role of a weight (note \$\phi\_i \in \[0,1]\$). Because lower-\$f\$ walkers have smaller \$\phi\_i\$, they contribute **less** to \$x\_{cm}\$, whereas higher-\$f\$ (worse) walkers contribute more weight. In effect, \$x\_{cm}\$ is biased towards the region of the search space where the swarm’s solutions are relatively poor. This might seem counter-intuitive, but the rationale is to obtain a point that is **not simply the current best** (which would correspond to an unweighted centroid or a best-only approach). Instead, \$x\_{cm}\$ lies in a region that might still need improvement. Using this as a starting point for local search encourages exploration of a different basin of attraction than the very best point, potentially leading to discovering a new local minimum.

2. **Identify current best walker:** Determine the index of the best walker in the swarm after cloning, \$i\_{\min} := \arg\min\_i f(x\_i(t))\$, and let \$x\_{\min}(t) = x\_{i\_{\min}}(t)\$ (the position of the current best solution in the swarm). This point has \$\phi\_{i\_{\min}}(t)=0\$ by construction. It represents the **elite solution** of the current iteration.

3. **Local optimization from \$x\_{cm}\$:** Launch a local search (using L-BFGS-B \[10] or any efficient local optimizer) starting at the point \$x\_{cm}(t)\$. Let this local search converge to a local minimum \$t\_{\text{new}} \in D\$. We will add this point to the tabu memory. The reasoning is that \$x\_{cm}\$, being in a region influenced by average/worse solutions, may lead the local search to a *new* local minimum that the swarm has not yet fully exploited. Denote this found local optimum as \$t\_{\text{new,1}} := t\_{\text{new}}\$ for clarity.

4. **Local optimization from \$x\_{\min}\$:** In parallel or next, perform another local search starting at the best swarm position \$x\_{\min}(t)\$ (although this is already a good point, this step can polish it further). Let this yield a local minimum \$t'*{\text{new}}\$. Often \$x*{\min}(t)\$ might already be near a local minimum (possibly the global one), so \$t'*{\text{new}}\$ could be the same point or a very close refined point. Denote this result as \$t*{\text{new,2}} := t'\_{\text{new}}\$.

5. **Update tabu list (first new optimum):** Take the first new local optimum \$t\_{\text{new,1}}\$ from step 2.3 and insert it into the tabu memory list \$T\$. The memory list \$T\$ has size \$N\$ (fixed), so the insertion is done by **overwriting** one existing entry. For example, choose a random index \$r\_0 \in {1,\dots,N}\$ and set \$t\_{r\_0}(t) := t\_{\text{new,1}}\$. This effectively *adds* the newly found local minimum to the set of tabu points (replacing an older one if the list was full).

   Immediately after updating the memory, perform the **Memory Flow & Cloning routine** (described in detail in the next section) **once**. This routine will adjust the tabu memory list by possibly cloning memory entries among themselves, ensuring the memory remains diverse and focused (preventing stagnation of stored points).

6. **Update tabu list (second new optimum):** Similarly, take the second local optimum \$t\_{\text{new,2}}\$ from step 2.4 and overwrite another randomly chosen memory entry, say index \$r\_1\$, setting \$t\_{r\_1}(t) := t\_{\text{new,2}}\$. Then again execute the **Memory Flow & Cloning routine**. By adding both \$t\_{\text{new,1}}\$ and \$t\_{\text{new,2}}\$, we ensure that both a potentially *new basin* (\$x\_{cm}\$'s result) and a refined *elite basin* (\$x\_{\min}\$'s result) are recorded as tabu. Overwriting random entries (as opposed to, say, the oldest) keeps the memory management simple and stochastic.

7. **Best solution update:** After incorporating these new optima, determine the lowest objective value among all tabu memory entries:
   $x^*(t) \;:=\; \arg\min\{\,f(t_r(t)) : 1 \le r \le N\,\}\,, \tag{2.7}$
   and let \$f^*(t) = \min\_r f(t\_r(t))\$. This \$x^*(t)\$ represents the best solution found *up to the current iteration* (considering that memory holds all recent local optima). We then update the global best record: **BEST** \$\leftarrow x^*(t)\$ if \$f^*(t)\$ is better (lower) than the previous \$f^{}\_{\text{BEST}}\$. In other words, {\small BEST} always stores the best function value seen so far in the entire run.

Together, steps 2.1–2.7 emphasize **exploitation** of promising regions: by performing local searches, GAS leverages gradient information to find precise local optima. Inserting those into memory (tabu list) helps steer the swarm away from revisiting them blindly (since being at a tabu point gives no advantage in flow) and instead encourages the swarm to find *other* optima. The memory list can be seen as a repository of “discovered valleys” in the landscape – the algorithm will try to avoid falling into the same valley repeatedly and instead focus on new areas until the global minimum is found.

### Memory Flow & Cloning Routine (Memory Self-Optimization)

After updating the tabu list with a new point (in steps 2.5 and 2.6 above), GAS immediately runs a **memory optimization routine**. This routine is analogous to the walker flow/cloning (Step 1), but it operates on the memory list \$T\$ itself. Its purpose is to prevent the set of stored tabu points from becoming stagnant or too spread-out; it helps concentrate the memory around the best solutions found. Here is the routine (executed on the list \${t\_1,\dots,t\_N}\$):

* **M.1 Compute memory range:** Calculate the current best and worst function values in the memory: \$\tilde f\_{\min}(t) = \min\_r f(t\_r(t))\$ and \$\tilde f\_{\max}(t) = \max\_r f(t\_r(t))\$. For each memory entry \$r\$, define a normalized value
  $\displaystyle \tilde{\phi}_r(t) \;=\; \frac{\,f(t_r(t)) - \tilde f_{\min}(t)\,}{\,\tilde f_{\max}(t) - \tilde f_{\min}(t)\,}\,, \tag{2.8}$
  which lies in $\[0,1]\$. So \$\tilde{\phi}\_r=0\$ for the best memory entry and \$\tilde{\phi}\_r=1\$ for the worst.

* **M.2 Random pair among memories:** Each memory entry \$r\$ selects another distinct memory entry \$s\neq r\$ at random. Compute the squared distance between these two tabu points: \$\tilde{d}\_{r,s}^2 = |,t\_r(t) - t\_s(t),|^2\$.

* **M.3 Memory flow:** Define a “flow” for each memory entry similarly to walkers:
  $\displaystyle \tilde{F}_r(t) \;=\; \Big(\tilde{\phi}_r(t) + 1\Big)^2 \;\cdot\; \tilde{d}_{r,s}^2~. \tag{2.9}$
  (This is analogous to formula for \$F\_i\$, but note there is no tabu-memory-of-memory distance \$\delta\$ term – we only consider pairwise memory distances. The memory points themselves serve as reference for each other.) If \$t\_r\$ is suboptimal (high \$\tilde{\phi}\$) and far from another memory \$t\_s\$, then \$\tilde{F}\_r\$ will be large. If \$t\_r\$ is a very good point (low \$\tilde{\phi}\$) or close to others, \$\tilde{F}\_r\$ will be smaller.

* **M.4 Memory cloning:** For each memory entry \$r\$, pick another random entry \$u \neq r\$. Define the probability that memory \$r\$ **clones** memory \$u\$ as:

  $$$\tilde{P}_{r \leftarrow u}(t) ;=; \begin{cases}
       \min\!\Big\{\,1,\;\frac{\tilde{F}_r(t) - \tilde{F}_u(t)}{\tilde{F}_r(t)}\Big\}, & \text{if }\tilde{F}_u \le \tilde{F}_r,\\
       0, & \text{if }\tilde{F}_u > \tilde{F}_r~,
     \end{cases}$$
  analogous to walker cloning. Then draw $\rho \sim U(0,1)$ and if $\rho < \tilde{P}_{r\leftarrow u}(t)$, set $t_r(t) := t_u(t)$ (memory entry $r$ copies the state of entry $u$). This cloning step causes higher-flow memory points (usually worse local optima or outlier memory entries) to copy lower-flow memory points (better optima) with some probability.
  $$$

The **effect** of Memory Flow & Cloning is to concentrate the memory list on the best solutions found: any inferior memory entry tends to eventually copy the best one, unless new discoveries replace them first. It’s a form of **crowding** or “competition” among stored optima, ensuring that ultimately the tabu list is filled with copies of the globally best optimum found. This helps reinforce the global optimum once it is found (all memory entries may become \$x^\*\$ after many memory cloning updates, making it a strong attractor for walkers via the \$\delta\$ distances). At the same time, memory cloning can also remove duplicate entries in memory (if two entries are equally good, one might clone the other, reducing diversity in memory intentionally to focus on the best).

### Walker Position Update (Random Walk Step)

**Step 3:** After exploration (Step 1) and exploitation (Step 2) are done in an iteration, the final phase is to **move each walker** in the search space via a random step (akin to a mutation or diffusion step). This ensures continuous exploration of the domain and prevents stagnation. The update is as follows:

1. **Jump size determination:** For each walker \$i\$, define a step-size parameter \$\Delta\_i(t)\$ based on its fitness \$\phi\_i(t)\$ (from Step 1.1). The rule is:
   $\displaystyle \Delta_i(t) \;=\; 10^{-\left(5 \;-\; 4\,\phi_i(t)\right)}\,. \tag{2.10}$
   This formula yields \$\Delta\_i \in \[10^{-5},,10^{-1}]\$. If \$\phi\_i=0\$ (walker is best in swarm), \$\Delta\_i = 10^{-5}\$ (a very small step – the best walker moves very cautiously to fine-tune its position). If \$\phi\_i=1\$ (walker is worst), \$\Delta\_i = 10^{-1}\$ (a much larger step – a poor solution takes a big leap). Walkers in between interpolate exponentially between these step sizes. This mechanism implements a **scale-invariant search**: at any given time, some walkers are making tiny adjustments in high-fitness regions while others make large jumps exploring new regions. This multi-scale approach is a hallmark of the algorithm and reflects principles of *fractal search* – exploration occurs on all scales simultaneously, yielding a form of self-similar search behavior.

2. **Random displacement:** For each walker \$i\$, independently generate a random displacement vector \$\xi\_i = (\xi\_{i}^{(1)},\dots,\xi\_{i}^{(d)})\$ where each component \$\xi\_{i}^{(n)}\$ is drawn from a normal distribution \$\mathcal{N}(0,,\Delta\_i(t))\$ with zero mean and variance \$\Delta\_i\$. This yields a Gaussian **step** with standard deviation \$\sqrt{\Delta\_i}\$ along each dimension. Then update the walker’s position:
   $x_i^{(n)}(t) \;:=\; x_i^{(n)}(t) \;+\; L^{(n)}\,\xi_{i}^{(n)}, \qquad \text{for } n=1,\dots,d, \tag{2.11}$
   where \$L^{(n)}\$ is the length (extent) of the domain \$D\$ in the \$n\$-th coordinate. (For example, if \$D = \[a^{(n)}, b^{(n)}]\$ in the \$n\$-th dimension, then \$L^{(n)} = b^{(n)} - a^{(n)}\$.) Multiplying by \$L^{(n)}\$ scales the normalized step \$\xi\$ to the actual size of the domain. Essentially, this adds a Gaussian perturbation to each coordinate of \$x\_i\$, with magnitude relative to domain size and tuned by the walker’s \$\phi\_i\$.

3. **Domain boundary check:** If a walker’s new position after the jump falls outside the domain \$D\$, it is projected back into \$D\$. In practice, the algorithm redraws the random step with a smaller variance until the walker lands inside \$D\$. Specifically, if \$x\_i(t) \notin D\$, we repeat the update with a halved jump magnitude (e.g. replace \$\Delta\_i\$ by \$\Delta\_i/2\$ and redraw \$\xi\$) until the new \$x\_i\$ lies in \$D\$. This ensures walkers do not disappear outside the search space.

4. **Loop continuation:** Increment \$t\$ to \$t+1\$ and return to Step 1 (Flow computation) for the next iteration.

The random walker move in Step 3 provides a persistent random *exploration* pressure. Importantly, its magnitude is **adaptive**: good solutions perform a **fine-grained local search** (small \$\Delta\$) around their neighborhood (exploitation), while bad solutions perform a **coarse global search** (large \$\Delta\$) for better regions (exploration). This adaptive step scheme contributes to GAS’s ability to cover the search space in a multi-resolution manner, a concept closely related to fractal search patterns where both global structure and local detail are explored. Over many iterations, this process allows walkers to wander throughout \$D\$ (especially those not near good solutions), ensuring any region can eventually be reached.

### Halt Criterion

GAS iterates the above steps until a specified **termination condition** is met. Common stopping criteria include:

* Reaching a maximum number of iterations or function evaluations (computational budget).
* Convergence of the best solution: e.g., if the best objective value has not improved beyond a threshold in the last \$K\$ iterations (stability of **BEST**). A suggested criterion is to compare \$\textbf{BEST}(t)\$ with an average of recent best values and stop if the difference falls below a tolerance.
* Achieving a desired objective value (if the global optimum value is known or an acceptable error threshold is given).

When the halt condition triggers, the algorithm outputs the current \$\textbf{BEST}\$ solution \$x^*\$ and its value \$f^* = f(x^\*)\$ as the result of the optimization. In summary, the algorithm will have hopefully navigated the search space and discovered a solution close or equal to the global optimum.

## Theoretical Analysis of GAS

### Multi-Scale Exploration and Exploitation (Fractal Search Insights)

The GAS algorithm exhibits a balance of exploration and exploitation across **multiple scales**, embodying principles of fractal search and self-organization:

* **Scale Invariance:** At any iteration, the population of walkers has heterogeneous step sizes \$\Delta\_i\$ ranging over several orders of magnitude (from \$10^{-5}\$ to \$10^{-1}\$). This means some agents are effectively performing a microscopic search (refining known good optima) while others execute macroscopic leaps (scanning broad unexplored areas). This concurrent multi-scale search is analogous to a fractal (self-similar) process, where structure is present at every scale. It contrasts with algorithms that have a single temperature or step size at a time (like standard simulated annealing, which gradually moves from large to small steps). GAS instead **maintains diversity in scale** automatically: good solutions naturally slow down (like “cooling” locally), and poor solutions continue with large moves. This mechanism significantly improves the algorithm’s ability to avoid local optima and find the global optimum without an externally imposed cooling schedule.

* **Collective Intelligence and Cloning:** The cloning step can be viewed through the lens of **collective intelligence** or evolutionary dynamics. The probability \$P\_{i\leftarrow k} = \max{0, (F\_i - F\_k)/F\_i}\$ resembles a selection rule where walker \$i\$ “imitates” walker \$k\$ if \$k\$’s state appears more promising (lower flow). Notably, if \$F\_k \ll F\_i\$, then \$P\_{i\leftarrow k} \approx 1\$ – a very poor walker almost certainly jumps to a good walker’s position. This is akin to a particle swarm or genetic algorithm effect where bad particles are re-initialized to good locations, except here it’s done gradually and probabilistically each iteration. This **information sharing** greatly accelerates convergence once high-quality solutions appear, while the randomization in choosing \$k\$ and the threshold \$\rho\$ preserve some stochastic exploratory behavior (not all poor walkers deterministically jump to the very best walker, some might copy moderately good ones, maintaining diversity).

* **Tabu Memory and Avoidance of Repetition:** By flagging found local minima as tabu and incorporating them into the walker flow calculation (\$\delta\_{i,r}\$ distances), GAS effectively **repels the swarm from already-found optima** unless they are the absolute best. The memory entries act like charged points that push walkers away. Moreover, the memory cloning ensures that once the global optimum is found and stored, it will tend to dominate the memory list (all \$t\_r \to x^*\$ over time). This creates a **steady attractor** at the global optimum: walkers have \$\delta\_{i,r}^2\$ terms in their flow, so if they stray near \$x^*\$ which is heavily represented in memory, their flow will decrease (since \$\delta\$ to some memory entry at \$x^*\$ becomes small) making them less likely to clone away – effectively they can settle there. Meanwhile, any walker at \$x^*\$ itself has special handling (\$\delta=1\$ set) to not get artificially stuck, but the overall effect is that \$x^\*\$ becomes a focal point of the search once found. This mechanism is borrowed from **Tabu Search** ideas: it diversifies the search by preventing the swarm from spending too long exploiting suboptimal valleys, thereby encouraging exploration of new basins.

In summary, GAS’s design can be seen as implementing a **fractal AI principle**: it simultaneously explores the search space at all relevant scales and dynamically re-distributes computational effort towards the most promising regions (entropy-reducing clustering around optima) while still maintaining random exploration (entropy-increasing diffusion) in other parts. This yields a self-organizing search behavior that is highly effective in complex multimodal landscapes.

### Convergence Properties

GAS is a stochastic algorithm, so convergence can be examined in a probabilistic sense. We consider two aspects: **(a) convergence of the best-found solution** to the global optimum, and **(b) convergence of the distribution of walkers** to an equilibrium distribution (which indicates how walkers spread in the limit).

**(a) Convergence to the optimum:** Under mild conditions, GAS is **globally convergent in probability** to the global optimum. In informal terms, as the number of iterations \$t \to \infty\$, the probability that GAS has found the global minimizer \$x^*\$ (at least one walker at \$x^*\$) approaches 1. A key reason is that the random walk step (Step 3) ensures *irreducible exploration* of the domain: even if the swarm currently is far from \$x^*\$, there is always a non-zero probability that a sequence of random jumps (possibly aided by cloning) will bring some walker arbitrarily close to \$x^*\$. Once a walker enters the basin of attraction of \$x^*\$ (the region from which local search would converge to \$x^*\$), the local search in Step 2 will with high probability find \$x^*\$ and store it in memory. After that point, \$x^*\$ becomes {\small BEST} and remains in memory; the algorithm would only terminate or continue searching elsewhere, but the optimum is essentially found. Formally, one can model the sequence of walker positions as a Markov chain over \$D^N\$. The chain’s transition kernel has a support that, through the random Gaussian moves, eventually covers \$D^N\$ (GAS does not reject moves permanently except those outside \$D\$, which are retried). This suggests the chain is **ergodic** (aperiodic and irreducible) on the state space \$D^N\$. By ergodicity and the inclusion of local search (which ensures any point in the basin leads exactly to the optimum in memory), one can argue that \$x^\*\$ will be discovered almost surely given infinite time. In practice, GAS finds the global optimum in finite time with high probability for the tested benchmark functions, especially when using the **concurrent run strategy** (multiple independent runs and taking the best) which exponentially boosts success probability.

**(b) Convergence of swarm distribution:** We can also inquire about the *steady-state distribution* of walkers as \$t \to \infty\$. Because of the randomness in cloning selection and the random walk, the swarm can be seen as sampling from a certain distribution in \$D\$ when it has converged. Intuitively, one expects that in the long run, walkers will spend more time in regions of low \$f(x)\$ (near global or good local minima) than in regions of high \$f(x)\$. We can make this more precise by considering a **reward function** \$R(x)\$ that maps objective values to a positive reward measure. For instance, since we minimize \$f\$, a reasonable choice is \$R(x) = M - f(x)\$ for some constant \$M > \max\_{y\in D} f(y)\$, so that \$R(x)\$ is larger for better (lower \$f\$) solutions. Then \$R(x^\*)\$ is maximal. We can conjecture that GAS induces a stationary distribution \$\pi(x)\$ over the domain proportional to \$R(x)\$ (or some function thereof). In other words, we hypothesize:

*In the limit of a large number of walkers \$N \to \infty\$ and long time \$t \to \infty\$, the empirical distribution of walkers \$\rho\_W(x,t)\$ converges to a distribution \$\rho\_\infty(x)\$ that is **proportional to the reward**:*

$\rho_\infty(x) \;\propto\; R(x). \tag{2.12}$

Equivalently, \$\rho\_\infty(x) = \frac{R(x)}{Z}\$ on \$D\$, where \$Z = \int\_D R(u),du\$ is a normalizing constant. In our specific choice, \$\rho\_\infty(x) = \frac{M - f(x)}{\int\_D (M - f(u)),du}\$. This means the **density of walkers at any location is higher for lower \$f(x)\$**, achieving its maximum at the global minimum \$x^*\$ (since \$R(x^*)\$ is largest). This property aligns with the goal of optimization: at convergence, most walkers concentrate near optimal or near-optimal solutions. A similar result has been **proven in the context of Fractal AI theory** for a related algorithm (Fractal Monte Carlo) in a dynamic setting, using mean-field theory and stochastic process convergence: the swarm density was shown to converge to the reward distribution. For GAS, the presence of cloning and the flow mechanism biases the stationary distribution towards good solutions in a manner akin to a **Gibbs distribution** in simulated annealing (though here no temperature parameter is explicitly present, the distribution is effectively fixed by the algorithm’s dynamics).

*Sketch of justification:* Each cloning event can be seen as a step that *increases* the concentration of walkers in regions of higher reward (lower \$f\$). Meanwhile, the random walk step ensures a diffusive pressure that spreads walkers out. One can define a potential function or Lyapunov function for the swarm’s distribution, for example the **Kullback-Leibler divergence** between the current swarm distribution \$\rho\_W(x,t)\$ and the target distribution \$\rho\_R(x) = R(x)/Z\$. Using techniques from stochastic approximation or mean-field analysis, one can argue that the expected cloning effect **decreases** this divergence over time (i.e., the swarm distribution moves toward the reward-weighted distribution). In fact, one can derive that the cloning mechanism implements a form of **entropy minimization**: it causes an **entropy flow from low-reward regions to high-reward regions**, thereby redistributing probability mass in proportion to \$R(x)\$. A simplified view is to consider an infinite population (\$N\to\infty\$) so that we deal with densities instead of discrete walkers. The cloning rule \$P\_{i\leftarrow k} = \max(0,1 - F\_k/F\_i)\$ effectively means regions with lower \$F\$ (which correlates with higher \$R\$ and denser population) attract mass from regions with higher \$F\$ (lower density/reward). In the mean-field limit, one can show the density \$\rho\_W(x,t)\$ evolves according to a nonlinear partial differential equation that has equilibrium solution \$\rho\_W(x) \propto R(x)\$ (this is analogous to a replicator equation driving the distribution towards the “fitness” function \$R\$). A rigorous convergence proof would involve showing that \$\rho\_R(x)\$ is a stationary solution of the derived Fokker-Planck equation of the process and that the solution is globally asymptotically stable in the space of distributions (perhaps by showing the KL divergence \$D(\rho\_R \parallel \rho\_W(t))\$ decreases to 0). While a full formal proof is beyond the scope of this report, these arguments strongly suggest that GAS will asymptotically concentrate the walkers around the global optimum in proportion to how optimal those points are.

**Implications:** In practice, the above means that **the best solution found by GAS tends to improve (or at least not deteriorate) over time**, and ultimately the global optimum will dominate the swarm. Even if not all walkers collapse to exactly \$x^*\$, a large fraction will hover in its vicinity. This also implies that if one runs GAS indefinitely, the fraction of time the best solution is \$x^*\$ approaches 1. Many metaheuristics share a similar property of *convergence in probability* to global optima given infinite time (assuming full mixing of the search space), but GAS achieves this without requiring a cooling schedule or hyperparameter tuning of probabilities – it is an emergent property of the flow and cloning dynamics.

One practical convergence indicator is that the tabu memory list will start filling up with identical entries \$x^\*\$ once the global optimum is found (memory cloning drives this). At that point, new local searches will not find anything better, and the **BEST** solution stabilizes. The algorithm can then be halted confidently.

### Computational Complexity

The time complexity of GAS per iteration is \$O(N)\$ plus the cost of local searches. Specifically, each iteration each walker performs a constant amount of work to compute flows and clone (pairwise operations and random draws), which is \$O(N)\$. The memory routine is also \$O(N)\$. If \$N\$ walkers each take a step and two local searches are done, the cost of local search depends on function complexity and is usually minor if using a few gradient iterations. Over \$T\$ iterations, the complexity is \$O(N \cdot T)\$ (linear in the total number of walker updates). In many applications, the dominant cost is actually the number of function evaluations. GAS tries to be efficient in function evaluations by doing cloning (which reuses evaluations from copied walkers) and focusing local search only on a couple of points per iteration. The **concurrent optimization** strategy – running multiple instances of GAS independently and taking the best result – is embarrassingly parallel and linear in the number of runs. The authors note that GAS scales well with multi-threading or multi-run setups, showing improved solution quality when, for example, 20 or 50 independent runs are done concurrently (the first run to hit the optimum can terminate the others). In summary, GAS is computationally on par with other swarm methods, and its convergence speed (in terms of iterations or evaluations to reach optima) has been observed to be competitive or superior on challenging test functions.

## Conclusion

We have provided a detailed mathematical formalization of the General Algorithmic Search algorithm, articulating its step-by-step operations and internal mechanisms. GAS can be seen as a **physics-inspired swarm optimization method**, where the concept of “flow” drives a particle-like system toward low-energy (low-\$f\$) states, and cloning acts as a teleportation or reproduction mechanism for favorable states. By blending global random walks, dynamic step sizing, local gradient searches, and a tabu memory, GAS achieves a powerful balance between exploration and exploitation.

From a theoretical standpoint, GAS leverages **entropic principles**: it effectively redistributes the “search probability” in proportion to solution quality (reward), which is a desirable property ensuring focus on good solutions without getting trapped. The algorithm’s inherent stochasticity and multi-scale search fulfill key conditions for global convergence in probability, and connections to Fractal AI theory suggest that as the number of agents and iterations grow, the swarm’s distribution aligns with the optimal reward landscape, thereby increasingly concentrating around the global optimum. Consequently, the best solution found monotonically approaches the true optimum (and in practice often finds it exactly).

Empirical results in the original work support these claims: GAS not only found global minima reliably for a suite of 31 multimodal test functions, but it did so faster (in terms of function evaluations) than classical algorithms like BH, CS, and DE in most cases. In particular, the advantage of GAS became more pronounced when multiple runs were allowed, indicating its robustness in **concurrent optimization** scenarios where a quick first hit on the optimum is rewarded. The integration of ideas from Tabu Search, evolutionary cloning, and adaptive random walks makes GAS a noteworthy advancement in global optimization heuristics.

In conclusion, the GAS algorithm offers a formally grounded and empirically validated approach to difficult optimization problems. Its mathematical structure – combining dynamic normalization (\$\phi\_i\$), flow metrics \$F\_i\$, cloning probabilities, and memory adaptation – provides new insights into designing search algorithms that **self-tune** and **self-balance** the exploration-exploitation trade-off. This formal understanding not only helps in analyzing GAS itself but also paves the way for developing future algorithms under the same principles, potentially extending to broader domains (e.g. sequential decision making as in the Fractal AI planning frameworks). GAS exemplifies how a synergy of techniques can yield an algorithm that is greater than the sum of its parts, achieving efficient global search in complex landscapes with high probability of success.

**References:**

\[1] D. J. Wales, J. P. K. Doye. *Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters containing up to 110 atoms*. J. Phys. Chem. A **101**, 5111–5116 (1997).&#x20;

\[2] X. S. Yang. *Firefly algorithms for multimodal optimization*. In **Stochastic Algorithms: Foundations and Applications (SAGA)**, LNCS 5792, pp. 169–178. Springer (2009).&#x20;

\[3] R. Storn, K. Price. *Differential Evolution – a simple and efficient heuristic for global optimization over continuous spaces*. J. Global Optimization **11**, 341–359 (1997).&#x20;

\[4] S. Voss. *Meta-heuristics: the state of the art*. In **Local Search for Planning and Scheduling**, LNCS 2148, pp. 1–23. Springer (2001).

\[5] S. Hernández, G. Durán, J. M. Amigó. *General Algorithmic Search*. arXiv:1705.08691 \[math.OC] (2017).&#x20;

\[6] F. Glover. *Tabu Search – Part 1*. ORSA Journal on Computing **1**, 190–206 (1989).&#x20;

\[7] F. Glover. *Tabu Search – Part 2*. ORSA Journal on Computing **2**, 4–32 (1990).

\[8] R. H. Byrd, P. Lu, J. Nocedal. *A limited memory algorithm for bound constrained optimization*. SIAM J. Scientific and Statistical Computing **16**, 1190–1208 (1995).&#x20;

\[9] X. S. Yang, S. Deb. *Engineering optimization by cuckoo search*. Int. J. Mathematical Modelling and Numerical Optimisation **1**(4), 330–343 (2010).

\[10] J. Kennedy, R. C. Eberhart. *Particle swarm optimization*. In **Proc. IEEE Int. Conf. Neural Networks**, pp. 1942–1948 (1995).
