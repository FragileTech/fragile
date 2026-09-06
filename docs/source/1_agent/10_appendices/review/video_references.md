youtube videos for reference:

- https://www.youtube.com/watch?v=XD9Fumzf57Y
- https://www.youtube.com/watch?v=HLbThk624jI
- https://www.youtube.com/watch?v=DsvSH3cNhnE
- https://www.youtube.com/watch?v=OFhBKZ0l6fw&t=11s
- https://www.youtube.com/watch?v=cyibNzyU4ug


original fractalAI paper:
- https://arxiv.org/abs/1803.05049


A complete breakdown of technical and functional engine requirements has been compiled directly from reviewing the five simulation videos:

* [Kart simulation 19: Ants and drops](http://www.youtube.com/watch?v=XD9Fumzf57Y)
* [Asteroid harvesting](http://www.youtube.com/watch?v=HLbThk624jI)
* [Tamdem AI](http://www.youtube.com/watch?v=DsvSH3cNhnE)
* [Collaborative Mining](http://www.youtube.com/watch?v=OFhBKZ0l6fw)
* [Mining Rocket thinking graphs 2](http://www.youtube.com/watch?v=cyibNzyU4ug)

---

## 1. Multi-Agent Coordination & Control Architecture

* **Joint High-Dimensional Action Space:**
* Support treating multiple distinct bodies as a single FMC meta-agent with joint state–action spaces (e.g., $N$ spaceships each with 2 continuous degrees of freedom control simultaneously an action vector of size $2N$).
* Seen in [Tamdem AI](http://www.youtube.com/watch?v=DsvSH3cNhnE) and [Collaborative Mining](http://www.youtube.com/watch?v=OFhBKZ0l6fw), where two ships (yellow and red) operate with coordinated relative geometry.


* **Massive Swarm/Crowd Capacity:**
* Capable of running dozens to hundreds of low-complexity agent bodies simultaneously on a single track ([Kart simulation 19](http://www.youtube.com/watch?v=XD9Fumzf57Y) showcases over 40–50 karts dynamically avoiding collisions and chasing targets).


* **Virtual Anchor / Formation Barycenter:**
* In tandem flight mode, support a virtual target anchor or triangle formation barycenter ([Tamdem AI](http://www.youtube.com/watch?v=DsvSH3cNhnE) shows a dotted guide triangle between both karts tracking ahead of them).



---

## 2. 2D Continuous Physics & Dynamics

* **Inertial Spacecraft Dynamics:**
* Continuous 2D position, linear momentum, angular orientation, and drag/friction.
* Directional thrusters indicated by exhaust vector lines shooting from the rear of the ship ([Asteroid harvesting](http://www.youtube.com/watch?v=HLbThk624jI)).


* **Gravitational Attractors:**
* Central or local gravity wells represented by circular hatched zones ([Asteroid harvesting](http://www.youtube.com/watch?v=HLbThk624jI)) that pull karts/ships and free-floating asteroids toward their center.


* **Distance Constraints & Elastic Tethers:**
* Spring/elastic hook mechanics connecting a ship to a polygon asteroid or connecting two ships to each other ([Collaborative Mining](http://www.youtube.com/watch?v=OFhBKZ0l6fw)).
* Visualized as dashed tether lines with realistic tension pulling and momentum transfer.


* **Map Geometry & Collision Masks:**
* Arbitrary non-convex 2D cave/track boundary meshes with interior holes/pillars.
* High-speed raycasting/distance-field checks against static cave walls to calculate lethal collision risks.



---

## 3. Game Mechanics & Entity Ecosystem

* **Asteroid Mining & Delivery:**
* Rigid-body polygon asteroids of varying masses and sizes.
* Receptacles / Base zones: Circular drop-off zones that flash green when an asteroid or cargo is successfully delivered ([Asteroid harvesting](http://www.youtube.com/watch?v=HLbThk624jI), [Collaborative Mining](http://www.youtube.com/watch?v=OFhBKZ0l6fw)).


* **Resource / Food Foraging:**
* Dynamic spawning of pickup items (food pellets, circular drops) that agents consume on touch ([Kart simulation 19](http://www.youtube.com/watch?v=XD9Fumzf57Y)).


* **Racing Waypoints & Gates:**
* Virtual checkpoint loops/rings that agents must navigate through in sequence ([Tamdem AI](http://www.youtube.com/watch?v=DsvSH3cNhnE)).



---

## 4. Fractal AI / FMC Simulation & Debug Visualizer

* **Tree & Path Rollout Visualization:**
* Render real-time Monte Carlo / Fractal search trees branching forward from the agent's current pose ([Mining Rocket thinking graphs 2](http://www.youtube.com/watch?v=cyibNzyU4ug) and [Collaborative Mining](http://www.youtube.com/watch?v=OFhBKZ0l6fw)).


* **Semantic Path Color-Coding:**
* **Green paths:** Safe, high-reward, goal-oriented trajectory clusters.
* **Red/Dark paths:** Collision trajectories, boundary strikes, or catastrophic dead ends.
* **Blue/Purple paths:** Tether dynamics, asteroid interaction rollouts, or alternate branches.


* **Particle / Uncertainty Clouds:**
* Particle distribution rendering to show stochastic uncertainty across future states (visualizing how state variance expands over rollout time horizons).


* **Toggleable Render Layers:**
* Ability to switch between a clean game-view and deep diagnostic overlays (showing thousands of simulated rollout paths per frame without stalling the physics tick).



---

## 5. Telemetry & AI Diagnostic HUD

As seen consistently in the top-left corner of the mining simulations ([Asteroid harvesting](http://www.youtube.com/watch?v=HLbThk624jI), [Collaborative Mining](http://www.youtube.com/watch?v=OFhBKZ0l6fw), [Mining Rocket thinking graphs 2](http://www.youtube.com/watch?v=cyibNzyU4ug)), the engine requires a real-time metrics overlay:

| Metric | Description / Visualized Purpose |
| --- | --- |
| **`Toy number`** | Active scenario / task identifier |
| **`Dead ratio`** | Proportion of sampled rollout paths that lead to crashes/death (e.g., `18% (0/3)`) |
| **`Risk level`** | Estimated catastrophic risk of the chosen action distribution |
| **`Evaporated / Real Evapore`** | Fractal AI tree pruning / particle evaporation rate |
| **`AI used`** | Percentage of computational budget or agent capacity utilized |
