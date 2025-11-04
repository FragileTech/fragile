import matplotlib
import numpy as np

from manim import *


matplotlib.use("Agg")  # Use non-interactive backend for headless environment


class LangevinMulti(Scene):
    def construct(self):
        # Parameters
        N = 15  # Number of particles
        dt = 0.05
        gamma = 0.5  # friction
        kT = 0.2  # temperature
        total_time = 25  # seconds

        # Create fitness landscape (2D Gaussian mixture)
        def fitness(x, y):
            """V(x,y) - lower is better (particle seeks minima)"""
            return (
                -2.0 * np.exp(-((x - 1) ** 2 + (y - 1) ** 2) / 0.5)
                + -1.5 * np.exp(-((x + 1.5) ** 2 + (y + 1.5) ** 2) / 0.8)
                + -1.0 * np.exp(-((x - 1.5) ** 2 + (y + 2) ** 2) / 0.6)
                + 0.3 * (x**2 + y**2)  # slight quadratic to keep bounded
            )

        def gradient_V(x, y):
            """∇V(x,y)"""
            eps = 1e-4
            dx = (fitness(x + eps, y) - fitness(x - eps, y)) / (2 * eps)
            dy = (fitness(x, y + eps) - fitness(x, y - eps)) / (2 * eps)
            return np.array([dx, dy])

        # Create landscape visualization (heatmap)
        resolution = 100
        x_range = np.linspace(-3, 3, resolution)
        y_range = np.linspace(-3, 3, resolution)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = fitness(X[i, j], Y[i, j])

        # Normalize for color mapping
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min())

        # Create heatmap
        landscape = ImageMobject(self.create_heatmap(Z_norm))
        landscape.scale(3)
        landscape.set_opacity(0.7)
        self.add(landscape)

        # Add title
        title = Text(f"Underdamped Langevin Dynamics: {N} Particles", font_size=36, weight=BOLD)
        title.to_edge(UP).shift(DOWN * 0.3)
        self.add(title)

        # Add equations in a nice box
        equations = VGroup(
            MathTex(
                r"\frac{dv_i}{dt} = -\nabla V(x_i) - \gamma v_i + \sqrt{2\gamma k_B T} \, \xi_i(t)",
                font_size=26,
            ),
            MathTex(r"\frac{dx_i}{dt} = v_i", font_size=26),
            MathTex(r"i = 1, \ldots, " + str(N), font_size=24, color=BLUE_C),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)

        # Add parameter values
        params = VGroup(
            MathTex(r"\gamma = " + f"{gamma:.1f}", font_size=24, color=YELLOW),
            MathTex(r"k_B T = " + f"{kT:.1f}", font_size=24, color=YELLOW),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        # Create info box
        info_group = VGroup(equations, params).arrange(DOWN, buff=0.4, aligned_edge=LEFT)

        # Add background rectangle
        info_bg = BackgroundRectangle(
            info_group, color=BLACK, fill_opacity=0.85, buff=0.3, corner_radius=0.1
        )

        info_box = VGroup(info_bg, info_group)
        info_box.to_corner(UL).shift(DOWN * 1.0 + RIGHT * 0.2)
        self.add(info_box)

        # Add legend for visualization elements
        legend_items = VGroup(
            VGroup(Dot(color=YELLOW, radius=0.08), Text("Particles", font_size=18)).arrange(
                RIGHT, buff=0.2
            ),
            VGroup(
                Line(ORIGIN, RIGHT * 0.3, color=BLUE_C, stroke_width=2),
                Text("Trajectories", font_size=18),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        legend_bg = BackgroundRectangle(
            legend_items, color=BLACK, fill_opacity=0.85, buff=0.2, corner_radius=0.1
        )

        legend = VGroup(legend_bg, legend_items)
        legend.to_corner(UR).shift(DOWN * 1.0 + LEFT * 0.2)
        self.add(legend)

        # Initialize particles with random positions
        np.random.seed(42)  # For reproducibility
        positions = []
        velocities = []
        particles = []
        trails = []

        # Color palette for particles
        colors = [
            YELLOW,
            RED,
            GREEN,
            BLUE,
            ORANGE,
            PURPLE,
            PINK,
            TEAL,
            LIGHT_BROWN,
            MAROON,
            GOLD,
            LIGHT_PINK,
            GREY,
            WHITE,
        ]

        for i in range(N):
            # Random initial positions
            pos = np.array([np.random.uniform(-2.5, 2.5), np.random.uniform(-2.5, 2.5)])
            vel = np.array([0.0, 0.0])

            positions.append(pos)
            velocities.append(vel)

            # Create particle dot
            color = colors[i % len(colors)]
            particle = Dot(point=[pos[0], pos[1], 0], color=color, radius=0.12)
            particle.set_z_index(2)
            particles.append(particle)

            # Create trail with subtle color variation
            trail = TracedPath(
                particle.get_center,
                stroke_color=color,
                stroke_width=2,
                stroke_opacity=0.6,
                dissipating_time=4,
            )
            trail.set_z_index(0)
            trails.append(trail)
            self.add(trail)

        # Add all particles
        for particle in particles:
            self.add(particle)

        # Create updater closure
        def create_updater(idx):
            def update_particle(mob, dt_frame=1 / 60):
                nonlocal positions, velocities

                # Use simulation timestep
                dt_sim = dt

                # Get current position and velocity
                pos = positions[idx]
                vel = velocities[idx]

                # Underdamped Langevin dynamics (Euler-Maruyama)
                grad = gradient_V(pos[0], pos[1])
                noise = np.sqrt(2 * gamma * kT / dt_sim) * np.random.randn(2)

                # Update velocity
                vel = vel - grad * dt_sim - gamma * vel * dt_sim + noise * np.sqrt(dt_sim)

                # Update position
                pos += vel * dt_sim

                # Boundary reflection (soft)
                if abs(pos[0]) > 2.8:
                    pos[0] = np.clip(pos[0], -2.8, 2.8)
                    vel[0] *= -0.5
                if abs(pos[1]) > 2.8:
                    pos[1] = np.clip(pos[1], -2.8, 2.8)
                    vel[1] *= -0.5

                # Store updated values
                positions[idx] = pos
                velocities[idx] = vel

                # Update particle position
                mob.move_to([pos[0], pos[1], 0])

            return update_particle

        # Add updaters to all particles
        for i, particle in enumerate(particles):
            particle.add_updater(create_updater(i))

        # Run animation
        self.wait(total_time)

        # Remove updaters
        for particle in particles:
            particle.remove_updater(particle.updaters[0])

    def create_heatmap(self, data):
        """Create a heatmap image from normalized data"""
        import matplotlib.pyplot as plt

        # Create figure without axes
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        # Plot heatmap
        ax.imshow(data, cmap="RdYlBu_r", origin="lower", extent=[-3, 3, -3, 3])
        ax.axis("off")

        # Save to temporary file
        plt.tight_layout(pad=0)
        plt.savefig("/tmp/heatmap_multi.png", dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        return "/tmp/heatmap_multi.png"
