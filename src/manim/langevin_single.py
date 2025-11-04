import matplotlib
import numpy as np

from manim import *


matplotlib.use("Agg")  # Use non-interactive backend for headless environment


class LangevinSingle(Scene):
    def construct(self):
        # Parameters
        dt = 0.05
        gamma = 0.5  # friction
        kT = 0.2  # temperature
        total_time = 20  # seconds

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
        title = Text("Underdamped Langevin Dynamics", font_size=36, weight=BOLD)
        title.to_edge(UP).shift(DOWN * 0.3)
        self.add(title)

        # Add equations in a nice box
        equations = VGroup(
            MathTex(
                r"\frac{dv}{dt} = -\nabla V(x) - \gamma v + \sqrt{2\gamma k_B T} \, \xi(t)",
                font_size=28,
            ),
            MathTex(r"\frac{dx}{dt} = v", font_size=28),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        # Add parameter values
        params = VGroup(
            MathTex(r"\gamma = " + f"{gamma:.1f}", font_size=24, color=YELLOW),
            MathTex(r"k_B T = " + f"{kT:.1f}", font_size=24, color=YELLOW),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        # Create info box
        info_group = VGroup(equations, params).arrange(DOWN, buff=0.5, aligned_edge=LEFT)

        # Add background rectangle
        info_bg = BackgroundRectangle(
            info_group, color=BLACK, fill_opacity=0.8, buff=0.3, corner_radius=0.1
        )

        info_box = VGroup(info_bg, info_group)
        info_box.to_corner(UL).shift(DOWN * 1.0 + RIGHT * 0.2)
        self.add(info_box)

        # Add legend for visualization elements
        legend_items = VGroup(
            VGroup(Dot(color=YELLOW, radius=0.1), Text("Particle", font_size=20)).arrange(
                RIGHT, buff=0.2
            ),
            VGroup(
                Arrow(ORIGIN, RIGHT * 0.3, color=RED, stroke_width=3),
                Text("Velocity", font_size=20),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Line(ORIGIN, RIGHT * 0.3, color=BLUE, stroke_width=3),
                Text("Trajectory", font_size=20),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        legend_bg = BackgroundRectangle(
            legend_items, color=BLACK, fill_opacity=0.8, buff=0.2, corner_radius=0.1
        )

        legend = VGroup(legend_bg, legend_items)
        legend.to_corner(UR).shift(DOWN * 1.0 + LEFT * 0.2)
        self.add(legend)

        # Initialize particle
        pos = np.array([2.0, -2.0])  # Start in upper right
        vel = np.array([0.0, 0.0])

        # Create particle dot
        particle = Dot(point=ORIGIN, color=YELLOW, radius=0.15)
        particle.move_to([pos[0], pos[1], 0])
        particle.set_z_index(2)  # Ensure particle is on top

        # Create velocity vector with prominent tip
        vel_arrow = Arrow(
            start=ORIGIN,
            end=RIGHT * 0.1,  # Initial small arrow
            color=RED,
            buff=0,
            stroke_width=4,
            tip_length=0.25,
            max_tip_length_to_length_ratio=0.5,
            max_stroke_width_to_length_ratio=10,
        )
        vel_arrow.set_z_index(1)

        # Create trail with fade
        trail = TracedPath(
            particle.get_center,
            stroke_color=BLUE,
            stroke_width=3,
            stroke_opacity=0.8,
            dissipating_time=5,  # Trail fades over 5 seconds
        )
        trail.set_z_index(0)

        self.add(trail, vel_arrow, particle)

        def update_particle(mob, dt_frame=1 / 60):
            nonlocal pos, vel

            # Use simulation timestep, not frame timestep
            dt_sim = dt

            # Underdamped Langevin dynamics (Euler-Maruyama)
            # dv = -∇V(x) dt - γv dt + √(2γkT) dW
            # dx = v dt

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

            # Update particle position
            mob.move_to([pos[0], pos[1], 0])

            # Update velocity arrow with better scaling
            vel_scale = 0.7  # Increased scale for visibility
            vel_norm = np.linalg.norm(vel)

            if vel_norm > 0.05:  # Only show arrow if velocity is significant
                # Calculate arrow end point
                arrow_end = np.array([pos[0] + vel[0] * vel_scale, pos[1] + vel[1] * vel_scale, 0])

                # Update arrow
                new_arrow = Arrow(
                    start=[pos[0], pos[1], 0],
                    end=arrow_end,
                    color=RED,
                    buff=0,
                    stroke_width=4,
                    tip_length=0.25,
                    max_tip_length_to_length_ratio=0.5,
                    max_stroke_width_to_length_ratio=10,
                )
                vel_arrow.become(new_arrow)
                vel_arrow.set_opacity(1)
            else:
                vel_arrow.set_opacity(0)

        # Run animation
        particle.add_updater(update_particle)
        self.wait(total_time)
        particle.remove_updater(update_particle)

    def create_heatmap(self, data):
        """Create a heatmap image from normalized data"""
        # Use a colormap (blue to red)
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
        plt.savefig("/tmp/heatmap.png", dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        return "/tmp/heatmap.png"
