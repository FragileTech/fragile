import numpy as np

from manim import *


class CloningMechanismScene(MovingCameraScene):
    """3Blue1Brown-style explanation of the Fragile cloning mechanism."""

    def construct(self):
        self.setup_canvas()
        self.show_title()
        self.storyboard_stage()
        self.error_signal_stage()
        self.pressure_curve_stage()
        self.coin_flip_stage()
        self.position_diffusion_stage()
        self.momentum_blend_stage()
        self.summary_stage()
        self.final_frame()

    def limit_width(self, mob: Mobject, max_width: float) -> Mobject:
        """Clamp an object's width so it stays inside the frame."""
        if mob.width > max_width:
            mob.scale_to_fit_width(max_width)
        return mob

    # ------------------------------------------------------------------
    # Scene sections
    # ------------------------------------------------------------------
    def setup_canvas(self):
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=["#0d1b2a", "#1b263b"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        overlay = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=BLACK,
            fill_opacity=0.45,
            stroke_width=0,
        )
        plane = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-3.5, 3.5, 1],
            faded_line_ratio=2,
            background_line_style={"stroke_color": GREY_B, "stroke_opacity": 0.15},
        )
        self.add(background, overlay, plane)

    def show_title(self):
        title = Text(
            "Cloning as an Error Corrector",
            gradient=(YELLOW_B, WHITE),
            font_size=62,
        )
        self.limit_width(title, config.frame_width - 1.5)
        subtitle = Text(
            "Walkers borrow state from stronger companions to stay balanced",
            font_size=32,
            color=GREY_B,
        )
        self.limit_width(subtitle, config.frame_width - 2.0)
        subtitle.next_to(title, DOWN, buff=0.35)
        underline = Line(
            title.get_left() + LEFT * 0.2,
            title.get_right() + RIGHT * 0.2,
            color=YELLOW_B,
            stroke_width=6,
        ).next_to(title, DOWN, buff=0.12)

        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle, shift=UP * 0.25), Create(underline))
        self.wait(1.2)
        self.play(
            FadeOut(title, shift=UP * 0.4),
            FadeOut(subtitle, shift=UP * 0.4),
            FadeOut(underline),
        )

    def storyboard_stage(self):
        header = Text("Four ingredients of cloning", font_size=44, color=TEAL_B)
        header.to_edge(UP, buff=0.6)

        captions = [
            ("Sense imbalance", "Compare a walker to its chosen companion."),
            ("Pressurize laggards", "Convert fitness gaps into a cloning urge."),
            ("Flip a biased coin", "Noise decides which walkers actually jump."),
            ("Blend states", "Cloners adopt the companion's position and flow."),
        ]

        cards = VGroup()
        for title, line in captions:
            box = RoundedRectangle(
                width=4.6,
                height=2.0,
                corner_radius=0.3,
                stroke_color=TEAL_E,
                fill_color="#18283a",
                fill_opacity=0.5,
            )
            title_text = Text(title, font_size=30, color=YELLOW_B, weight=BOLD)
            self.limit_width(title_text, box.width - 0.6)
            body_text = Text(line, font_size=24, color=WHITE)
            self.limit_width(body_text, box.width - 0.8)
            text_group = VGroup(title_text, body_text).arrange(DOWN, buff=0.22)
            text_group.move_to(box.get_center())
            cards.add(VGroup(box, text_group))

        cards.arrange_in_grid(rows=2, cols=2, buff=0.5)
        cards.scale_to_fit_width(config.frame_width - 1.2)
        cards.move_to(DOWN * 0.2)

        self.play(FadeIn(header, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(card, shift=DOWN * 0.2) for card in cards], lag_ratio=0.2))
        self.wait(1.0)
        self.play(FadeOut(header), FadeOut(cards))

    def error_signal_stage(self):
        title = Text("Step 1 · Sensing imbalance", font_size=40, color=YELLOW_B)
        title.to_edge(UP, buff=0.6)

        formula = MathTex(
            r"S_i = \frac{V_{c_i} - V_i}{V_i + \varepsilon}",
            font_size=48,
            color=WHITE,
        )
        self.limit_width(formula, config.frame_width - 3.0)
        formula.next_to(title, DOWN, buff=0.4)

        explanation_lines = [
            Text(
                "Positive S → companion is better → cloning pressure", font_size=28, color=TEAL_B
            ),
            Text("Negative S → walker is stronger → stay put", font_size=28, color=GREY_B),
        ]
        for line in explanation_lines:
            self.limit_width(line, config.frame_width - 3.5)
        explanation = VGroup(*explanation_lines).arrange(DOWN, buff=0.3)
        explanation.next_to(formula, DOWN, buff=0.4)

        number_line = NumberLine(
            x_range=[-1.5, 2.5, 0.5],
            length=7.0,
            color=GREY_B,
            include_numbers=True,
            font_size=28,
        )
        number_line.shift(DOWN * 1.2)
        origin = number_line.number_to_point(0)

        walker_bar = Rectangle(
            width=0.15,
            height=1.5,
            fill_color=TEAL_B,
            fill_opacity=0.8,
            stroke_width=0,
        ).next_to(origin, LEFT, buff=0)
        companion_bar = Rectangle(
            width=0.15,
            height=2.1,
            fill_color=YELLOW_B,
            fill_opacity=0.8,
            stroke_width=0,
        ).next_to(origin + RIGHT * 1.8, LEFT, buff=0)

        walker_label = Text("walker", font_size=24, color=TEAL_B).next_to(
            walker_bar, DOWN, buff=0.2
        )
        companion_label = Text("companion", font_size=24, color=YELLOW_B).next_to(
            companion_bar, DOWN, buff=0.2
        )

        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.play(Write(formula))
        self.play(
            LaggedStart(*[FadeIn(line, shift=DOWN * 0.2) for line in explanation], lag_ratio=0.2)
        )
        self.play(Create(number_line))
        self.play(
            FadeIn(walker_bar),
            FadeIn(companion_bar),
            FadeIn(walker_label),
            FadeIn(companion_label),
        )
        self.wait(1.0)
        self.play(
            FadeOut(title),
            FadeOut(formula),
            FadeOut(explanation),
            FadeOut(number_line),
            FadeOut(walker_bar),
            FadeOut(companion_bar),
            FadeOut(walker_label),
            FadeOut(companion_label),
        )

    def pressure_curve_stage(self):
        title = Text("Step 2 · Turning score into pressure", font_size=40, color=YELLOW_B)
        title.to_edge(UP, buff=0.6)

        curve_label = MathTex(
            r"\pi(S) = \text{clip}\!\left(\frac{S}{p_{\max}}\right)",
            font_size=42,
            color=WHITE,
        )
        self.limit_width(curve_label, config.frame_width - 3.0)
        curve_label.next_to(title, DOWN, buff=0.4)

        axes = Axes(
            x_range=[-0.5, 2.0, 0.5],
            y_range=[0, 1.1, 0.2],
            x_length=6.5,
            y_length=3.5,
            axis_config={"include_tip": False, "font_size": 26, "color": GREY_B},
        )
        axes.move_to(DOWN * 0.5)
        graph = axes.plot(
            lambda s: np.clip(s / 0.75, 0.0, 1.0),
            x_range=[-0.5, 2.0],
            color=TEAL_B,
            stroke_width=6,
        )

        no_pressure = Text("no pressure", font_size=26, color=RED_B)
        self.limit_width(no_pressure, 3.4)
        no_pressure.next_to(axes.c2p(-0.2, 0), DOWN, buff=0.35).shift(LEFT * 0.2)

        rising_urgency = Text("linearly rising urgency", font_size=26, color=YELLOW_B)
        self.limit_width(rising_urgency, 3.8)
        rising_urgency.next_to(axes.c2p(0.4, 0.5), UR, buff=0.35)

        full_replacement = Text("full replacement", font_size=26, color=GREEN_B)
        self.limit_width(full_replacement, 3.6)
        full_replacement.next_to(axes.c2p(1.4, 1.0), UR, buff=0.4)

        annotations = VGroup(no_pressure, rising_urgency, full_replacement)

        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.play(Write(curve_label))
        self.play(Create(axes), Create(graph))
        self.play(
            LaggedStart(*[FadeIn(note, shift=UP * 0.1) for note in annotations], lag_ratio=0.2)
        )
        self.wait(1.0)
        self.play(
            FadeOut(title),
            FadeOut(curve_label),
            FadeOut(axes),
            FadeOut(graph),
            FadeOut(annotations),
        )

    def coin_flip_stage(self):
        title = Text("Step 3 · Flipping a biased coin", font_size=40, color=YELLOW_B)
        title.to_edge(UP, buff=0.6)

        explanation_parts = [
            MathTex(r"U_i \sim \mathcal{U}(0,1)", font_size=34, color=GREY_B),
            MathTex(r"\text{clone if } \pi(S_i) > U_i", font_size=34, color=WHITE),
            Text(
                "Dead walkers are forced to clone so the swarm stays alive.",
                font_size=26,
                color=RED_B,
            ),
        ]
        for part in explanation_parts:
            self.limit_width(part, 4.6)
        explanation = VGroup(*explanation_parts).arrange(DOWN, buff=0.35, aligned_edge=LEFT)

        panel = RoundedRectangle(
            width=5.6,
            height=3.2,
            corner_radius=0.3,
            stroke_color=TEAL_E,
            fill_color="#142132",
            fill_opacity=0.6,
        )

        header = VGroup(
            Text("walker", font_size=24, color=GREY_B),
            Text("pressure", font_size=24, color=GREY_B),
            Text("threshold", font_size=24, color=GREY_B),
            Text("clones?", font_size=24, color=GREY_B),
        ).arrange(RIGHT, buff=0.6)

        rows = VGroup()
        sample_rows = [
            ("A", "0.72", "0.25", "✔"),
            ("B", "0.15", "0.74", "✘"),
            ("C (revive)", "1.00", "0.93", "✔"),
            ("D", "0.28", "0.19", "✔"),
        ]
        for walker, pressure, threshold, outcome in sample_rows:
            row = VGroup(
                Text(walker, font_size=26, color=WHITE),
                Text(pressure, font_size=26, color=YELLOW_B),
                Text(threshold, font_size=26, color=GREY_A),
                Text(outcome, font_size=30, color=GREEN_B if outcome == "✔" else RED_B),
            ).arrange(RIGHT, buff=0.6, aligned_edge=DOWN)
            rows.add(row)
        rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT)

        table_content = VGroup(header, rows).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        table_content.align_to(panel, LEFT).shift(RIGHT * 0.45)
        table_content.align_to(panel, UP).shift(DOWN * 0.55)

        table_group = VGroup(panel, table_content)

        content = VGroup(explanation, table_group).arrange(RIGHT, buff=1.4, aligned_edge=UP)
        content.next_to(title, DOWN, buff=0.5)
        content.shift(DOWN * 0.15 + RIGHT * 0.4)
        content[1].shift(RIGHT * 0.2)

        revive_highlight = SurroundingRectangle(
            rows[2],
            stroke_color=RED_B,
            stroke_width=3,
            buff=0.08,
        )

        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.play(LaggedStart(*[Write(line) for line in explanation], lag_ratio=0.2))
        self.play(Create(panel), FadeIn(header, shift=DOWN * 0.1))
        self.play(LaggedStart(*[FadeIn(row, shift=RIGHT * 0.2) for row in rows], lag_ratio=0.15))
        self.play(Create(revive_highlight))
        self.wait(1.0)
        self.play(
            FadeOut(title),
            FadeOut(content),
            FadeOut(revive_highlight),
        )

    def position_diffusion_stage(self):
        title = Text("Step 4 · Borrowing position with jitter", font_size=40, color=YELLOW_B)
        title.to_edge(UP, buff=0.6)

        formula = MathTex(
            r"x'_i = x_{c_i} + \sigma_x\,\zeta_i",
            r"\quad \text{with} \quad \zeta_i \sim \mathcal{N}(0, I)",
            font_size=38,
            color=WHITE,
        )
        formula.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        self.limit_width(formula, config.frame_width - 3.5)
        formula.next_to(title, DOWN, buff=0.4)

        cluster_center = ORIGIN
        companion = Dot(cluster_center, color=TEAL_B, radius=0.18)
        companion_label = MathTex(r"x_{c_i}", font_size=32, color=TEAL_B).next_to(
            companion, UP, buff=0.25
        )

        offsets = [
            np.array([0.7, 0.1, 0]),
            np.array([-0.6, 0.35, 0]),
            np.array([0.25, -0.45, 0]),
            np.array([-0.45, -0.35, 0]),
        ]

        clones = VGroup()
        for offset in offsets:
            glow = Circle(
                radius=0.32,
                color=WHITE,
                stroke_opacity=0.28,
                stroke_width=6,
            ).move_to(cluster_center + offset)
            clone = Dot(
                cluster_center + offset,
                color=interpolate_color(YELLOW_B, TEAL_B, np.random.uniform(0.2, 0.8)),
                radius=0.14,
            )
            clones.add(glow, clone)

        message = Text(
            "Small noise breaks symmetry so coupled swarms do not lock together.",
            font_size=28,
            color=YELLOW_C,
        )
        self.limit_width(message, config.frame_width - 3.5)
        message.next_to(companion, DOWN, buff=0.9)

        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.play(LaggedStart(*[Write(line) for line in formula], lag_ratio=0.2))
        self.play(FadeIn(companion), Write(companion_label))
        self.play(LaggedStart(*[FadeIn(obj, shift=OUT) for obj in clones], lag_ratio=0.2))
        self.play(FadeIn(message, shift=UP * 0.2))
        self.wait(1.0)
        self.play(
            FadeOut(title),
            FadeOut(formula),
            FadeOut(companion),
            FadeOut(companion_label),
            FadeOut(clones),
            FadeOut(message),
        )

    def momentum_blend_stage(self):
        title = Text("Step 5 · Blending momentum", font_size=40, color=YELLOW_B)
        title.to_edge(UP, buff=0.6)

        equations = VGroup(
            MathTex(r"V_{\text{COM}} = \frac{1}{M+1}\sum v_j", font_size=36, color=GREY_B),
            MathTex(
                r"v'_j = V_{\text{COM}} + \alpha_{\text{rest}}(v_j - V_{\text{COM}})",
                font_size=36,
                color=WHITE,
            ),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        self.limit_width(equations, 5.8)
        equations.next_to(title, DOWN, buff=0.4).to_edge(LEFT, buff=1.0)

        alpha_note = Text(
            "α_rest between 0 and 1 controls how inelastic the collision feels.",
            font_size=26,
            color=GREY_B,
        )
        self.limit_width(alpha_note, 5.8)
        alpha_note.next_to(equations, DOWN, buff=0.3, aligned_edge=LEFT)

        group_centers = [
            np.array([-2.3, -1.2, 0]),
            np.array([-0.6, -1.6, 0]),
            np.array([1.2, -1.1, 0]),
            np.array([2.5, -1.4, 0]),
        ]

        walkers = VGroup()
        for centre in group_centers:
            walkers.add(Dot(centre, radius=0.16, color=TEAL_B))

        initial_velocities = VGroup()
        for centre in group_centers:
            direction = np.array([0.55, 0.28, 0])
            initial_velocities.add(
                Arrow(
                    start=centre,
                    end=centre + direction,
                    buff=0,
                    color=YELLOW_B,
                    stroke_width=4,
                )
            )

        com_point = np.mean(group_centers, axis=0)
        com_dot = Dot(com_point, color=WHITE, radius=0.2)
        com_label = MathTex(r"V_{\text{COM}}", font_size=30, color=WHITE).next_to(
            com_dot, DOWN, buff=0.2
        )

        relaxed_velocities = VGroup()
        for centre in group_centers:
            relaxed_velocities.add(
                Arrow(
                    start=com_point,
                    end=com_point + 0.55 * (centre - com_point),
                    buff=0,
                    color=GREEN_B,
                    stroke_width=4,
                )
            )

        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.play(LaggedStart(*[Write(eq) for eq in equations], lag_ratio=0.2))
        self.play(FadeIn(alpha_note, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(dot) for dot in walkers], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowArrow(arr) for arr in initial_velocities], lag_ratio=0.1))
        self.play(FadeIn(com_dot), Write(com_label))
        self.play(ReplacementTransform(initial_velocities, relaxed_velocities))
        self.wait(1.0)
        self.play(
            FadeOut(title),
            FadeOut(equations),
            FadeOut(alpha_note),
            FadeOut(walkers),
            FadeOut(com_dot),
            FadeOut(com_label),
            FadeOut(relaxed_velocities),
        )

    def summary_stage(self):
        summary_box = RoundedRectangle(
            width=9.6,
            height=3.6,
            corner_radius=0.3,
            stroke_color=TEAL_E,
            fill_color="#121e2e",
            fill_opacity=0.7,
        )
        summary_box.move_to(ORIGIN)

        bullet_lines = [
            Text(
                "Cloning senses when walkers fall behind the local champions.",
                font_size=30,
                color=WHITE,
            ),
            Text(
                "The pressure curve ensures gentle nudges before drastic replacements.",
                font_size=30,
                color=YELLOW_B,
            ),
            Text(
                "Random thresholds keep diversity; jitter and momentum mixing smooth the reset.",
                font_size=30,
                color=TEAL_B,
            ),
        ]
        for line in bullet_lines:
            self.limit_width(line, summary_box.width - 1.0)
        bullets = VGroup(*bullet_lines).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        bullets.move_to(summary_box.get_center())

        self.play(Create(summary_box))
        self.play(
            LaggedStart(*[FadeIn(line, shift=RIGHT * 0.2) for line in bullets], lag_ratio=0.2)
        )
        self.wait(1.4)
        self.play(FadeOut(summary_box), FadeOut(bullets))

    def final_frame(self):
        outro = Text(
            "Together with kinetic transport, cloning keeps the swarm coherent yet agile.",
            font_size=34,
            color=YELLOW_B,
        )
        self.limit_width(outro, config.frame_width - 2.0)
        outro.to_edge(UP, buff=1.0)

        companion = Text(
            "It is the swarm's built-in error-correcting reflex.",
            font_size=30,
            color=WHITE,
        )
        self.limit_width(companion, config.frame_width - 2.2)
        companion.next_to(outro, DOWN, buff=0.6)

        self.play(FadeIn(outro, shift=DOWN * 0.3))
        self.play(FadeIn(companion, shift=DOWN * 0.3))
        self.wait(2.0)
        self.play(FadeOut(outro, shift=DOWN * 0.4), FadeOut(companion, shift=DOWN * 0.4))
