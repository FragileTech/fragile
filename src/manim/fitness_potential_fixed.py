import numpy as np

from manim import *


class FitnessPotentialCalculation(Scene):
    def construct(self):
        # Title
        title = Text("Fitness Potential Calculation", font_size=48, weight=BOLD, color=BLUE)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Scene 1: Overview
        self.overview()
        self.wait(2)
        self.clear()

        # Scene 2: Raw measurements
        self.raw_measurements()
        self.wait(2)
        self.clear()

        # Scene 3: Z-score concept
        self.z_score_concept()
        self.wait(2)
        self.clear()

        # Scene 4: Z-score calculation example
        self.z_score_example()
        self.wait(2)
        self.clear()

        # Scene 5: Logistic rescale concept
        self.logistic_concept()
        self.wait(2)
        self.clear()

        # Scene 6: Logistic rescale graph
        self.logistic_graph()
        self.wait(2)
        self.clear()

        # Scene 7: Multiplicative combination concept
        self.multiplicative_concept()
        self.wait(2)
        self.clear()

        # Scene 8: Complete example calculation
        self.complete_example()
        self.wait(3)
        self.clear()

        # Scene 9: Complete pipeline overview
        self.complete_pipeline()
        self.wait(3)

    def overview(self):
        """Scene 1: Clean pipeline overview"""
        title = Text("Complete Fitness Pipeline", font_size=44, color=TEAL, weight=BOLD)
        title.to_edge(UP, buff=0.8)
        self.play(Write(title))

        # Pipeline stages with more space
        stages = VGroup(
            VGroup(
                Text("Raw Values", font_size=28, weight=BOLD),
                MathTex(r"r_i, d_i", font_size=24, color=GREY),
            ).arrange(DOWN, buff=0.2),
            VGroup(
                Text("Z-Scores", font_size=28, weight=BOLD),
                MathTex(r"z_{r,i} = \frac{r_i - \mu_r}{\sigma'_r}", font_size=20, color=RED),
                MathTex(r"z_{d,i} = \frac{d_i - \mu_d}{\sigma'_d}", font_size=20, color=BLUE),
            ).arrange(DOWN, buff=0.15),
            VGroup(
                Text("Rescaled", font_size=28, weight=BOLD),
                MathTex(r"r'_i = g_A(z_{r,i}) + \eta", font_size=20, color=RED),
                MathTex(r"d'_i = g_A(z_{d,i}) + \eta", font_size=20, color=BLUE),
            ).arrange(DOWN, buff=0.15),
            VGroup(
                Rectangle(width=5, height=1.2, color=YELLOW, stroke_width=3, fill_opacity=0.1),
                Text("Fitness Potential", font_size=28, weight=BOLD, color=YELLOW),
                MathTex(r"V_i = (d'_i)^\beta \cdot (r'_i)^\alpha", font_size=24, color=YELLOW),
            ).arrange(DOWN, buff=0.1, center=False),
        ).arrange(DOWN, buff=0.8)

        stages.move_to(ORIGIN).shift(DOWN * 0.3)

        # Add flowing arrows
        arrows = VGroup()
        for i in range(len(stages) - 1):
            arrow = Arrow(
                start=stages[i].get_bottom() + DOWN * 0.15,
                end=stages[i + 1].get_top() + UP * 0.15,
                color=GREEN,
                stroke_width=4,
                tip_length=0.25,
            )
            arrows.add(arrow)

        self.play(LaggedStart(*[FadeIn(s) for s in stages], lag_ratio=0.4))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3))
        self.wait(2)

    def raw_measurements(self):
        """Scene 2: Raw measurements with clear separation"""
        title = Text("Step 1: Raw Measurements", font_size=40, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Central walker
        walker = Dot(point=ORIGIN, color=YELLOW, radius=0.25)
        walker_label = MathTex("w_i", font_size=36, color=YELLOW)
        walker_label.next_to(walker, DOWN, buff=0.4)

        self.play(FadeIn(walker), Write(walker_label))
        self.wait(1)

        # Left: Reward channel (clear positioning)
        reward_content = VGroup(
            Text("Reward Channel", font_size=24, weight=BOLD, color=RED),
            MathTex(
                r"r_i = R(x_i) - c_v ||v_i||^2 - \varphi_{\text{barrier}}(x_i)",
                font_size=18,
                color=RED,
            ),
            VGroup(
                Text("• External reward", font_size=16, color=WHITE),
                Text("• Velocity penalty", font_size=16, color=WHITE),
                Text("• Boundary barrier", font_size=16, color=WHITE),
            ).arrange(DOWN, buff=0.1, aligned_edge=LEFT),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)

        reward_box = Rectangle(width=4.5, height=2.5, color=RED, stroke_width=2, fill_opacity=0.05)
        reward_group = VGroup(reward_box, reward_content)
        reward_group.to_edge(LEFT, buff=0.5).shift(UP * 0.5)

        # Right: Diversity channel
        diversity_content = VGroup(
            Text("Diversity Channel", font_size=24, weight=BOLD, color=BLUE),
            MathTex(r"d_i = d_{\text{alg}}(i, \text{companion}_i)", font_size=18, color=BLUE),
            Text("Distance to paired companion", font_size=16, color=WHITE),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)

        diversity_box = Rectangle(
            width=4.5, height=2.5, color=BLUE, stroke_width=2, fill_opacity=0.05
        )
        diversity_group = VGroup(diversity_box, diversity_content)
        diversity_group.to_edge(RIGHT, buff=0.5).shift(UP * 0.5)

        self.play(FadeIn(reward_group))
        self.wait(1)
        self.play(FadeIn(diversity_group))
        self.wait(1)

        # Example at bottom with clear separation
        example = VGroup(
            Text("Example walker:", font_size=22, weight=BOLD),
            MathTex(r"r_i = 3.7", font_size=28, color=RED),
            MathTex(r"d_i = 1.2", font_size=28, color=BLUE),
        ).arrange(RIGHT, buff=0.8)
        example.to_edge(DOWN, buff=1.0)

        self.play(Write(example))
        self.wait(2)

    def z_score_concept(self):
        """Scene 3: Z-score concept with clean layout"""
        title = Text("Step 2: Z-Score Standardization", font_size=40, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Main concept
        concept = VGroup(
            Text("Convert raw values to standard scores", font_size=28),
            Text("Compare each walker to the swarm distribution", font_size=28),
        ).arrange(DOWN, buff=0.3)
        concept.shift(UP * 1.5)

        self.play(Write(concept))
        self.wait(1)

        # Large, clear formula
        formula = MathTex(r"z_i = \frac{\text{value}_i - \mu}{\sigma'}", font_size=56)
        formula.move_to(ORIGIN)

        self.play(Write(formula))
        self.wait(1)

        # Separated labels below with ample space
        labels = VGroup(
            VGroup(
                MathTex(r"\text{Walker's value}", font_size=24, color=GREEN),
                Arrow(ORIGIN, UP * 0.8, color=GREEN, stroke_width=3),
            ).arrange(DOWN, buff=0.1),
            VGroup(
                MathTex(r"\text{Swarm mean}", font_size=24, color=BLUE),
                Arrow(ORIGIN, UP * 0.8, color=BLUE, stroke_width=3),
            ).arrange(DOWN, buff=0.1),
            VGroup(
                MathTex(r"\text{Patched std dev}", font_size=24, color=ORANGE),
                Arrow(ORIGIN, UP * 0.8, color=ORANGE, stroke_width=3),
            ).arrange(DOWN, buff=0.1),
        ).arrange(RIGHT, buff=1.5)
        labels.shift(DOWN * 2.0)

        # Position arrows to point to formula components
        labels[0][1].put_start_and_end_on(labels[0][0].get_top(), formula[0][0:7].get_bottom())
        labels[1][1].put_start_and_end_on(labels[1][0].get_top(), formula[0][8].get_bottom())
        labels[2][1].put_start_and_end_on(labels[2][0].get_top(), formula[0][9].get_bottom())

        self.play(LaggedStart(*[Write(l) for l in labels], lag_ratio=0.3))
        self.wait(2)

    def z_score_example(self):
        """Scene 4: Z-score calculation example"""
        title = Text("Step 2: Z-Score Example", font_size=40, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Two channel calculations side by side
        reward_calc = VGroup(
            Text("Reward Channel", font_size=26, weight=BOLD, color=RED),
            MathTex(r"z_{r,i} = \frac{r_i - \mu_r}{\sigma'_r}", font_size=24, color=RED),
            MathTex(r"z_{r,i} = \frac{3.7 - 2.5}{1.0} = 1.2", font_size=22, color=RED),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)

        diversity_calc = VGroup(
            Text("Diversity Channel", font_size=26, weight=BOLD, color=BLUE),
            MathTex(r"z_{d,i} = \frac{d_i - \mu_d}{\sigma'_d}", font_size=24, color=BLUE),
            MathTex(r"z_{d,i} = \frac{1.2 - 2.0}{1.0} = -0.8", font_size=22, color=BLUE),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)

        # Position with clear separation
        both_calcs = VGroup(reward_calc, diversity_calc).arrange(RIGHT, buff=2.0)
        both_calcs.move_to(ORIGIN)

        self.play(Write(reward_calc))
        self.wait(1)
        self.play(Write(diversity_calc))
        self.wait(2)

        # Result summary at bottom
        result = VGroup(
            Text("Results:", font_size=24, weight=BOLD),
            MathTex(r"z_{r,i} = 1.2 \text{ (above average)}", font_size=24, color=RED),
            MathTex(r"z_{d,i} = -0.8 \text{ (below average)}", font_size=24, color=BLUE),
        ).arrange(DOWN, buff=0.3)
        result.to_edge(DOWN, buff=1.0)

        self.play(Write(result))
        self.wait(2)

    def logistic_concept(self):
        """Scene 5: Logistic rescale concept"""
        title = Text("Step 3: Logistic Rescale", font_size=40, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Problem statement
        problem = VGroup(
            Text("Problem:", font_size=28, weight=BOLD, color=RED),
            Text("Z-scores can be any real number", font_size=24),
            Text("Need bounded positive interval", font_size=24),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        problem.shift(UP * 1.0 + LEFT * 2.0)

        self.play(Write(problem))
        self.wait(1)

        # Solution
        solution = VGroup(
            Text("Solution:", font_size=28, weight=BOLD, color=GREEN),
            Text("Map to bounded positive interval", font_size=24),
            MathTex(r"g_A(z) = \frac{2}{1 + e^{-z}}", font_size=32, color=GREEN),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        solution.shift(UP * 1.0 + RIGHT * 2.0)

        self.play(Write(solution))
        self.wait(1)

        # Properties box at bottom
        properties = VGroup(
            Text("Properties:", font_size=26, weight=BOLD, color=TEAL),
            Text("• Range: (0, 2)", font_size=22),
            Text("• Smooth & monotonic", font_size=22),
            Text("• High z → high value", font_size=22),
            Text("• Low z → low value", font_size=22),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        prop_box = Rectangle(width=6, height=2.5, color=TEAL, stroke_width=2, fill_opacity=0.1)
        prop_group = VGroup(prop_box, properties)
        prop_group.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(prop_group))
        self.wait(2)

    def logistic_graph(self):
        """Scene 6: Clean logistic function graph"""
        title = Text("Step 3: Logistic Function Graph", font_size=40, color=YELLOW)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Formula prominently displayed
        formula = MathTex(r"g_A(z) = \frac{2}{1 + e^{-z}}", font_size=48, color=GREEN)
        formula.next_to(title, DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait(1)

        # Graph with clear axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 2.2, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"color": WHITE, "stroke_width": 2},
            tips=True,
        )
        axes.to_edge(DOWN, buff=0.8)

        x_label = axes.get_x_axis_label(MathTex("z", font_size=28))
        y_label = axes.get_y_axis_label(MathTex("g_A(z)", font_size=28), edge=UP, direction=UP)

        # Plot the logistic function
        def logistic(z):
            return 2 / (1 + np.exp(-z))

        graph = axes.plot(logistic, x_range=[-4, 4], color=GREEN, stroke_width=5)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph), run_time=2)
        self.wait(1)

        # Key points with better positioning
        points_data = [(-2, logistic(-2), RED), (0, logistic(0), YELLOW), (2, logistic(2), BLUE)]
        dots = VGroup()
        labels = VGroup()

        for z_val, y_val, color in points_data:
            dot = Dot(axes.c2p(z_val, y_val), color=color, radius=0.08)
            label = MathTex(f"z={z_val} \\rightarrow {y_val:.2f}", font_size=18, color=color)

            # Position labels to avoid overlap
            if z_val < 0:
                label.next_to(dot, LEFT + UP, buff=0.3)
            elif z_val == 0:
                label.next_to(dot, DOWN, buff=0.4)
            else:
                label.next_to(dot, RIGHT + UP, buff=0.3)

            dots.add(dot)
            labels.add(label)

        self.play(FadeIn(dots))
        self.play(Write(labels))
        self.wait(2)

    def multiplicative_concept(self):
        """Scene 7: Multiplicative combination concept"""
        title = Text("Step 4: Multiplicative Combination", font_size=40, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # First: Add stability floor
        floor_step = VGroup(
            Text("First: Add stability floor η", font_size=28, weight=BOLD),
            MathTex(r"r'_i = g_A(z_{r,i}) + \eta", font_size=32, color=RED),
            MathTex(r"d'_i = g_A(z_{d,i}) + \eta", font_size=32, color=BLUE),
        ).arrange(DOWN, buff=0.4)
        floor_step.shift(UP * 1.5)

        self.play(Write(floor_step))
        self.wait(1)

        # Main multiplicative formula with emphasis
        main_formula = MathTex(
            r"V_i = (d'_i)^\beta \cdot (r'_i)^\alpha", font_size=64, color=YELLOW
        )
        main_formula.move_to(ORIGIN).shift(DOWN * 0.5)

        box = SurroundingRectangle(
            main_formula, color=YELLOW, buff=0.5, corner_radius=0.3, stroke_width=4
        )

        self.play(Write(main_formula))
        self.play(Create(box))
        self.wait(2)

        # Key insight at bottom
        insight = VGroup(
            Text("Key Insight:", font_size=26, weight=BOLD, color=TEAL),
            Text("Multiplicative: BOTH channels must be high for high fitness", font_size=24),
            Text("Low on either channel → Low overall fitness", font_size=24, color=RED),
        ).arrange(DOWN, buff=0.3)
        insight.to_edge(DOWN, buff=0.8)

        insight_box = Rectangle(width=10, height=1.8, color=TEAL, stroke_width=2, fill_opacity=0.1)
        insight_group = VGroup(insight_box, insight)

        self.play(FadeIn(insight_group))
        self.wait(2)

    def complete_example(self):
        """Scene 8: Complete worked example"""
        title = Text("Complete Example Calculation", font_size=40, color=YELLOW)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Parameters
        params = MathTex(r"\text{Parameters: } \alpha=1, \beta=1, \eta=0.1", font_size=24)
        params.next_to(title, DOWN, buff=0.4)
        self.play(Write(params))
        self.wait(1)

        # Step by step calculation
        steps = VGroup(
            # Raw values
            VGroup(
                Text("1. Raw values:", font_size=22, weight=BOLD),
                MathTex(r"r_i = 3.7, \quad d_i = 1.2", font_size=24),
            ).arrange(DOWN, buff=0.2, aligned_edge=LEFT),
            # Z-scores
            VGroup(
                Text("2. Z-scores:", font_size=22, weight=BOLD),
                MathTex(r"z_{r,i} = 1.2, \quad z_{d,i} = -0.8", font_size=24),
            ).arrange(DOWN, buff=0.2, aligned_edge=LEFT),
            # Logistic rescale
            VGroup(
                Text("3. Logistic rescale:", font_size=22, weight=BOLD),
                MathTex(r"g_A(1.2) = 1.663, \quad g_A(-0.8) = 0.690", font_size=24),
            ).arrange(DOWN, buff=0.2, aligned_edge=LEFT),
            # Add floor
            VGroup(
                Text("4. Add floor:", font_size=22, weight=BOLD),
                MathTex(r"r'_i = 1.663 + 0.1 = 1.763", font_size=24, color=RED),
                MathTex(r"d'_i = 0.690 + 0.1 = 0.790", font_size=24, color=BLUE),
            ).arrange(DOWN, buff=0.2, aligned_edge=LEFT),
            # Final calculation
            VGroup(
                Text("5. Final fitness:", font_size=22, weight=BOLD),
                MathTex(r"V_i = (0.790)^1 \cdot (1.763)^1 = 1.393", font_size=28, color=GREEN),
            ).arrange(DOWN, buff=0.2, aligned_edge=LEFT),
        ).arrange(DOWN, buff=0.6, aligned_edge=LEFT)

        steps.next_to(params, DOWN, buff=0.8).shift(LEFT * 2.0)

        # Animate steps one by one
        for step in steps:
            self.play(Write(step))
            self.wait(1)

        # Highlight final result
        final_box = SurroundingRectangle(
            steps[-1], color=GREEN, buff=0.3, corner_radius=0.2, stroke_width=3
        )
        self.play(Create(final_box))
        self.wait(2)

    def complete_pipeline(self):
        """Scene 9: Final complete pipeline view"""
        title = Text("Complete Fitness Pipeline", font_size=44, weight=BOLD, color=TEAL)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Streamlined pipeline view
        pipeline_elements = VGroup(
            # Raw
            VGroup(
                Rectangle(width=3, height=1, color=WHITE, stroke_width=2, fill_opacity=0.1),
                Text("Raw Values", font_size=20, weight=BOLD),
                MathTex(r"r_i, d_i", font_size=18),
            ).arrange(DOWN, buff=0.1),
            # Z-scores
            VGroup(
                Rectangle(width=3, height=1, color=ORANGE, stroke_width=2, fill_opacity=0.1),
                Text("Z-Scores", font_size=20, weight=BOLD),
                MathTex(r"z_{r,i}, z_{d,i}", font_size=18),
            ).arrange(DOWN, buff=0.1),
            # Rescaled
            VGroup(
                Rectangle(width=3, height=1, color=GREEN, stroke_width=2, fill_opacity=0.1),
                Text("Rescaled", font_size=20, weight=BOLD),
                MathTex(r"r'_i, d'_i", font_size=18),
            ).arrange(DOWN, buff=0.1),
            # Final
            VGroup(
                Rectangle(width=4, height=1.5, color=YELLOW, stroke_width=3, fill_opacity=0.2),
                Text("Fitness Potential", font_size=22, weight=BOLD, color=YELLOW),
                MathTex(r"V_i = (d'_i)^\beta \cdot (r'_i)^\alpha", font_size=20, color=YELLOW),
            ).arrange(DOWN, buff=0.1),
        ).arrange(RIGHT, buff=1.2)

        pipeline_elements.next_to(title, DOWN, buff=1.0)

        # Arrows between elements
        arrows = VGroup()
        for i in range(len(pipeline_elements) - 1):
            arrow = Arrow(
                pipeline_elements[i].get_right(),
                pipeline_elements[i + 1].get_left(),
                color=BLUE,
                stroke_width=4,
                tip_length=0.3,
            )
            arrows.add(arrow)

        # Animate complete pipeline
        self.play(LaggedStart(*[FadeIn(elem) for elem in pipeline_elements], lag_ratio=0.3))
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.2))
        self.wait(2)

        # Summary insights
        insights = VGroup(
            Text("Key Properties:", font_size=24, weight=BOLD, color=TEAL),
            Text("✓ Standardizes heterogeneous measurements", font_size=20),
            Text("✓ Ensures bounded positive values", font_size=20),
            Text("✓ Multiplicative: both channels matter", font_size=20),
            Text("✓ Stable numerical computation", font_size=20),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        insights.to_edge(DOWN, buff=0.8)
        insights_bg = Rectangle(width=8, height=2.5, color=TEAL, stroke_width=2, fill_opacity=0.05)
        insights_group = VGroup(insights_bg, insights)

        self.play(FadeIn(insights_group))
        self.wait(3)


# Alternative compact scene for quick overview
class CompactPipeline(Scene):
    def construct(self):
        title = Text("Fitness Pipeline: Quick Overview", font_size=36, weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Horizontal pipeline with formulas
        pipeline = VGroup(
            VGroup(
                Text("Raw", font_size=18, weight=BOLD),
                MathTex(r"r_i, d_i", font_size=16),
            ).arrange(DOWN, buff=0.1),
            MathTex(r"\rightarrow", font_size=24),
            VGroup(
                Text("Z-Score", font_size=18, weight=BOLD),
                MathTex(r"\frac{x_i - \mu}{\sigma'}", font_size=16),
            ).arrange(DOWN, buff=0.1),
            MathTex(r"\rightarrow", font_size=24),
            VGroup(
                Text("Logistic", font_size=18, weight=BOLD),
                MathTex(r"\frac{2}{1 + e^{-z}}", font_size=16),
            ).arrange(DOWN, buff=0.1),
            MathTex(r"\rightarrow", font_size=24),
            VGroup(
                Text("Fitness", font_size=18, weight=BOLD, color=YELLOW),
                MathTex(r"(d'_i)^\beta (r'_i)^\alpha", font_size=16, color=YELLOW),
            ).arrange(DOWN, buff=0.1),
        ).arrange(RIGHT, buff=0.5)

        pipeline.move_to(ORIGIN)

        self.play(LaggedStart(*[Write(elem) for elem in pipeline], lag_ratio=0.3))
        self.wait(3)
