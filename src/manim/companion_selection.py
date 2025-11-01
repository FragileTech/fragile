from manim import *
import numpy as np

class CompanionSelection(Scene):
    def construct(self):
        # Title
        title = Text("Companion Selection Mechanism", font_size=48, weight=BOLD, color=BLUE)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))
        
        # Scene 1: Setup the problem
        self.setup_problem()
        self.wait(2)
        self.clear()
        
        # Scene 2: Show viscous forces
        self.viscous_forces()
        self.wait(2)
        self.clear()
        
        # Scene 3: Momentum phases
        self.momentum_phases()
        self.wait(2)
        self.clear()
        
        # Scene 4: Complex amplitude calculation
        self.complex_amplitude()
        self.wait(2)
        self.clear()
        
        # Scene 5: Quantum interference
        self.quantum_interference()
        self.wait(2)
    
    def setup_problem(self):
        """Scene 1: Introduce the problem"""
        # Title
        problem_title = Text("The Problem", font_size=40, color=YELLOW)
        problem_title.to_edge(UP)
        self.play(Write(problem_title))
        
        # Show a particle that wants to clone
        parent = Dot(point=ORIGIN, color=RED, radius=0.2)
        parent_label = Text("Parent\n(wants to clone)", font_size=24, color=RED)
        parent_label.next_to(parent, DOWN, buff=0.5)
        
        self.play(FadeIn(parent), Write(parent_label))
        self.wait(1)
        
        # Show potential companions in a circle
        n_companions = 8
        radius = 2.5
        companions = VGroup()
        companion_labels = VGroup()
        
        for i in range(n_companions):
            angle = i * 2 * PI / n_companions
            pos = radius * np.array([np.cos(angle), np.sin(angle), 0])
            companion = Dot(point=pos, color=BLUE, radius=0.15)
            label = MathTex(f"w_{{{i+1}}}", font_size=28, color=BLUE)
            label.next_to(companion, direction=pos/np.linalg.norm(pos), buff=0.3)
            
            companions.add(companion)
            companion_labels.add(label)
        
        self.play(LaggedStart(*[FadeIn(c) for c in companions], lag_ratio=0.1))
        self.play(LaggedStart(*[Write(l) for l in companion_labels], lag_ratio=0.1))
        self.wait(1)
        
        # Question
        question = Text("Which companion should the parent choose?", 
                       font_size=32, color=YELLOW)
        question.to_edge(DOWN)
        self.play(Write(question))
        self.wait(2)
    
    def viscous_forces(self):
        """Scene 2: Explain viscous forces"""
        title = Text("Step 1: Viscous Forces", font_size=40, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Central parent particle
        parent = Dot(point=ORIGIN, color=RED, radius=0.2)
        parent_label = Text("Parent", font_size=24, color=RED)
        parent_label.next_to(parent, DOWN, buff=0.3)
        
        # Three companion candidates
        companions_pos = [
            np.array([2, 1.5, 0]),
            np.array([-1.5, 2, 0]),
            np.array([1, -2, 0])
        ]
        companions = VGroup()
        force_arrows = VGroup()
        force_labels = VGroup()
        
        for i, pos in enumerate(companions_pos):
            companion = Dot(point=pos, color=BLUE, radius=0.15)
            companions.add(companion)
            
            # Force arrow from companion to parent
            force = Arrow(
                start=pos,
                end=pos * 0.4,  # Pointing toward parent
                color=GREEN,
                buff=0.15,
                stroke_width=4,
                tip_length=0.25
            )
            force_arrows.add(force)
            
            # Force label
            label = MathTex(f"F_{{{i+1}}}^{{\\text{{visc}}}}", font_size=28, color=GREEN)
            label.next_to(force, RIGHT if pos[0] > 0 else LEFT, buff=0.2)
            force_labels.add(label)
        
        self.play(FadeIn(parent), Write(parent_label))
        self.play(LaggedStart(*[FadeIn(c) for c in companions], lag_ratio=0.2))
        self.wait(1)
        
        # Explanation text
        explanation = VGroup(
            Text("Each companion exerts a", font_size=26),
            Text("viscous force on the parent", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT)
        explanation.to_corner(DL).shift(UP * 0.5)
        
        self.play(Write(explanation))
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in force_arrows], lag_ratio=0.3),
            LaggedStart(*[Write(label) for label in force_labels], lag_ratio=0.3)
        )
        self.wait(2)
        
        # Show formula
        formula = MathTex(
            "F_i^{\\text{visc}} = \\text{force from walker } i \\text{ to parent}",
            font_size=30,
            color=GREEN
        )
        formula.to_edge(DOWN).shift(UP * 0.3)
        self.play(Write(formula))
        self.wait(2)
    
    def momentum_phases(self):
        """Scene 3: Introduce momentum-based phases"""
        title = Text("Step 2: Momentum Phases", font_size=40, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show a companion with velocity
        companion = Dot(point=np.array([2, 1, 0]), color=BLUE, radius=0.2)
        companion_label = MathTex("w_i", font_size=32, color=BLUE)
        companion_label.next_to(companion, UP, buff=0.3)
        
        # Velocity vector
        vel_vector = Arrow(
            start=companion.get_center(),
            end=companion.get_center() + np.array([1.5, 0.8, 0]),
            color=ORANGE,
            buff=0,
            stroke_width=5,
            tip_length=0.3
        )
        vel_label = MathTex("v_i", font_size=32, color=ORANGE)
        vel_label.next_to(vel_vector.get_end(), RIGHT, buff=0.2)
        
        self.play(FadeIn(companion), Write(companion_label))
        self.play(GrowArrow(vel_vector), Write(vel_label))
        self.wait(1)
        
        # Explain the phase
        explanation = VGroup(
            Text("The velocity encodes a quantum phase:", font_size=28),
            MathTex("\\phi_i = \\frac{m}{\\hbar_{\\text{eff}}} v_i", font_size=36, color=ORANGE),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        explanation.to_corner(DL).shift(UP * 0.5)
        
        self.play(Write(explanation[0]))
        self.play(Write(explanation[1]))
        self.wait(2)
        
        # Show complex plane
        plane = ComplexPlane(
            x_range=[-1.5, 1.5],
            y_range=[-1.5, 1.5],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)
        
        plane_title = Text("Complex Plane", font_size=24, color=BLUE_C)
        plane_title.next_to(plane, UP, buff=0.3)
        
        self.play(Create(plane), Write(plane_title))
        
        # Show phase as rotating vector
        phase_angle = PI / 3
        phase_vector = Arrow(
            start=plane.n2p(0),
            end=plane.n2p(np.exp(1j * phase_angle)),
            color=YELLOW,
            buff=0,
            stroke_width=4,
            tip_length=0.2
        )
        phase_label = MathTex("e^{i\\phi_i}", font_size=28, color=YELLOW)
        phase_label.next_to(phase_vector.get_end(), UR, buff=0.1)
        
        self.play(GrowArrow(phase_vector), Write(phase_label))
        self.wait(2)
        
        # Rotate the phase vector
        self.play(
            Rotate(phase_vector, angle=PI, about_point=plane.n2p(0)),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)
    
    def complex_amplitude(self):
        """Scene 4: Build the complex amplitude"""
        title = Text("Step 3: Complex Amplitude", font_size=40, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        
        # The formula builds step by step
        explanation = Text(
            "Combine the viscous force with the phase:",
            font_size=28
        )
        explanation.next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation))
        
        # Build formula step by step
        formulas = [
            MathTex("c_i", font_size=40),
            MathTex("c_i", "=", "F_i^{\\text{visc}}", font_size=40),
            MathTex("c_i", "=", "F_i^{\\text{visc}}", "\\cdot", font_size=40),
            MathTex("c_i", "=", "F_i^{\\text{visc}}", "\\cdot", "e^{i \\phi_i}", font_size=40),
            MathTex(
                "c_i", "=", "F_i^{\\text{visc}}", "\\cdot", 
                "e^{i m v_i / \\hbar_{\\text{eff}}}",
                font_size=40
            ),
        ]
        
        # Color the parts
        for formula in formulas:
            if len(formula) >= 3:
                formula[2].set_color(GREEN)  # Force
            if len(formula) >= 5:
                formula[4].set_color(ORANGE)  # Phase
        
        current_formula = formulas[0]
        current_formula.move_to(ORIGIN)
        self.play(Write(current_formula))
        self.wait(1)
        
        for next_formula in formulas[1:]:
            next_formula.move_to(ORIGIN)
            self.play(TransformMatchingTex(current_formula, next_formula))
            current_formula = next_formula
            self.wait(1)
        
        # Box the final formula
        box = SurroundingRectangle(current_formula, color=YELLOW, buff=0.3)
        self.play(Create(box))
        self.wait(1)
        
        # Move to top
        final_group = VGroup(current_formula, box)
        self.play(final_group.animate.scale(0.8).to_edge(UP).shift(DOWN * 2))
        
        # Explanation
        interpretation = VGroup(
            Text("This complex amplitude encodes:", font_size=28),
            Text("• Magnitude: strength of viscous coupling", font_size=24, color=GREEN),
            Text("• Phase: quantum interference from velocity", font_size=24, color=ORANGE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        interpretation.next_to(final_group, DOWN, buff=0.8)
        
        self.play(Write(interpretation))
        self.wait(3)
    
    def quantum_interference(self):
        """Scene 5: Show quantum interference pattern"""
        title = Text("Step 4: Quantum Interference", font_size=40, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show parent
        parent = Dot(point=ORIGIN, color=RED, radius=0.2)
        parent_label = Text("Parent", font_size=24, color=RED)
        parent_label.next_to(parent, DOWN, buff=0.3)
        
        self.play(FadeIn(parent), Write(parent_label))
        
        # Show multiple companions with their complex amplitudes
        n_companions = 6
        radius = 2.5
        
        for i in range(n_companions):
            angle = i * 2 * PI / n_companions
            pos = radius * np.array([np.cos(angle), np.sin(angle), 0])
            
            # Companion
            companion = Dot(point=pos, color=BLUE, radius=0.12)
            
            # Random complex amplitude (for visualization)
            phase = np.random.uniform(0, 2*PI)
            magnitude = np.random.uniform(0.3, 1.0)
            
            # Arrow representing complex amplitude
            amplitude_end = companion.get_center() + magnitude * 0.8 * np.array([
                np.cos(phase), np.sin(phase), 0
            ])
            amplitude_arrow = Arrow(
                start=companion.get_center(),
                end=amplitude_end,
                color=YELLOW,
                buff=0,
                stroke_width=3,
                tip_length=0.15
            )
            
            # Label
            label = MathTex(f"c_{i+1}", font_size=20, color=YELLOW)
            label.next_to(amplitude_arrow.get_end(), 
                         direction=amplitude_end - companion.get_center(), 
                         buff=0.1)
            
            self.play(
                FadeIn(companion),
                GrowArrow(amplitude_arrow),
                Write(label),
                run_time=0.5
            )
        
        self.wait(2)
        
        # Selection probability formula
        formula = MathTex(
            "P(\\text{select } i) \\propto |c_i|^2",
            font_size=36,
            color=YELLOW
        )
        formula.to_edge(DOWN).shift(UP * 0.5)
        
        box = SurroundingRectangle(formula, color=YELLOW, buff=0.2)
        
        self.play(Write(formula))
        self.play(Create(box))
        self.wait(1)
        
        # Final explanation
        explanation = VGroup(
            Text("The companion is selected probabilistically", font_size=26),
            Text("based on the squared magnitude of", font_size=26),
            Text("the complex amplitude |c_i|²", font_size=26, color=YELLOW),
            Text("", font_size=20),
            Text("This creates quantum interference patterns!", font_size=26, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        explanation.to_corner(UL).shift(DOWN * 0.5 + RIGHT * 0.3)
        
        explanation_bg = BackgroundRectangle(
            explanation,
            color=BLACK,
            fill_opacity=0.8,
            buff=0.3
        )
        
        self.play(FadeIn(explanation_bg), Write(explanation))
        self.wait(3)
