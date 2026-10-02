from pathlib import Path
import re
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

ROOT = Path('/home/guillem/fragile')
BASE = ROOT / 'docs/source/2_fractal_gas'
OUT = ROOT / 'output/pdf/volume_2_qft_yang_mills_proof_roadmap.pdf'
pdfmetrics.registerFont(TTFont('Lato', '/usr/share/fonts/truetype/lato/Lato-Regular.ttf'))
pdfmetrics.registerFont(TTFont('LatoBold', '/usr/share/fonts/truetype/lato/Lato-Bold.ttf'))
pdfmetrics.registerFont(TTFont('LatoItalic', '/usr/share/fonts/truetype/lato/Lato-Italic.ttf'))
pdfmetrics.registerFontFamily('Lato', normal='Lato', bold='LatoBold', italic='LatoItalic', boldItalic='LatoBold')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body', fontName='Lato', fontSize=10.4, leading=15.1, spaceAfter=9, textColor=colors.HexColor('#233342')))
styles.add(ParagraphStyle(name='TitleCustom', fontName='LatoBold', fontSize=31, leading=36, textColor=colors.HexColor('#12354A'), spaceAfter=20))
styles.add(ParagraphStyle(name='Section', fontName='LatoBold', fontSize=21, leading=25, spaceAfter=17, textColor=colors.HexColor('#12354A')))
styles.add(ParagraphStyle(name='Sub', fontName='LatoBold', fontSize=11.3, leading=15, spaceBefore=9, spaceAfter=6, textColor=colors.HexColor('#087F8C')))
styles.add(ParagraphStyle(name='SmallCustom', fontName='Lato', fontSize=8.3, leading=11.5, spaceAfter=6, textColor=colors.HexColor('#4B5C68')))
styles.add(ParagraphStyle(name='Equation', fontName='Lato', fontSize=11, leading=17, leftIndent=15, rightIndent=12, spaceBefore=5, spaceAfter=11, textColor=colors.HexColor('#12354A'), backColor=colors.HexColor('#EFF5F7'), borderPadding=9))
styles.add(ParagraphStyle(name='Theorem', fontName='Lato', fontSize=9.1, leading=12.6, leftIndent=10, borderColor=colors.HexColor('#B9D7DB'), borderWidth=.5, borderPadding=8, spaceAfter=9))

catalog = {}
for path in BASE.rglob('*.md'):
    text = path.read_text()
    for match in re.finditer(r'^:+\{prf:([^}]+)\} ([^\n]+)\n:label: ([^\n]+)', text, re.M):
        label = match[3].strip()
        catalog[label] = dict(kind=match[1], title=match[2].strip(), file=str(path.relative_to(BASE)), line=text[:match.start()].count('\n')+1)

used = []
story = []
def p(text, style='Body'):
    story.append(Paragraph(text, styles[style]))
def eq(text): p(escape(text), 'Equation')
def sub(text): p(escape(text), 'Sub')
def theorem(label, role):
    item = catalog[label]
    if label not in used: used.append(label)
    p('<b>'+escape(item['title'])+'</b><br/>'+escape(role)+'<br/><font color="#52747C">'+escape(label)+'</font>', 'Theorem')
def stage(number, title, paragraphs, equations, tools, refs, handoff):
    story.append(PageBreak())
    p(f'STAGE {number:02d}', 'SmallCustom')
    p(title, 'Section')
    for text in paragraphs: p(text)
    for text in equations: eq(text)
    sub('Tools and quantitative inputs')
    p(tools)
    if refs: sub('Principal results in Volume 2')
    for label, role in refs: theorem(label, role)
    sub('Output carried to the next stage')
    p(handoff)

p('VOLUME 2 / PROOF PROGRAM', 'SmallCustom')
story.append(Spacer(1, 55))
p('From the Euclidean Gas<br/>to Quantum Yang-Mills<br/>and a Mass Gap', 'TitleCustom')
p('An end-to-end roadmap through the recorded dynamics, population limit, analytic estimates, field reconstruction, and continuum passage.', 'Body')
story.append(Spacer(1, 24))
eq('Gas dynamics → population control → recorded fields → continuum correlations → quantum reconstruction → surviving physical gap')
p('Prepared from the repository version of Volume 2, Fractal Gas.<br/>Source snapshot: 1 October 2026.', 'SmallCustom')
sub('How to read this document')
p('This document compiles the proof program already developed in Volume 2. Each stage identifies its objects, analytic tools, principal named results, and the output needed downstream. The theorem titles and labels are taken directly from the source chapters; the source register at the end gives their locations.')
p('The explanatory organization is a synthesis of those chapters. Published theorem statements retain their specified laws and hypotheses. Completion targets are stated as tasks, rather than as additional proved theorems. The endpoint is a nontrivial four-dimensional Yang-Mills quantum field theory with a positive physical mass gap.')
p('The gas fixes the stochastic law. The Fractal Set preserves its recorded interactions. The mean field controls population approximation and smooth reconstruction. Poincare, LSI, entropy, moment, covariance, and consistency estimates control the limits. The physical field reconstruction identifies the Hamiltonian to which the final gap theorem applies.')

story.append(PageBreak())
p('The dependency structure', 'Section')
p('The program has two tracks that join at the field law. The analytic track controls the actual swarm and its limiting population. The reconstruction track expresses the same history as geometry, gauge observables, and matter operators. Their common law allows estimates to pass from one description to the other.')
rows = [
('Stages', 'Construction', 'Required output'),
('1-3', 'Gas, collisions, confinement', 'Complete kernel; moments and tails'),
('4-6', 'Long-time law and mean field', 'Selected ensemble; chaos and time control'),
('7-9', 'Poincare, LSI, regularity', 'Coercivity; concentration; smooth fields'),
('10-12', 'Fractal Set and native action', 'Exact observable law; gauge response'),
('13-14', 'Matter and fluctuations', 'CAR evolution; correlation hierarchy'),
('15-16', 'Continuum control', 'Operator, action, and hierarchy limits'),
('17-18', 'Physical QFT and gap', 'Physical Hamiltonian; positive gap'),
]
t=Table([[Paragraph(escape(c), styles['SmallCustom']) for c in row] for row in rows], colWidths=[55,175,260], hAlign='LEFT')
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFEDF0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D1DEE3')),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
story.append(t)
sub('The law ledger')
p('Keep four objects visible: the finite conservative invariant law; the finite killed-chain QSD; the nonlinear fixed-step population law; and the reconstructed field history law. A theorem transfers between them through an explicit density, conditioning, observable, or operator identification.')
sub('The scale ledger')
p('N denotes walkers; M denotes observations used by a local estimator; h is the algorithm timestep; n is the number of relaxation updates; epsilon is a reconstruction bandwidth; a is a field cutoff; and L denotes physical volume scale. The continuum chapter uses N for estimator samples. Here M distinguishes that sample count from swarm size. Correlated recordings are counted through their covariance bound.')
sub('The central analytic bridge')
eq('Confinement → functional inequalities → relaxation and concentration → smooth population fields → controlled shrinking neighborhoods → continuum field laws')
p('The endpoint also needs physical spacetime symmetry, reflection positivity, a nontrivial field sector, and identification of the physical transfer Hamiltonian. These are gathered as explicit completion tasks in Stage 17, keeping the rest of the roadmap focused on the constructions and estimates.')

stage(1, 'Fix the Euclidean gas', [
'Begin with the complete marked swarm S = ((x_i, v_i, a_i)) with positions, velocities, and alive/dead status. Dead slots retain their coordinates. Freeze the input frame for measurement, fitness, donor choice, and acceptance; this fixes the update order and prevents updated donor coordinates from leaking into the same cloning stage.',
'The canonical update samples one measurement companion, computes regularized population statistics and positive logistic fitness, samples a cloning donor, and accepts according to the frozen score. Accepted edges define connected collision components. Copy recipient positions with Gaussian jitter, rotate each component about its center-of-mass velocity, apply BAOAB and final position noise, cap velocities, and classify the terminal boundary once.',
'Declare the absorbing or conservative configuration, the objective and force, the noise laws, the internal marks, and all parameter floors. If historical donors or mutable providers are used, include their memory in the complete state. This stage fixes the stochastic object used throughout the program.'
], ['L_N = (1/N) sum_i delta_(x_i,v_i,a_i) ;   S_(n+1) ~ P_(N,h)(S_n, ·)'],
'Marked Markov state; positive fitness and normalization floors; bounded comparison features; Gaussian innovations; Feller regularity. The Euclidean Gas Update is the defining algorithm in convergence_program/02_euclidean_gas.md, label alg-euclidean-gas.', [],
'A complete finite-particle transition and a record contract that specifies every random choice and intermediate stage. Subsequent mean-field and field constructions use this kernel.')

stage(2, 'Resolve cloning and collision geometry', [
'An accepted donor edge changes the recipient position and joins its endpoints into a collision component. Overlapping groups form a single connected component. The shared orthogonal rotation acts on the component relative velocities, so each walker has one velocity destination.',
'The tagged-walker population description must retain its random collision neighborhood. Strict fitness increase along accepted live edges supplies component control in the canonical current-frame construction. Rooted component convergence connects these finite neighborhoods to the limiting collision law.',
'Compute momentum, restitution energy, and the cross-walker covariance created by the shared rotation. Retain these terms in fluctuation and mean-field calculations: the noises of members of one component are correlated.'
], ['v_i^new = v_bar_C + alpha R_C (v_i - v_bar_C) ;   sum_(i in C) v_i^new = sum_(i in C) v_i'],
'Accepted-edge graph; rooted-component truncation; moment bounds at every component order; shared Haar rotation; self-exclusion correction; two-root independence in the population limit.', [
('thm-mean-field-component-identities','Component momentum, energy, and shared covariance.'),
('thm-mean-field-one-step-consistency','One-step collision consistency for the rooted population law.')
],
'The actual collision law and its conservation, moment, and covariance data. These feed the fixed-step population map and its fluctuation equations.')

stage(3, 'Control confinement, moments, and tails', [
'Combine cloning displacement with kinetic transport to derive a Foster-Lyapunov estimate for the complete update. Choose positive weights for the component drift inequalities and evaluate the resulting rate and source. Iterate the drift to obtain finite-time and stationary moment budgets.',
'The structural landscape chapter also derives selection-driven confinement from accepted inward and outward donor fluxes. Its complete moment inequality retains the selection excess and the kinetic force-growth coefficients. It provides an explicit criterion for contracting moments without a prescribed restoring trap.',
'Use these moment bounds to localize unbounded reward estimates, control population coverage failures, obtain tightness, and justify uniform integrability. The entropy-to-tail and population-fraction bounds give a second quantitative route to controlling the mass of poorly covered or distant regions.'
], ['P_N W_p ≤ r_p W_p + B_p + a_p E_p ;   r_p < 1 → a uniform moment budget when the defect budget is controlled'],
'Foster-Lyapunov drift; positive-weight spectral criterion; Gaussian radial moments; regional selection flux; coverage and tail budgets; entropy confinement floor.', [
('thm-foster-lyapunov-main','Composes the drift estimates at the level of the actual kernel.'),
('thm-slceg-full-moment','Evaluates selection-driven moment drift, including the zero-restoring-trap regime.'),
('thm-slce-entropy-floor','Converts verified selection into entropy contraction toward an explicit confinement floor.')
], 'Population-uniform physical moment and tail controls for the chosen parameter regime. These support tightness, stationary limits, smooth reconstruction, and control of field moments.')

stage(4, 'Select the stationary or conditioned ensemble', [
'For the conservative process, use drift and common-mass control to prove Harris-type convergence to its invariant law. For the killed process, use surviving-block estimates to establish QSD convergence. Keep its survival eigenvalue in every normalized calculation.',
'A QSD is an eigenmeasure of the killed evolution. Starting from it and conditioning on a finite future survival window yields explicitly reweighted histories. The Yang-Mills chapter derives those history weights and compares relaxed recorded windows with their selected QSD window law.',
'This fixes the ensemble used for field expectations. It also gives a quantitative burn-in before recording a window of gauge or matter observables. The relaxation counter remains tied to the configured algorithmic clock until the physical time coordinate is identified.'
], ['Conservative: pi_N P = pi_N.   Killed: nu_N Q = alpha_N nu_N, with finite-window survival reweighting.'],
'Weighted Harris coupling; surviving-block minorization; eigenmeasure identities; survival normalization; total-variation window comparison.', [
('thm-convergence-conservative-harris','Conservative mixing from the declared drift and coupling inputs.'),
('thm-main-convergence','QSD convergence from a two-sided surviving-block bound.'),
('thm-ym-qsd-window-relaxation','Transfers the gas relaxation rate to a selected recorded gauge window.')
], 'A selected long-time law and its exact history-window law, together with burn-in and conditioning errors. Field reconstruction takes expectations under this specified ensemble.')

stage(5, 'Derive the actual fixed-step mean field', [
'Pass from the swarm empirical measure to a nonlinear population law on marked one-slot states. Keep the alive restriction and its normalized donor law distinct from the full population law, because dead positions and velocities affect revival and collision behavior.',
'The construction first samples the measurement mark, then applies population standardization and fitness, then samples accepted edges and the limiting collision neighborhood. It finally composes the actual kinetic stages and terminal classification. Averaging the measurement before its nonlinear acceptance step would change the map.',
'The resulting population equation is discrete at the fixed configured timestep. Derive its mass balance, weak field balances, positive alive mass, and finite-horizon moments. Stationary solutions are fixed points of this map. A continuous-time generator requires its own local consistency calculation if h is later changed.'
], ['mu_(n+1) = T_h(mu_n) ;   rho_mu = mu_alive / mu(a=1)'],
'Normalized companion integrals; measurement marks; finite limiting collision neighborhoods; exact composition of BAOAB, diffusion, cap, and classification.', [
('thm-mean-field-equation','The exact fixed-step mean-field equation.'),
('thm-mass-conservation','Exact mass and weak field balances for the full population update.')
],
'The nonlinear population update, its sampled interaction law, and its exact balance relations. This is the background law for the smooth field and fluctuation constructions.')

stage(6, 'Control particles, observation time, and stationarity', [
'Prove one-step consistency of the finite update with the nonlinear population map, including measurement concentration, collision-component convergence, and boundary/revival contributions. Iterate to obtain finite-horizon propagation of chaos.',
'For long times, use the structural landscape results appropriate to the declared regime. The weighted active-contraction branch provides a unique stationary population law and a population-independent attraction rate. Its trajectory corollary supplies uniform-time approximation, stationary chaos, and commuting population/time limits with explicit initial moment control.',
'The growing-horizon branch instead supplies trajectory approximation through a diverging observation horizon using higher moments and the actual consistency constants. This provides a controlled joint population/time limit without assuming attraction. Keep the branch used by the field ensemble explicit.'
], ['Particle consistency + population stability + moment control → trajectory approximation and selected stationary identification'],
'Rooted collision convergence; one-step bias and mean-square consistency; conditional concentration; weighted TV contraction; finite-row chaos; growing-horizon error recursion.', [
('thm-chaos-finite-time-consistency','Finite-horizon propagation of chaos for the actual update.'),
('thm-slcw-active-contraction','Closed full-law population convergence with unbounded raw reward.'),
('cor-slcw-trajectories','Uniform-time trajectories, stationary chaos, and commuting limits.'),
('thm-slcgt-growing-trajectory','A separate explicit trajectory estimate through a diverging horizon.')
], 'Quantified control of the population approximation over the observation regime used in the reconstruction. This controls both finite-run correlations and the approach to the selected limiting background.')

stage(7, 'Build Poincare and logarithmic Sobolev control', [
'The confinement route begins with a quadratic Lyapunov inequality. A weighted energy estimate controls the exterior region, while a local Poincare inequality controls the interior. Together they produce a global Poincare bound. A defective LSI is then tightened using that Poincare inequality.',
'For the kinetic reference law, combine the spatial inequality with the Gaussian velocity inequality by tensorization. For joint swarm laws, Volume 2 supplies four explicit N-uniform routes: a product reference, a uniformly bounded joint tilt, uniform joint curvature, or a contractive additive-noise invariant flow.',
'Identify which route applies to the selected joint law and retain its constant C_*. Include the status-stratum entropy term when discrete alive/dead variables are part of the law. This supplies the functional inequality used for the interacting swarm and downstream reconstructed observables.'
], ['Ent_(pi_N)(f²) ≤ 2 C_* ∫ sum_i (|grad_xi f|² + |grad_vi f|²) d pi_N'],
'Lyapunov weighted energy; local-to-global Poincare; defective-LSI tightening; Bakry-Emery curvature; tensorization; bounded-density perturbation; flow contraction.', [
('thm-unconditional-lsi','LSI from a quadratic Lyapunov bound, including the global Poincare step.'),
('thm-kinetic-lsi','LSI for the kinetic reference measure.'),
('cor-n-uniform-lsi','N-uniform full-gradient LSI for the specified joint-law families.')
], 'A declared joint-law LSI with an explicit population-uniform constant. Its Poincare consequence and concentration estimates are used next; its coercivity is retained for the eventual field Hamiltonian identification.')

stage(8, 'Turn functional inequalities into relaxation and fluctuations', [
'Linearize the joint LSI at the constant function to obtain Poincare. Applied to a Lipschitz empirical average, the gradient sum is of order 1/N. The variance therefore scales as C_* L²/N, even when the walkers are dependent.',
'For evolution, combine the LSI with the kinetic Fisher-information identities and dissipation of modified entropy. Estimate the actual cloning kernel and the killing normalization in the same balance. The full normalized evolution result and the Doob-transform route yield entropy relaxation for their identified law families.',
'At discrete time, use the full-step survivor chain rule and numerical entropy-defect estimates for the actual kernel. Relative entropy controls bounded-observable bias through Pinsker, while concentration and Poincare control sampling fluctuations. Fixed marginals inherit the same LSI constant, and the inequality passes to their limits.'
], ['Var_(pi_N)(F_N) ≤ C_* L²/N ;   F_N = (1/N) sum_i phi(z_i)', 'Entropy dissipation + LSI + complete-update balance → an explicit relaxation estimate'],
'Poincare linearization; full Fisher information; modified hypocoercive entropy; conditioned entropy identity; Doob transform; Pinsker and entropy transport; exact discrete entropy accounting.', [
('cor-quantitative-lsi-final','Poincare inequality and empirical variance control.'),
('thm-villani-hypocoercivity','Kinetic hypocoercive entropy convergence.'),
('thm-kl-convergence-euclidean','Entropy convergence for the full normalized swarm evolution.'),
('cor-kl-lsi-mean-field-limit','LSI inheritance by fixed marginals and their limits.')
], 'Relaxation, concentration, and moment tools under the same selected law. These control burn-in, empirical observables, distribution-valued fluctuations, and later continuum test-function estimates.')

stage(9, 'Obtain smooth empirical and mean-field fields', [
'Differentiate the normalized companion kernels before differentiating the fitness. Volume 2 tracks numerator derivatives, normalization derivatives, moments, variance, standardized scores, and smooth fitness maps. Positive floors control denominators throughout the calculation.',
'The C3 chapter supplies force and finite-order derivative bounds. The all-order chapter extends the normalized calculus with recurrences and majorants, retaining bandwidth, regularizer, and population-count dependence. Smooth localization and density estimates connect these quantities to geometric coefficients.',
'The normalized mean-field integral theorem replaces sums by integrals with the same derivative recurrence. The smooth empirical-field convergence theorem then upgrades weak population convergence to C^n convergence on compact parameter sets for each fixed order, under its denominator and equicontinuity inputs.'
], ['F_mu(x) = [∫ a(x,y)m(x,y) mu(dy)] / [∫ a(x,y) mu(dy)] ;   F_(mu_N) → F_mu in C^n(K)'],
'Normalized-weight derivative recurrence; exact normalization cancellations; regularized variance calculus; dominated differentiation; equicontinuity; compact parameter nets; spectral bounds for reconstructed coefficients.', [
('thm-c3-regularity','Third-order regularity of the fitness construction.'),
('thm-main-cinf-regularity-fitness-potential-full','All-order regularity of the complete fitness potential.'),
('thm-cinf-mean-field-integrals','Differentiation of normalized mean-field integrals.'),
('thm-cinf-empirical-field-convergence','Smooth convergence of normalized empirical fields.')
], 'Controlled smooth coefficient fields and convergence of their derivatives. These supply the Taylor, metric, force, and local-operator regularity inputs to the continuum construction.')

stage(10, 'Reconstruct the same law on the Fractal Set', [
'Encode the complete history as a directed two-complex. CST edges record temporal transport, IG edges record selection interactions, IA edges record influence attribution, and interaction triangles carry ordered transport products. Specify the oriented incidence data used by each face.',
'Lossless reconstruction recovers trajectories, force and diffusion data, population statistics, landscape information, and cloning events from the covered record. Frame changes transport the coordinate description while retaining the invariant readouts.',
'Use the direct observable construction to pass to Gram and determinant coordinates for the internal orbits. Its measure isomorphism retains expectations and correlations of invariant observables. The recorded transition is intertwined with the original update; compressed channels retain exact eliminated-coordinate memory until their predictive completion is constructed.'
], ['Recorded observable expectation = original history expectation, under the reconstruction and measure identification'],
'Exact vector encoding; oriented two-complex; lossless decoder; invariant coordinates; orbit and measure isomorphisms; Markov intertwining; predictive completion.', [
('thm-fractal-set-lossless','Lossless reconstruction of the covered gas history.'),
('prop-fractal-set-analytic-transfer','Transports record observables and established estimates.'),
('thm-sm-direct-measure-isomorphism','Measure and observable isomorphism for invariant coordinates.'),
('thm-sm-instantiated-record-transition','Exact transition and history isomorphism for the recorded algorithm.')
], 'A field-readable history with the same native law. Gas moment, concentration, entropy, and approximation estimates can now be applied to the corresponding reconstructed observables.')

stage(11, 'Construct gauge and matter observables', [
'Choose the internal representations used for the recorded color and companion data. The direct algebra contains Hermitian contractions, alternating determinant contractions, doublet coordinates, and composite triangle observables. The orbit theorem specifies precisely which internal frame equivalence those coordinates resolve.',
'Attach oriented transports to the declared links and form ordered products around interaction triangles and closed faces. The interaction holonomy detects the mismatch between attribution and interaction transports. Transported companion doublets retain both amplitude and angular matter energy.',
'Use gauge-invariant loop traces and transported matter contractions as the physical readouts. Distinguish their native values from auxiliary scalar proxies. Gauge covariance is implemented by transporting endpoint frames, so adjacent transformations cancel around a closed loop.'
], ['U_loop = U_(e1) U_(e2) ... U_(ek) ;   W_loop = Re tr(U_loop) / dim(rep)'],
'Gram and determinant invariants; SU(n) orbit reconstruction; Hermitian and alternating doublet contractions; projector triangle observables; oriented holonomy; angular matter energy.', [
('thm-sm-direct-orbit-isomorphism','Gram and determinant coordinates for SU(n) orbits.'),
('thm-sm-direct-su2-invariants','SU(2) frame structure from doublet contractions.'),
('prop-ym-recorded-interaction-sector','Interaction holonomy and angular matter energy of the recorded connection.'),
('thm-wilson-action-gauge-invariance','Gauge invariance of Wilson observables and actions.')
], 'The concrete invariant field algebra and its native interaction observables. This algebra is the descriptor on which the history likelihood induces the effective action.')

stage(12, 'Derive the native field action and response', [
'Start from the likelihood of the complete stochastic history, including selection, masks, cloning writes, rotations, and kinetic noises. Push the history and reference laws forward through the field descriptor, and condition the likelihood on that descriptor. Its negative logarithm is the effective field action.',
'The generating functional is an expectation under that native field law. Its derivatives reproduce the recorded correlation functions. The effective predictive kernel and the completed descriptor algebra preserve the original history predictions; channel compression carries the exact memory of discarded coordinates.',
'Disintegrate the action into recorded geometry and normalized conditional gauge fibers. Retain the fiber denominator and geometry weight so their recombination recovers the native sourced law. Differentiate actual source-dependent updates to obtain native response currents and the weak connection-variation identity. The discrete Ward identity and Wilson action give the gauge equations and curvature functional.'
], ['a(y) = E_R[dP/dR | Y=y] ;   S_alg(y) = -log a(y) ;   d nu = exp(-S_alg) d lambda', 'Z(J) = E_P exp(i sum_r J_r O_r(Y))'],
'Conditional likelihood; descriptor action; predictive kernel; characteristic functional; geometry-fiber disintegration; source tangents; native response identities; graph Ward identity.', [
('thm-ym-recorded-action-emergence','Action emergence from the recorded stochastic dynamics.'),
('thm-sm-effective-recorded-gauge-dynamics','Effective gauge-channel action and predictive kernel.'),
('thm-ym-native-geometry-fiber-action','Exact action on recorded geometry fibers.'),
('thm-ym-native-source-response','Exact source response of the algorithm-derived field action.')
], 'A native field measure, generating functional, and source-response calculus whose correlations remain those of the complete gas history. These are the objects carried into the continuum hierarchy.')

stage(13, 'Construct the recorded quantum matter algebra', [
'Build the exterior algebra of recorded modes. Alternating insertions encode fermionic words with the specified graded signs and preserve independent exterior combinations as operators. The construction retains the meaning of its recorded contractions and replica correlations.',
'Transport the actual contraction into the CAR algebra. The recorded channel construction supplies a unital completely positive evolution, a two-point correspondence, and a higher-word prescription. Differentiate the transported evolution to identify its native generator.',
'For local matter observables, use regional covariance and locality-defect coefficients derived from the recorded evolution. The local-net application is made in the declared even observable sector. Together with the gauge readouts, this gives the finite operator formulation used by the connected algorithmic field theory.'
], ['Recorded modes → exterior insertion algebra → CAR operators → transported complete-update evolution'],
'Exterior Hilbert construction; faithful insertions; CAR relations; replica contractions; isometric compression and complete positivity; native generator; regional covariance.', [
('thm-lqft-oriented-word-algebra','Faithful oriented-word algebra for recorded modes.'),
('thm-lqft-record-car-channel','Unital completely positive CAR evolution from the recorded contraction.'),
('thm-ym-hk-record-instantiation','Separate local-net application to the recorded CAR representation.')
], 'A precise matter operator algebra with the recorded dynamics and word correlations. Its continuum identification proceeds alongside the native gauge correlation hierarchy.')

stage(14, 'Retain fluctuations and a common correlation hierarchy', [
'Center empirical fields about their selected population expectation and use the square-root population scaling. The joint LSI bounds their moments. Distribution-valued compactness places the fluctuation correlations on a common subsequence in the declared test-function topology.',
'Compute drift and covariance from the complete selected update. In particular, retain covariance from shared collision rotations, cloning writes, measurement marks, and kinetic innovations. These equations determine the fluctuation dynamics around the mean-field background.',
'The native gauge-hierarchy theorems carry finite words and reflected products of normalized geometric readouts and the full color invariant algebra on a common subsequence. Uniform-integrability and mark-reconstruction estimates control passage of expectations. Source-action and geometry-fiber continuum results retain the sourced history interpretation on this same construction.'
], ['Xi_N(phi) = sqrt(N) [L_N(phi) - E L_N(phi)]'],
'Joint-LSI moment bounds; test-space compactness; all-order moment passage; full-update conditional variance; common subsequences; uniform integrability; reconstructed reflected-matrix control.', [
('thm-ym-spacetime-fluctuation-compactness','Distribution-valued compactness from the established stationary LSI.'),
('prop-ym-complete-fluctuation-equations','Drift and covariance equations for the complete selected update.'),
('thm-ym-native-physical-gauge-hierarchy','Common native physical hierarchy of full recorded color invariants.'),
('thm-ym-native-fiber-continuum','Geometry-fiber action in the common native continuum limit.')
], 'A distributional fluctuation theory and a common native gauge correlation hierarchy. The next stage controls the spatial reconstruction scales and differential operators used to interpret that hierarchy.')

stage(15, 'Control the continuum operator with derived tools', [
'Identify the reconstructed product geometry and its sampling measure q dvol_g. Smoothed coefficient bounds and spectral bounds supply geometric regularity; the geometry lemma gives a sufficient global-hyperbolicity condition. Normalize sampling using the selected history law and density weights.',
'Verify the kernel moments in local normal coordinates. The local Taylor estimate bounds the operator bias by C_b epsilon². The covariance estimate controls the actual local summands, including their density correction, and gives variance at most C_mix C_v /(M epsilon^(D+2)).',
'Choose epsilon_M = ell_0 M^(-alpha), with 0 < alpha < 1/(D+4), as in the chapter. Both squared bias and variance vanish. The balanced exponent 1/(D+6) gives the stated mean-square rate. Finally compare the reconstructed episode operator with this analyzed estimator, controlling metric, density, neighborhood, and kernel errors after normalization.'
], ['E|L_hat f - Box_g f|² ≤ C_b² epsilon⁴ + C_mix C_v / (M epsilon^(D+2))', 'epsilon_M = ell_0 M^(-alpha) ;   alpha = 1/(D+6) balances the bounds: MSE = O(M^(-4/(D+6)))'],
'Smooth metric coefficients; geometric identification; importance weights; normal-coordinate kernel moments; Taylor bias; shrinking-neighborhood covariance; bandwidth schedule; normalized reconstruction comparison.', [
('lem-continuum-local-bias','Direct local bias estimate in specified normal coordinates.'),
('lem-continuum-a4-mixing','Covariance condition sufficient for the estimator variance bound.'),
('lem-continuum-a6-scaling','Explicit sufficient bandwidth schedule.'),
('cor-continuum-consistency-conditional','Transfers estimator consistency to the reconstructed operator and action under its listed inputs.')
], 'A controlled continuum differential operator. The bias, covariance, and reconstruction estimates also supply the local integration bounds needed for an action limit.')

stage(16, 'Pass actions, sources, and correlations through the limits', [
'Use geometric quadrature and the recorded face-action curvature expansion to obtain Wilson action consistency. Keep face orientation, matrix-valued curvature, face weights, and coupling normalization. The classical continuum term is the quadratic Yang-Mills curvature energy.',
'Combine local operator convergence with weighted quadrature to pass scalar actions. For actions evaluated on the same interacting records as the operator, carry the joint weighted error estimate. The native source-action and geometry-fiber results carry the selected likelihood and its sourced correlations into the common continuum hierarchy.',
'Assemble the quantitative errors: particle approximation, relaxation, survival conditioning, temporal consistency, local bias, correlated sampling, and record reconstruction. The time-consistency theorem propagates a local kernel defect using stability. QSD perturbation uses conditioned-block comparison with survival normalization. State a joint scale schedule that makes these errors vanish while retaining the required constants.'
], ['Small faces: U_face = I + i a² F + higher-order terms', 'S_W → (1/(4g²)) ∫ F^a_(mu nu) F^a_(mu nu) d⁴x, with the stated geometric and normalization inputs'],
'Curvature remainder; geometric quadrature; same-sample action error; BAOAB weak consistency; split-system commutator terms; invariant-law perturbation; conditioned-block QSD stability; total observable error.', [
('thm-continuum-limit-ym','Wilson action consistency under geometric quadrature.'),
('thm-ym-continuum-source-action','Common continuum limit of the native source likelihood and descriptor action.'),
('thm-correct-continuum-limit','Time consistency for the specified transition family.'),
('thm-total-error-bound','Total observable error: population bias, sampling variance, time defect, and mixing error.')
], 'A continuum action and correlation hierarchy with an explicit error budget. The field-law, physical symmetry, and Hamiltonian identifications are completed on this limiting object.')

stage(17, 'Complete the physical quantum-field reconstruction', [
'Use the moment and test-space bounds to establish Euclidean regularity of the limiting correlations. Bosonic symmetry follows from commuting fields; the matter algebra supplies the graded rule. The local-net results provide isotony and the locality statement for their specified representations.',
'The completion target is the physical four-dimensional hierarchy: establish Euclidean covariance and physical translation symmetry, reflection positivity on every finite future-supported combination, clustering, and a nontrivial limiting field sector. The physical reflected-matrix calculations in the chapter are the concrete tests to be completed for the chosen family.',
'Resolve the source-identified anchor and normalization issues in that physical family, retaining the native law and physical coordinates. Apply the selected quantum reconstruction theorem to obtain a Hilbert space, vacuum, positive self-adjoint physical Hamiltonian, and spacetime fields. Identify the limiting non-Abelian gauge dynamics with the Yang-Mills target and establish locality and the spectrum condition in this physical representation.'
], ['Reflection form: (F,G)_OS = E[conjugate(Theta F) G] ;   (F,F)_OS ≥ 0 for every F in the future algebra'],
'Tempered correlation bounds; bosonic/graded symmetry; native reflected matrices; physical translation identities; local algebras; vacuum reconstruction; transfer Hamiltonian and physical clock identification.', [
('thm-os-os0-fg','Euclidean regularity from actual moment bounds.'),
('thm-os-os4-fg','Symmetry of bosonic correlations.'),
('prop-ym-native-color-reflected-matrices','Physical reflection calculation on the full native color algebra.'),
('thm-ym-algorithmic-qft-synthesis','Connected algorithmic field theory from recorded evolution and native bounds.')
], 'Completion target: a nontrivial physical Yang-Mills QFT with an identified self-adjoint transfer Hamiltonian. This is the precise operator to which the final spectral-gap passage is applied.')

stage(18, 'Preserve the physical Yang-Mills mass gap', [
'Carry the functional-inequality and evolution constants along the selected cutoff and volume family. Poincare supplies coercivity of its associated form; hypocoercivity supplies relaxation for the kinetic evolution. Complete the identification that connects these estimates to the reconstructed physical gauge transfer Hamiltonian.',
'For that self-adjoint Hamiltonian family H_a, prove a common positive gap lambda_* above the normalized vacuum. Construct isometric embeddings J_a into the limiting Hilbert space, show convergence of the vacuum vectors, and establish strong convergence of the embedded semigroups.',
'The gap-survival theorem then passes the vacuum and spectral bound to H. Restriction to an invariant gauge sector preserves the gap under the chapter\'s operator hypotheses. Fix the physical action and time units so the limiting energy scale remains positive. The endpoint theorem combines the physical reconstruction of Stage 17 with this gap passage.'
], ['||exp(-t H_a)(I-P_(Omega_a))|| ≤ exp(-lambda_* t), uniformly in the selected cutoffs and volumes', 'J_a exp(-t H_a) J_a* → exp(-t H) strongly ;   spec(H) is contained in {0} union [lambda_*, infinity)', 'Energy gap ≥ hbar_eff lambda_* ;   mass gap = energy gap / c², with fixed physical units'],
'Same-law coercivity; physical transfer identification; self-adjoint semigroup estimate; compatible Hilbert-space embeddings; vacuum convergence; strong semigroup convergence; invariant-sector restriction.', [
('thm-mass-gap-rg-fixed-point','Survival of a uniformly controlled self-adjoint gap.'),
('thm-gauge-sector-mass-gap','Restriction of a gap to an invariant gauge sector.')
], 'Completion target: a nontrivial four-dimensional Yang-Mills quantum field theory with a unique vacuum and a strictly positive physical mass gap. Its construction retains the original history law through the recorded and continuum identifications.')

story.append(PageBreak())
p('The continuum-control ledger', 'Section')
p('The proof closes by assembling estimates for the same observable and law. The following ledger identifies the purpose of each derived tool and the quantity that must remain controlled along the selected limit.')
ledger=[
('Tool', 'What it controls', 'Quantity retained'),
('Moment / tail drift', 'Escape, localization, tightness', 'Moment order, drift rate, defects'),
('Poincare', 'Variance and form coercivity', 'Joint-law constant C_*'),
('LSI', 'Entropy, concentration, moments', 'Full-gradient and status terms'),
('Hypocoercivity', 'Kinetic relaxation', 'Modified norm, rate, prefactor'),
('Chaos / trajectory estimates', 'Finite-particle approximation', 'Initialization and time horizon'),
('Normalized C^n calculus', 'Field and derivative convergence', 'Denominators, widths, floors'),
('Kernel moments / Taylor bias', 'Differential-operator identification', 'Geometry and C_b'),
('Covariance bound', 'Correlated local sampling', 'C_mix / (M epsilon^(D+2))'),
('Reconstruction comparison', 'Recorded operator error', 'Error after kernel normalization'),
('Weak / stationary / QSD errors', 'Temporal and ensemble approximation', 'Stability and survival weights'),
('Native source / fiber identities', 'Action and sourced correlations', 'Common hierarchy and law'),
('Strong semigroup convergence', 'Physical spectral-gap passage', 'Embeddings, vacuum, lambda_*'),
]
t=Table([[Paragraph(escape(c),styles['SmallCustom']) for c in row] for row in ledger],colWidths=[130,180,180],repeatRows=1,hAlign='LEFT')
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFEDF0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D1DEE3')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
story.append(t)
sub('One assembled endpoint')
p('The mean field determines the controlled population background. The native history law determines interactions and correlations. Smooth reconstruction and local sampling estimates carry those correlations to continuum fields. Physical quantum reconstruction produces the Hamiltonian. Uniform gap control and strong semigroup convergence preserve its positive gap.')

story.append(PageBreak())
p('Principal theorem register', 'Section')
p('Titles and labels below are copied from the current Volume 2 source. Locations are relative to docs/source/2_fractal_gas/. These are navigation references to the full statements and proofs; the roadmap does not replace their hypotheses.', 'Body')
for i,label in enumerate(used,1):
    item=catalog[label]
    block=[Paragraph(f'<b>{i:02d}. {escape(item["title"])}</b>',styles['SmallCustom']),Paragraph(escape(label)+'<br/>'+escape(item['file'])+f' : line {item["line"]}',styles['SmallCustom']),Spacer(1,5)]
    story.append(KeepTogether(block))

def footer(canvas, doc):
    canvas.saveState()
    w,h=doc.pagesize
    canvas.setStrokeColor(colors.HexColor('#CADBE1'));canvas.line(52,43,w-52,43)
    canvas.setFont('Lato',8);canvas.setFillColor(colors.HexColor('#627782'))
    canvas.drawString(52,29,'FRAGILE / VOLUME 2 / QFT AND YANG-MILLS PROOF PROGRAM')
    canvas.drawRightString(w-52,29,str(doc.page))
    if doc.page>1:
        canvas.setFont('Lato',7.8);canvas.drawString(52,h-30,'EUCLIDEAN GAS  /  ANALYTIC CONTROL  /  QUANTUM RECONSTRUCTION')
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=52,rightMargin=52,topMargin=55,bottomMargin=60,title='Volume 2: Euclidean Gas to Quantum Yang-Mills and a Mass Gap',author='Compiled from Fragile Volume 2',subject='Source-grounded proof roadmap and continuum-control theorem register')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(str(OUT))
print(f'{len(used)} verified theorem references')
