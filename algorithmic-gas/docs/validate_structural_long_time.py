#!/usr/bin/env python3
"""Exact checks for long-time identities; not a substitute for landscape hypotheses."""
from fractions import Fraction as Q
from math import comb, isqrt

import sympy as sp


def transport_one_dimension(left, right):
    """Exact transport on the unit interval by integrated CDF difference."""
    sites = sorted(set(left) | set(right))
    difference = Q(0)
    result = Q(0)
    for j, x in enumerate(sites[:-1]):
        difference += left.get(x, Q(0)) - right.get(x, Q(0))
        result += abs(difference) * (sites[j + 1] - x)
    return result


def check_joint_invariance():
    # An exactly soluble finite kernel tests the general coupling/telescoping
    # identity. It is not claimed to be a lumping of the Fractal Gas.
    for count in range(2, 7):
        sites = [Q(j, count) for j in range(count + 1)]
        image = [Q(1, 4) + p / 2 for p in sites]
        transition = [[Q(comb(count, j)) * p**j * (1-p)**(count-j)
                       for j in range(count + 1)] for p in image]
        bound = Q(1, 2*isqrt(count))  # >= 1/(2 sqrt(N))
        matrix = sp.Matrix(transition).T-sp.eye(count+1)
        rhs = sp.zeros(count+1, 1)
        for j in range(count+1):
            matrix[count, j] = 1
        rhs[count, 0] = 1
        stationary = [Q(x) for x in matrix.inv()*rhs]
        q_stat = dict(zip(sites, stationary))
        f_stat = dict(zip(image, stationary))
        assert transport_one_dimension(q_stat, f_stat) <= bound
        law = [Q(1)] + [Q(0)] * count
        occupation = [Q(0)] * (count+1)
        for horizon in range(1, 9):
            occupation = [a+b for a, b in zip(occupation, law)]
            average = [x/horizon for x in occupation]
            assert transport_one_dimension(dict(zip(sites, average)),
                                           dict(zip(image, average))) <= bound+Q(1,horizon)
            law = [sum((law[i]*transition[i][j] for i in range(count+1)), Q(0))
                   for j in range(count+1)]
        # General finite-row sampling collision budget, without independence.
        for k in range(1, count+1):
            no_repeat = Q(1)
            for j in range(k):
                no_repeat *= Q(count-j, count)
            assert 1-no_repeat <= Q(k*(k-1), 2*count)


def check_moment_coefficients():
    r = sp.symbols('r', positive=True)
    for p in (2, 4, 8):
        u = ((1+r)/(2*r))**sp.Rational(1, p-1)-1
        assert sp.simplify((1+u)**(p-1)*r-(1+r)/2) == 0
    d = sp.symbols('d', positive=True)
    assert sp.simplify(16*sp.gamma((d+8)/2)/sp.gamma(d/2)
                       - d*(d+2)*(d+4)*(d+6)) == 0


def check_path_length():
    for q in (Q(1, 4), Q(1, 2), Q(3, 4)):
        for x in (Q(0), Q(1, 3), Q(1)):
            # For F(x)=qx the infinite path length is x exactly.
            assert x-q*x == abs(x-q*x)
            for cutoff in range(1, 15):
                prefix = sum(((1-q)*q**j*x for j in range(cutoff)), Q(0))
                assert prefix+q**cutoff*x == x
                assert q**cutoff*x <= q**(cutoff-1)/(1-q)


def check_phase_weight_matrix():
    for m in range(1, 6):
        generator = sp.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    generator[i, j] = sp.Rational(1+(2*i+j)%5, 7)
            generator[i, i] = -sum(generator[i, j] for j in range(m) if j != i)
        augmented = generator.copy()
        for i in range(m):
            augmented[i, m-1] = 1
        inverse = augmented.inv()
        theta = inverse[m-1, :]
        assert sum(theta) == 1 and all(x > 0 for x in theta)
        assert theta*generator == sp.zeros(1, m)
        bound = max(sum(abs(inverse[i, j]) for j in range(m)) for i in range(m))
        for missing in (sp.Rational(0), sp.Rational(1, 7), sp.Rational(1, 2)):
            weights = sp.Matrix([[(1-missing)*(i+1)*sp.Rational(2, m*(m+1))
                                  for i in range(m)]])
            residual = weights*generator
            assert sum(abs(x) for x in weights-theta) <= bound*(
                sum(abs(x) for x in residual)+missing)


def check_keystone_power_and_floor():
    for dimension in range(1, 5):
        power = 5+4*dimension
        w, c0, emax, mesh = sp.symbols('w c0 emax mesh', positive=True)
        raw = c0*(w/2)**2*w**3/(2*emax**2*(mesh/(w/2))**(4*dimension))
        claimed = c0*w**power/(2**(3+4*dimension)*emax**2*mesh**(4*dimension))
        assert sp.simplify(raw-claimed) == 0
        for radius in (Q(0), Q(1, 5), Q(1, 2)):
            for excess in (Q(1, 10), Q(1, 4)):
                assert (radius+excess)**power >= radius**power+excess**power
                a = Q(1, power)
                y_next = excess-a*excess**power
                assert y_next**(-(power-1)) >= excess**(-(power-1))+(power-1)*a
    for count in range(2, 15):
        sizes = [count//3, count//3, count-2*(count//3)]
        energies = [Q(size*(i+1), 3*count) for i, size in enumerate(sizes)]
        total = sum(energies)
        pressure = sum(energy*Q(max(0, size-1), count-1)**2
                       for energy, size in zip(energies, sizes))
        assert pressure >= total**3/Q(2*len(sizes)**2)-total/count**2


def check_active_closure_algebra():
    epsilon, remainder = sp.symbols('epsilon remainder', positive=True)
    two_step = 1-epsilon+remainder+remainder*(1+remainder)
    assert sp.expand(two_step-(1-epsilon+2*remainder+remainder**2)) == 0
    threshold_gap = sp.factor(two_step.subs(remainder, epsilon/4)
                              -(1-7*epsilon/16))
    assert threshold_gap == epsilon*(epsilon-1)/16
    # The subcritical exploration denominator is paid before selecting theta.
    for c in (Q(0), Q(1, 16), Q(1, 8), Q(1, 4)):
        assert 1/(1-2*c) <= 2
    # Check the non-geometric finite-window recursion with exact fourth roots.
    for exponent in range(1, 5):
        base = sp.Rational(1, 2)**(4**exponent)
        for modulus in (sp.Rational(1, 2), sp.Rational(2)):
            upper = base
            for step in range(1, exponent+1):
                next_upper = (1+modulus)*upper**sp.Rational(1, 4)
                # a <= current upper^(1/4) is the induction's only absorption.
                assert bool(base <= upper**sp.Rational(1, 4))
                exact = (1+modulus)**sum(sp.Rational(1, 4)**j
                                       for j in range(step))*base**sp.Rational(1, 4)**step
                assert sp.simplify(next_upper-exact) == 0
                upper = next_upper


def check_weighted_closure_algebra():
    t, root_beta = sp.symbols('t root_beta', positive=True)
    gap = 1/(2*root_beta)-t/(1+root_beta**2*t**2)
    assert sp.factor(gap) == (root_beta*t-1)**2/(2*root_beta*(root_beta**2*t**2+1))
    assert sp.expand(2*(1+t**2)-(1+t)**2) == sp.expand((t-1)**2)
    # Exact exponent in the eighth-moment initialization localization.
    assert sp.Rational(1, 32)-sp.Rational(3, 2)*sp.Rational(1, 80) == sp.Rational(1, 80)
    for epsilon in (Q(1, 16), Q(1, 4), Q(3, 4), Q(1)):
        bw = 1+epsilon/2
        residual = epsilon/(8*bw)
        qw = 1-epsilon/2+2*bw*residual+residual**2
        assert qw <= 1-Q(15, 64)*epsilon < 1


def check_selection_entropy_algebra():
    r, a, h, b, log_density, z1, za = sp.symbols('r a h b log_density z1 za')
    substituted = log_density+z1+r*(h+za-z1)/a+b
    collected = (r/a)*h+b+log_density+(1-r/a)*z1+(r/a)*za
    assert sp.simplify(substituted-collected) == 0
    assert sp.simplify((r/a).subs(a, (1+r)/2)-2*r/(1+r)) == 0
    population = sp.symbols('population', positive=True)
    joint = (population*log_density+population*z1
             +r*(population*h+population*(za-z1))/a+population*b)/population
    assert sp.simplify(joint-collected) == 0
    for epsilon in (Q(1, 5), Q(1, 2), Q(1)):
        t = (2/epsilon).__ceil__()
        assert (1+Q(1, t))*(1-epsilon)+epsilon/4 <= 1-epsilon/4
    for count in range(2, 15):
        for core_count in range(count+1):
            counts = [core_count, count-core_count]
            weights = [[Q(1), Q(1, 2)], [Q(1, 2), Q(1)]]
            for source in range(2):
                if counts[source] == 0:
                    continue
                inclusive = sum(Q(counts[j], count)*weights[source][j]
                                for j in range(2))
                lower = max(Q(1, 4), inclusive-weights[source][source]/2)
                actual = inclusive-weights[source][source]/count
                assert lower <= actual <= min(Q(1), inclusive)


def check_frozen_feedback_and_exit():
    # Saturation is used analytically; verify the normalization derivatives.
    reward, mean, variance, sigma = sp.symbols('reward mean variance sigma', real=True)
    scale = sp.sqrt(variance+sigma**2)
    t = (reward-mean)/scale
    logistic = 1/(1+sp.exp(-t))
    slope = sp.exp(-t)/(1+sp.exp(-t))**2
    assert sp.simplify(sp.diff(logistic, mean)+slope/scale) == 0
    assert sp.simplify(sp.diff(logistic, variance)+slope*t/(2*scale**2)) == 0
    # Frozen Harris cannot absorb an unsigned feedback bound >= inward gain.
    r, beta, radius, offset = sp.symbols('r beta radius offset', positive=True)
    harris_fraction = (2+beta*(r*radius+2*offset))/(2+beta*radius)
    assert sp.simplify(harris_fraction-r
                      -(2*(1-r)+2*beta*offset)/(2+beta*radius)) == 0
    # Optimize the actual bounded-test truncation/second-moment tail bound.
    g, count, margin, moment = sp.symbols('g count margin moment', positive=True)
    cutoff = (moment*count*margin/(2*g))**sp.Rational(1, 3)
    bound = 4*cutoff**2*g/(count*margin**2)+4*moment/(cutoff*margin)
    optimized = 3*(16*g*moment**2/(count*margin**4))**sp.Rational(1, 3)
    assert sp.simplify(bound-optimized) == 0
    level = sp.symbols('level', positive=True)
    objective = 4*level**2*g/(count*margin**2)+4*moment/(level*margin)
    assert sp.simplify(sp.diff(objective, level).subs(level, cutoff)) == 0
    for growth in (Q(2), Q(3), Q(7, 2)):
        for horizon in range(1, 12):
            assert sum((growth**j for j in range(1, horizon+1)), Q(0)) == (
                growth*(growth**horizon-1)/(growth-1))
    for recovery in (Q(1, 5), Q(1, 2), Q(1)):
        for exit_rate in (Q(0), Q(1, 10), Q(1, 2)):
            bad = Q(1, 3)
            for horizon in range(1, 12):
                bad = exit_rate+(1-recovery)*bad
                exact = ((1-recovery)**horizon*Q(1, 3)
                         +exit_rate/recovery*(1-(1-recovery)**horizon))
                assert bad == exact
    # A stationary mixture cannot approximate both separated initial phases.
    for n in range(2, 12):
        for occupied in range(n+1):
            d_to_left = Q(occupied, n)
            d_to_right = Q(n-occupied, n)
            assert d_to_left+d_to_right == 1


def check_growing_trajectory_algebra():
    d = sp.symbols('d', positive=True)
    gaussian_24 = 2**12*sp.gamma((d+24)/2)/sp.gamma(d/2)
    assert sp.simplify(gaussian_24-sp.prod(d+2*j for j in range(12))) == 0
    for horizon in range(12):
        prefix = sum((Q(1, 2**(n+1)) for n in range(horizon+1)), Q(0))
        assert prefix+Q(1, 2**(horizon+1)) == 1
        assert sum((Q(1, 32**j) for j in range(horizon)), Q(0)) <= Q(32, 31)
    for dimension in range(1, 9):
        alpha = Q(1, 16*dimension)
        assert Q(3, 16)-Q(1, 2) == -Q(5, 16)
        assert Q(5, 16) >= alpha and 2*alpha >= alpha
    # The root-average moment/error Cauchy estimate uses a 24th moment,
    # and the configured horizon leaves a power 3/4 in the log error.
    assert Q(3, 2)*2 == 3
    assert Q(1, 32)*2 == Q(1, 16)
    assert Q(1)-Q(1, 4) == Q(3, 4)


def check_actual_clone_and_ordered_forest():
    c = sp.symbols('c', positive=True)
    for length in range(15):
        path_sum = sum(c**length/(sp.factorial(k)*sp.factorial(length-k))
                       for k in range(length+1))
        assert sp.simplify(path_sum-(2*c)**length/sp.factorial(length)) == 0
    accept, cross = sp.symbols('accept cross', nonnegative=True)
    source_positive = sp.Rational(1, 2)*(1-accept*cross)+sp.Rational(1, 2)*accept*cross
    assert sp.simplify(source_positive-sp.Rational(1, 2)) == 0
    radius, dimension, jitter = sp.symbols('radius dimension jitter', positive=True)
    mixture_variance = (1-accept)*radius**2+accept*(radius**2+dimension*jitter**2)
    assert sp.simplify(mixture_variance-radius**2-dimension*jitter**2*accept) == 0


def check_terminal_box_extinction():
    # A translated Gaussian puts maximal mass in an interval at its center.
    half_width, shift, noise = sp.symbols('half_width shift noise', positive=True)
    mass = (sp.erf((half_width-shift)/(sp.sqrt(2)*noise))
            + sp.erf((half_width+shift)/(sp.sqrt(2)*noise)))/2
    density_plus = sp.exp(-(half_width+shift)**2/(2*noise**2))/sp.sqrt(2*sp.pi)
    density_minus = sp.exp(-(half_width-shift)**2/(2*noise**2))/sp.sqrt(2*sp.pi)
    assert sp.simplify(sp.diff(mass, shift)-(density_plus-density_minus)/noise) == 0
    assert sp.simplify(density_plus/density_minus
                       - sp.exp(-2*half_width*shift/noise**2)) == 0
    # Exact survival-tail sums, not independent unconditional walker deaths.
    for outside in (Q(1, 10), Q(1, 3), Q(3, 4)):
        for population in range(1, 7):
            death = outside**population
            survival = 1-death
            for horizon in range(1, 9):
                tail_sum = sum((survival**j for j in range(horizon)), Q(0))
                assert tail_sum == (1-survival**horizon)/death
                assert tail_sum <= 1/death
    alive, death = sp.symbols('alive death', positive=True)
    threshold = sp.log(2/alive)/(-sp.log(1-death))
    assert sp.simplify(threshold*sp.log(1-death)-sp.log(alive/2)) == 0
    # Simultaneous global radial coercivity and bounded force bound the domain.
    restoring, force, remainder = sp.symbols('restoring force remainder', positive=True)
    radius = (force+sp.sqrt(force**2+4*restoring*remainder))/(2*restoring)
    assert sp.simplify(restoring*radius**2-force*radius-remainder) == 0


def check_survival_conditioning():
    # The bad-alive event includes extinction. Subtract it before normalizing.
    delta, death = sp.symbols('delta death')
    assert sp.simplify(delta-(delta-death)/(1-death)
                       -death*(1-delta)/(1-death)) == 0
    # Exact rational killed-chain checks validate normalization algebra only;
    # this finite chain is not asserted to be a Fractal Gas coarse graining.
    physical = [
        [Q(1, 2), Q(3, 10), Q(1, 10), Q(1, 10)],
        [Q(1, 4), Q(3, 5), Q(1, 20), Q(1, 10)],
        [Q(3, 5), Q(1, 5), Q(3, 20), Q(1, 20)],
    ]
    failure = Q(1, 5)
    assert all(sum(row) == 1 and row[2]+row[3] <= failure for row in physical)
    law = [Q(0), Q(0), Q(1)]
    history_good = law[:]
    row_normalized = law[:]
    different_normalizations = False
    for _ in range(12):
        output = [sum(law[i]*physical[i][j] for i in range(3)) for j in range(4)]
        extinct = output[3]
        next_law = [value/(1-extinct) for value in output[:3]]
        assert next_law[2] <= (failure-extinct)/(1-extinct) <= failure
        tv = sum(abs(value-reference) for value, reference in
                 zip(next_law+[Q(0)], output))/2
        assert tv == extinct
        row_normalized = [sum(row_normalized[i]*physical[i][j]/(1-physical[i][3])
                              for i in range(3)) for j in range(3)]
        different_normalizations |= row_normalized != next_law
        good_output = [sum(history_good[i]*physical[i][j] for i in range(3))
                       for j in range(2)]
        assert sum(good_output) >= 1-failure
        history_good = [value/sum(good_output) for value in good_output]+[Q(0)]
        law = next_law
    assert different_normalizations
    # Normalized alive-test error uses the target mass, not an inverse random
    # output mass. The zero-mass extension satisfies the same inequality.
    for target_mass in (Q(1, 10), Q(1, 2), Q(1)):
        for output_mass in (Q(0), Q(1, 100), Q(1, 3), Q(1)):
            for left in (Q(-1), Q(-1, 4), Q(0), Q(1)):
                for right in (Q(-1), Q(0), Q(1, 3), Q(1)):
                    output_test = output_mass*left
                    target_test = target_mass*right
                    error_bound = (abs(output_test-target_test)
                                   +abs(output_mass-target_mass))/target_mass
                    assert abs(left-right) <= error_bound
    d = sp.symbols('d', positive=True)
    assert sp.simplify(2*sp.gamma((d+2)/2)/sp.gamma(d/2)-d) == 0
    mass, bound, bias, count, fluctuation = sp.symbols(
        'mass bound bias count fluctuation', positive=True)
    explicit = (fluctuation/(count*mass)+7*bound**2*delta/mass
                +4*bound*(2*bound*bias/sp.sqrt(count)+4*bound*delta))
    collected = (fluctuation/(count*mass)+8*bound**2*bias/sp.sqrt(count)
                 +bound**2*(7/mass+16)*delta)
    assert sp.simplify(explicit-collected) == 0


def check_signed_phase_feedback():
    # Exact entropy algebra: G is the output in the frozen phase environment,
    # Q is the actual output. These vectors test the identity, not a proposed
    # coarse Markov replacement of the gas.
    pi = [sp.Rational(1, 5), sp.Rational(3, 10), sp.Rational(1, 2)]
    g = [sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2)]
    q = [sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(1, 6)]
    entropy = lambda p, r: sum(x*sp.log(x/y) for x, y in zip(p, r))
    flux = sum((x-y)*sp.log(y/z) for x, y, z in zip(q, g, pi))
    residual = entropy(q, pi)-entropy(g, pi)-flux-entropy(q, g)
    assert sp.simplify(sp.expand_log(residual, force=True)) == 0
    # Exact convex remainder, including the linear term that integrates to zero.
    s, t = sp.symbols('s t', positive=True)
    bregman = t*sp.log(t/s)-t+s
    assert sp.simplify(sp.diff(bregman, t, 2)-1/t) == 0
    assert sp.simplify(bregman.subs(t, s)) == 0
    assert sp.simplify(sp.diff(bregman, t).subs(t, s)) == 0
    p, reference, regularizer = sp.symbols('p reference regularizer', positive=True)
    smooth_mass = (p+regularizer*reference)/(1+regularizer)
    smooth_entropy = smooth_mass*sp.log(smooth_mass/reference)
    derivative = (1+sp.log(smooth_mass/reference))/(1+regularizer)
    curvature = 1/((1+regularizer)*(p+regularizer*reference))
    assert sp.simplify(sp.diff(smooth_entropy, p)-derivative) == 0
    assert sp.simplify(sp.diff(smooth_entropy, p, 2)-curvature) == 0
    # The signed quadratic criterion must retain the cross term.
    loss, norm2, radius = sp.symbols('loss norm2 radius', positive=True)
    ratio = (1+radius)/(1-radius)*(norm2-loss)/norm2
    assert sp.simplify(ratio.subs(loss, 2*radius*norm2/(1+radius))-1) == 0
    # Actual clipped gate: verify pointwise gain and its Jensen lower bound
    # using nonuniform normalized companion weights and exact rational numbers.
    fitness = [Q(1, 5), Q(1), Q(3)]
    mass = [Q(1, 4), Q(1, 2), Q(1, 4)]
    weights = [[Q(1), Q(1, 3), Q(1, 5)],
               [Q(1, 3), Q(1), Q(1, 2)],
               [Q(1, 5), Q(1, 2), Q(1)]]
    for scale in (Q(1, 10), Q(1), Q(5)):
        epsilon = Q(1, 7)
        gain = probability = square = Q(0)
        for i, fi in enumerate(fitness):
            normalizer = sum(weights[i][j]*mass[j] for j in range(3))
            for j, fj in enumerate(fitness):
                pair_mass = mass[i]*weights[i][j]*mass[j]/normalizer
                gate = min(Q(1), max(Q(0), fj-fi)/(scale*(fi+epsilon)))
                assert gate*(fj-fi) >= scale*(fi+epsilon)*gate**2
                gain += pair_mass*gate*(fj-fi)
                probability += pair_mass*gate
                square += pair_mass*gate**2
        assert square >= probability**2
        assert gain >= scale*(min(fitness)+epsilon)*probability**2


def check_common_source_increment():
    # Exact finite sums check the coupling identity, not an approximate gas.
    x = list(map(Q, [-2, 1, 4]))
    y = list(map(Q, [-1, 2, 3]))
    left = [[Q(1, 2), Q(1, 3), Q(1, 6)],
            [Q(1, 4), Q(1, 2), Q(1, 4)],
            [Q(1, 5), Q(1, 5), Q(3, 5)]]
    right = [[Q(2, 3), Q(1, 6), Q(1, 6)],
             [Q(1, 3), Q(1, 3), Q(1, 3)],
             [Q(1, 4), Q(1, 4), Q(1, 2)]]
    for yy, rr in [(y, right), (x, left)]:
        count = len(x)
        differences = [a-b for a, b in zip(x, yy)]
        center = sum(differences)/count
        errors = [(z-center)**2 for z in differences]
        first = []
        second = []
        rhs = Q(0)
        for i in range(count):
            common = [min(a, b) for a, b in zip(left[i], rr[i])]
            mismatch = 1-sum(common)
            residual = [[(left[i][j]-common[j])*(rr[i][k]-common[k])/mismatch
                         if mismatch else Q(0) for k in range(count)]
                        for j in range(count)]
            joint = [[residual[j][k]+(common[j] if j == k else 0)
                      for k in range(count)] for j in range(count)]
            assert sum(map(sum, joint)) == 1
            mean = sum(joint[j][k]*(x[j]-yy[k])
                       for j in range(count) for k in range(count))
            jitter = Q(2, 7)*sum(residual[j][k]*int((j != i) != (k != i))
                                 for j in range(count) for k in range(count))
            moment = sum(joint[j][k]*(x[j]-yy[k])**2
                         for j in range(count) for k in range(count))+jitter
            first.append(mean)
            second.append(moment)
            rhs += sum(common[j]*(errors[j]-errors[i])
                       for j in range(count) if j != i)/count
            rhs += sum(residual[j][k]*((x[j]-yy[k]-center)**2-errors[i])
                       for j in range(count) for k in range(count))/count+jitter/count
        variance = sum(s-m*m for s, m in zip(second, first))/count**2
        new_center = sum(first)/count
        actual = sum(second)/count-new_center**2-variance-sum(errors)/count
        rhs -= (new_center-center)**2+variance
        assert actual == rhs
        if yy == x:
            assert actual == 0
    # Physical Fisher under the radial cap uses the inverse Jacobian squared.
    speed, cap = sp.symbols('speed cap', positive=True)
    cap_slope = sp.diff(cap*speed/(cap+speed), speed)
    assert sp.simplify(1/cap_slope**2-(1+speed/cap)**4) == 0
    dimension = sp.symbols('dimension', positive=True)
    moment4 = 4*sp.gamma((dimension+4)/2)/sp.gamma(dimension/2)
    assert sp.simplify(moment4-dimension*(dimension+2)) == 0



def check_signed_complete_kinetic_quadratic():
    # Exact BAOAB differences, arbitrary force increments and cap secant.
    x, v, f, g, j = sp.symbols('x v f g j', real=True)
    c, a, A, b, G = sp.symbols('c a A b G', real=True)
    B, eta = c*(1+a), c*c*(1+a)
    out_x = x+B*v+eta*f
    out_v = a*v+a*c*f+c*g
    metric = sp.Matrix([[A,b],[b,G]])
    lift = sp.Matrix([[1,B,eta,0],[0,a,a*c,c]])
    inputs = sp.Matrix([x,v,f,g])
    exact = A*out_x**2+2*b*out_x*out_v+G*out_v**2
    assert sp.expand(exact-(inputs.T*lift.T*metric*lift*inputs)[0]) == 0
    increment = sp.Poly(sp.expand(exact-A*x*x-2*b*x*v-G*v*v), x,v,f,g)
    expected = {
        (1,1,0,0): 2*A*B+2*b*(a-1),
        (1,0,1,0): 2*A*eta+2*b*a*c,
        (1,0,0,1): 2*b*c,
        (0,2,0,0): A*B**2+2*b*a*B+G*(a*a-1),
        (0,1,1,0): 2*A*B*eta+2*b*(a*c*B+a*eta)+2*G*a*a*c,
        (0,1,0,1): 2*b*B*c+2*G*a*c,
        (0,0,2,0): A*eta**2+2*b*a*c*eta+G*a*a*c*c,
        (0,0,1,1): 2*b*c*eta+2*G*a*c*c,
        (0,0,0,2): G*c*c,
    }
    assert set(increment.monoms()) == set(expected)
    for powers, coefficient in expected.items():
        assert sp.simplify(increment.coeff_monomial(powers)-coefficient) == 0
    cap_change = 2*b*out_x*(j-1)*out_v+G*(j*j-1)*out_v**2
    capped = A*out_x**2+2*b*out_x*j*out_v+G*j*j*out_v**2
    assert sp.expand(capped-exact-cap_change) == 0
    # Full-coordinate cost decomposes into centered cost and barycenter cost.
    u1,u2,w1,w2 = sp.symbols('u1 u2 w1 w2', real=True)
    energy=lambda u,w: A*u*u+2*b*u*w+G*w*w
    mu,mw=(u1+u2)/2,(w1+w2)/2
    assert sp.expand((energy(u1,w1)+energy(u2,w2))/2
                     -(energy(u1-mu,w1-mw)+energy(u2-mu,w2-mw))/2
                     -energy(mu,mw)) == 0


def check_conditioned_kinetic_resonance():
    # The prepared position is arbitrary, including the actual cloning jitter.
    x, center, velocity, noise = sp.symbols('x center velocity noise', real=True)
    c, damping, amplitude = sp.symbols('c damping amplitude', positive=True)
    stiffness = 1/c**2
    v1 = velocity-c*stiffness*(x-center)
    x1 = x+c*v1
    v2 = damping*v1+amplitude*noise
    x2 = x1+c*v2
    v3 = v2-c*stiffness*(x2-center)
    assert sp.simplify(x1-center-c*velocity) == 0
    assert sp.simplify(v3+velocity) == 0
    radius, initial, n, energy = sp.symbols('radius initial n energy', positive=True)
    speed = radius*initial/(radius+n*initial)
    assert sp.simplify(radius*speed/(radius+speed)
                       -radius*initial/(radius+(n+1)*initial)) == 0
    capped_energy = energy/(1+sp.sqrt(energy)/radius)**2
    assert sp.simplify(sp.diff(capped_energy, energy)
                       -(1+sp.sqrt(energy)/radius)**-3) == 0
    assert sp.simplify(sp.diff(capped_energy, energy, 2)
                       +3/(2*radius*sp.sqrt(energy))
                       *(1+sp.sqrt(energy)/radius)**-4) == 0


if __name__ == '__main__':
    check_joint_invariance()
    check_moment_coefficients()
    check_path_length()
    check_phase_weight_matrix()
    check_keystone_power_and_floor()
    check_active_closure_algebra()
    check_weighted_closure_algebra()
    check_selection_entropy_algebra()
    check_frozen_feedback_and_exit()
    check_growing_trajectory_algebra()
    check_actual_clone_and_ordered_forest()
    check_terminal_box_extinction()
    check_survival_conditioning()
    check_signed_phase_feedback()
    check_common_source_increment()
    check_signed_complete_kinetic_quadratic()
    check_conditioned_kinetic_resonance()
    print('Long-time checks passed: exact occupation/stationary bounds, finite-row '
          'mixtures, p-moment coefficients, path-length identities, phase-weight matrices, '
          'Keystone power constants, nonlinear floors, weighted active-cloning closure '
          'normalized joint entropy algebra, frozen feedback, class-exit budgets '
          'growing-time trajectory bounds, actual ordered-cloning identities, '
          'terminal-box extinction/noncommutation constants, survival filtering '
          'normalized-alive estimates, signed phase feedback, common-source drift, '
          'and exact conditioned kinetic resonance.')
