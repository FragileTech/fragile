# Actual-normalizer fitness and signed native source pressure

(sec-kusp-realized-fitness)=
## 1. A population-uniform bound for the complete sampled fitness array

:::{prf:lemma} Global logistic remainder with an explicit scalar certificate
:label: lem-kusp-logistic-remainder

For every real $z$,

$$
\tanh(z/2)\le z/2+.07z^2,
\qquad |\tanh(z/2)|\le |z|/2.
\tag{KUSP.1}
$$

:::

:::{prf:proof}
The second inequality follows from the derivative bound
$0\le\tanh'(x)\le1$ and oddness. For $z\ge0$, it also proves
the first inequality. For $z=-r<0$, the first assertion is
equivalent to $(r/2-\tanh(r/2))/r^2\le.07$.
For $0\le r\le1.5$, the identity
$\tanh'(x)=1-\tanh^2x\ge1-x^2$ gives
$\tanh x\ge x-x^3/3$. Thus the quotient is at most
$r/24\le.0625$. For $r\ge10$, positivity of $\tanh(r/2)$
gives an upper bound $1/(2r)\le.05$.

On $[1.5,10]$, use the 170 intervals $[l,l+.05]$ with
$l=1.5+.05j$, $0\le j<170$. Monotonicity and positivity give

$$
\sup_{r\in[l,l+.05]}
\frac{r/2-\tanh(r/2)}{r^2}
\le\frac{(l+.05)/2-(e^l-1)/(e^l+1)}{l^2}.
\tag{KUSP.2}
$$

The directed 85-digit interval evaluation in
`verify_logistic_fitness.py` bounds the largest right-hand side
by $.068885150602369<.07$. The three intervals cover the full
negative half-line. No Gaussian tail or fitness regime is
discarded. $\square$
:::

:::{prf:proposition} Actual-normalizer population-uniform mean fitness
:label: prop-kusp-native-mean-fitness

Let $M\ge1$ be the actual alive count. For each of the reward
and diversity channels, use its actual complete measurement array,
alive-only mean and standard deviation
$s=\sqrt{M^{-1}\sum_i(a_i-\bar a)^2+\sigma_0^2}$ with its
configured positive floor. Put $z_i=(a_i-\bar a)/s$ and
$G_{A,e}(z)=A/(1+e^{-z})+e$, where $A,e>0$.
Then, pathwise for every measurement array,

$$
\frac1M\sum_iG_{A,e}(z_i)^2
\le (A/2+e)^2+.07A(A/2+e)+A^2/16=:B(A,e).
\tag{KUSP.3}
$$

For the unit-power product of the two actual channels,

$$
\bar f\le\sqrt{B(A_R,e_R)B(A_D,e_D)}.
\tag{KUSP.4}
$$

In the native record $A_R=A_D=2$, $e_R=e_D=.1$,
this gives the explicit bound $\bar f\le1.614$, for every $N$,
every nonextinct alive mask and every joint companion mark array.
:::

:::{prf:proof}
The actual standardization gives $\bar z=0$ and
$\overline{z^2}\le1$ even for tied or nearly tied values.
Write $m=A/2+e$ and $T_i=\tanh(z_i/2)$, so
$G_{A,e}(z_i)=m+(A/2)T_i$. By (KUSP.1),
$\bar T\le.07\overline{z^2}\le.07$ and
$\overline{T^2}\le\overline{z^2}/4\le1/4$.
Expanding the square proves (KUSP.3). Cauchy--Schwarz on the
two realized arrays gives (KUSP.4); it does not use independence
of the channels or replace a sampled measurement by its mean.
For $A=2,e=.1$ the bound is $1.21+.154+.25=1.614$.
Dead entries do not enter either normalizer. $\square$
:::

(sec-kusp-signed-pressure)=
## 2. The signed source balance with original clone weights and revival

:::{prf:proposition} Signed native position-ancestry pressure
:label: prop-kusp-signed-native-source-pressure

Freeze the entire actual sampled fitness array before cloning.
Let $\mathcal A$ be its $M\ge2$ alive labels, $f_i\ge0$,
and keep the original symmetric Gaussian clone weights
$0<K_{ij}=K_{ji}\le1$. Alive queries exclude self; dead queries
choose an alive donor with their original Gaussian row normalization.
Use the native unit clone scale and original gate regularizer
$\epsilon>0$. Let $\mathcal F_{\rm pre}$ contain the entering
state and all reward/diversity measurement marks, before the
cloning donor and acceptance draws that are still averaged. For an alive position source $j$, set

$$
W_j=\sum_{i\in\mathcal A\setminus\{j\}}K_{ji},\quad
d_j=\frac{W_j}{M-1},\quad
A_j=\frac1{M-1}\sum_{i\in\mathcal A\setminus\{j\}}f_i,
\quad
R_j=\sum_{i\notin\mathcal A}
\frac{K_{ij}}{\sum_{k\in\mathcal A}K_{ik}}.
\tag{KUSP.5}
$$

If $L_j$ is the number of output rows whose position ancestry
is $j$, including its persistent token, then

$$
\mathbb E[L_j\mid\mathcal F_{\rm pre}]-1
\ge
\max\left\{-1,\frac{f_jd_j-A_j/d_j}{f_j+\epsilon}\right\}+R_j.
\tag{KUSP.6}
$$

When $M=1$, its sole source instead has $L_j=N$ exactly.
No statement that velocities are copied from the donor is made:
the collision retains every entering velocity, including dead rows.
:::

:::{prf:proof}
Write $b=f_j$, $a=f_i$, and
$g(a,b)=\min\{1,(b-a)_+/(a+\epsilon)\}$. The actual gates
satisfy

$$
g(a,b)\ge\frac{(b-a)_+}{b+\epsilon},\qquad
g(b,a)\le\frac{(a-b)_+}{b+\epsilon}.
\tag{KUSP.7}
$$

The first inequality follows because its right-hand side is
less than one and its denominator is at least $a+\epsilon$
on the positive branch. The second follows directly from the
clipped gate. Define
$X=\sum_{i\ne j}K_{ji}(b-f_i)_+$ and
$Y=\sum_{i\ne j}K_{ji}(f_i-b)_+$, with sums over alive labels.
The signed incoming minus outgoing alive source mass is at least

$$
\frac{X/(M-1)-Y/W_j}{b+\epsilon}
=\frac{d_j(X-Y)-(1-d_j)Y}{W_j(b+\epsilon)}.
\tag{KUSP.8}
$$

Indeed every incoming row has $W_i\le M-1$, while the outgoing
row has its exact denominator $W_j$; symmetry identifies their
edge weights. Keep this signed cancellation:
$X-Y=bW_j-\sum_iK_{ji}f_i$, $Y\le\sum_iK_{ji}f_i$ and
$0<d_j\le1$. Hence (KUSP.8) is at least

$$
\frac{bd_j-\sum_iK_{ji}f_i/W_j}{b+\epsilon}
\ge\frac{bd_j-A_j/d_j}{b+\epsilon}.
\tag{KUSP.9}
$$

Its actual alive signed excess is also at least $-1$: outgoing
mass cannot exceed its single entering token. Mandatory dead
revival contributes exactly $R_j$, independently of its zeroed
fitness. This proves (KUSP.6). In the singleton case every
revived position has the sole alive source, while that source
persists without jitter. $\square$
:::

:::{prf:corollary} An explicit native sufficient source-pressure margin
:label: cor-kusp-positive-native-source-pressure

In the native unit-power record, if the actual sampled source has
$M\ge1000$, $f_j\ge2.2$ and $d_j\ge.89$, then

$$
\mathbb E[L_j\mid\mathcal F_{\rm pre}]-1
>.0648+R_j.
\tag{KUSP.10}
$$

These are inequalities on the actual source and actual complete
measurement array. Their production on a reached class must be
established before they are used along successive updates.
:::

:::{prf:proof}
By (KUSP.4), $A_j\le1.614M/(M-1)\le1.614(1000/999)$.
The lower expression in (KUSP.6), with this overestimate, increases
with both $f_j$ and $d_j$. Its value at $2.2,.89$ and the actual
$\epsilon=10^{-6}$ is greater than $.06486431391989$ by the
directed scalar receipt. Revival is then added exactly.
$\square$
:::
