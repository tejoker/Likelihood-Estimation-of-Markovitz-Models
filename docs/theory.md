# Statistical framework

*Theory section of "Is Synthetic Financial Data Good Enough? A Wasserstein-Critic Two-Sample
Test, Likelihood Benchmarks for Markov Models, and Markowitz Portfolio Utility".*

This is the source for `paper/sections/theory.tex`. It must agree with `docs/DESIGN.md`, which
defines the binding interfaces. Section 11 lists implementation notes: recommendations to module
owners, not part of the paper.

---

## 0. Conventions, notation and status labels

**Status labels.** *Proposition, Lemma, Corollary*: proved here. A proof labelled *sketch* gives
every key inequality but leaves out routine measure-theoretic detail. *Fact*: a standard result,
quoted with a reference. **Heuristic**: a plausible argument or approximation that is not a
theorem. Heuristics are always flagged in bold.

**Data conventions (DESIGN §3.1).** Returns are daily log-returns $r_t=\log P_t-\log P_{t-1}$
under the physical measure, and one time step is one trading day. Windows do not overlap
(step $L$); the default is $L=20$. A univariate window lies in $\mathbb R^{L}$ and a multivariate
window in $\mathbb R^{N\times L}\cong\mathbb R^{NL}$. We write $d$ for the dimension ($L$ or
$NL$) and $\lVert\cdot\rVert$ for the Euclidean (Frobenius) norm.

| Symbol | Meaning |
|---|---|
| $P,\ Q$ | laws of one real and one synthetic window (Borel probability measures on $\mathbb R^d$) |
| $X_1,\dots,X_n$; $Y_1,\dots,Y_m$ | real and synthetic windows |
| $I_{\rm tr},I_{\rm ev}$; $J_{\rm tr},J_{\rm ev}$ | train/eval index sets, sizes $n_t,n_e$ and $m_t,m_e$; $N_e=n_e+m_e$ |
| $\mathcal G$ | $\sigma$-field of the training halves, the index sets and the training randomness $\xi$ (initialisation, mini-batches) |
| $\widehat f$ | trained critic, a $\mathcal G$-measurable random function $\mathbb R^d\to\mathbb R$ |
| $A$, $\widehat g$, $P_A,Q_A$ | input standardisation $A(x)=D^{-1}(x-c)$ fitted on the training half; network with $\widehat f=\widehat g\circ A$; standardised laws $A_\#P,A_\#Q$ (Remark 2.3) |
| $Pf$ | $\int f\,dP$; $P^{\rm ev}f=n_e^{-1}\sum_{i\in I_{\rm ev}}f(X_i)$, and likewise $Q^{\rm ev}f$, $P^{\rm tr}f$, $Q^{\rm tr}f$ |
| $\mathrm{Lip}(f)$, $\mathrm{Lip}_c$ | Lipschitz constant w.r.t. $\lVert\cdot\rVert$; $\mathrm{Lip}_c=\{f:\mathrm{Lip}(f)\le c\}$ |
| $\mathcal P_p(\mathbb R^d)$ | probability measures with $\int\lVert x\rVert^p\,dP<\infty$ |
| $\Sigma_P$ | covariance matrix of $P$ |
| $\Phi$, $z_{1-\alpha}$ | standard normal cdf and its $1-\alpha$ quantile |

---

## 1. Setup: windows, the absolute null, and Markov structure

### 1.1 Windows as random elements

Let $(r_t)_{t\in\mathbb Z}$ be a return process, with values in $\mathbb R$ or $\mathbb R^N$,
defined on $(\Omega,\mathfrak F,\mathbb P)$. For $s\in\mathbb Z$ the window
$W_s:=(r_{s+1},\dots,r_{s+L})$ is a random element of $\mathbb R^d$. If $(r_t)$ is strictly
stationary, every window has the same law $P_L$. The real sample consists of non-overlapping
windows $X_i=W_{s_0+(i-1)L}$ of the observed series. The synthetic sample $Y_j$ consists of
windows of paths simulated from a generator $G$ (DESIGN §3.4). We write $P=\mathrm{Law}(X_i)$
and $Q=\mathrm{Law}(Y_j)$.

**Definition 1.1 (absolute null).** $H_0:\ P=Q$ against $H_1:\ P\neq Q$.

$H_0$ is a statement about the *marginal* law of one window. Dependence *between* windows, such
as successive real windows of one path, is not part of $H_0$. It is a nuisance that changes the
null distribution of statistics. Sections 2–6 assume independent windows; Section 7 removes that
assumption.

### 1.2 What a window law identifies

**Proposition 1.2.** Let $(r_t)$ be strictly stationary with window law $P_L$.

(a) $P_L$ determines the joint law of $(r_{t_1},\dots,r_{t_k})$ for all
$t_1<\dots<t_k$ with $t_k-t_1\le L-1$. In particular, $P_L$ determines every autocovariance and
every cross-moment $\mathbb E\,g(r_t,r_{t+h})$ with $0\le h\le L-1$: the ACF of $r_t$, $|r_t|$
and $r_t^2$, and leverage correlations $\operatorname{Corr}(r_t,r_{t+h}^2)$. It also determines
the law of aggregated returns $\sum_{t=1}^{h}r_t$ for $h\le L$.

(b) The family $(P_L)_{L\ge1}$ determines the law of the process $(r_t)_{t\in\mathbb Z}$.

(c) For every $L\ge1$ there are strictly stationary processes with the same $P_L$ and different
$P_{L+1}$.

*Proof.* (a) By stationarity,
$(r_{t_1},\dots,r_{t_k})\overset d=(r_1,r_{1+t_2-t_1},\dots,r_{1+t_k-t_1})$. This is a coordinate
projection of $W_0=(r_1,\dots,r_L)$, because $1+t_k-t_1\le L$.
(b) By (a), every finite-dimensional distribution is a marginal of some $P_L$. Cylinder sets form
a $\pi$-system that generates the product $\sigma$-field, so the finite-dimensional distributions
determine the law (uniqueness part of Kolmogorov's extension theorem; Kallenberg 2002, Ch. 6).
(c) Let $(e_t)$ be i.i.d. $N(0,1)$, $\theta\neq0$, and
$r_t=(e_t+\theta e_{t-L})/\sqrt{1+\theta^2}$. This Gaussian process has $\gamma(0)=1$,
$\gamma(h)=0$ for $1\le h\le L-1$, and $\gamma(L)=\theta/(1+\theta^2)\neq0$. Its window law $P_L$
is $N(0,I_L)$, the same as for i.i.d. $N(0,1)$ returns, but the laws $P_{L+1}$ differ. $\square$

So a test on windows of length $L$ tests the necessary condition $P_L=Q_L$. That is, it tests
equality of every dependence feature at lags $0,\dots,L-1$ and of aggregated returns up to
horizon $L$. It has no power against deviations that appear only at lags $\ge L$, however large
the sample. This is the trade-off behind the choice of $L$ (E5 uses $L\in\{5,10,20,60\}$). Longer
windows see longer-range dependence. But they give fewer windows ($n=\lfloor T/L\rfloor$) in a
higher dimension ($d=L$ or $NL$), and the critic is harder to learn.

### 1.3 Markov structure and the "Markov models" of the title

**Proposition 1.3 (pairs determine a stationary Markov chain).** Let $(r_t)$ and $(r'_t)$ be
strictly stationary, time-homogeneous, first-order Markov chains with values in a Polish space
$S$. If $(r_0,r_1)\overset d=(r'_0,r'_1)$, then $(r_t)\overset d=(r'_t)$ as processes. For
order-$k$ chains, $P_{k+1}=P'_{k+1}$ implies equality of the processes.

*Proof.* Let $\mu$ be the common law of $r_0$ and $r'_0$, and let $K,K'$ be the transition
kernels. Under the common pair law, both $K(x,\cdot)$ and $K'(x,\cdot)$ are versions of the
regular conditional law of the second coordinate given the first. Disintegrations are unique
almost everywhere, so $K(x,\cdot)=K'(x,\cdot)$ for $\mu$-a.e. $x$ (Kallenberg 2002, Ch. 6). By the
Markov property and stationarity, the law of $(r_0,\dots,r_k)$ is
$\mu(dx_0)K(x_0,dx_1)\cdots K(x_{k-1},dx_k)$. Each $r_j$ has law $\mu$, so the $\mu$-null set on
which $K\neq K'$ does not matter, and induction on $k$ shows that all finite-dimensional laws
agree. Proposition 1.2(b) then gives the claim. For order $k$, apply the result to the
first-order chain $s_t=(r_t,\dots,r_{t+k-1})$, whose pair law is a function of $P_{k+1}$.
$\square$

**Remark 1.4 (both processes must be Markov).** Proposition 1.3 compares two Markov chains. If
only the generator is Markov, $P_2=Q_2$ says only that the real process has the same pair law as
the generator; the real process may still be non-Markov. So a test with $L=2$ is a complete test
of "the real process is the generator's chain" only if the real process is assumed to be
first-order Markov.

**Remark 1.5 (the models of this paper).**

| Model (DESIGN §3.5) | Markov state | Returns Markov? | Likelihood |
|---|---|---|---|
| GBM, Merton, i.i.d. Student-$t$ (E2) | none (i.i.d.) | order 0 | exact; a window density is a product of daily densities |
| GARCH(1,1)-$t$ | $(r_t,\sigma^2_{t+1})$, with $\sigma^2_{t+1}$ a function of past returns | not of any finite order | exact given $\sigma_1^2$ (prediction-error decomposition) |
| Heston (Euler) | $(r_t,v_{t+1})$, with $v_t$ latent and driven by its own noise | no | intractable |
| Stationary bootstrap | block position (not a parametric model) | no | none |

*GARCH.* Write $r_t=\mu+\sigma_tz_t$ with $\varepsilon_t=r_t-\mu$ and
$\sigma^2_{t+1}=\omega+a\varepsilon_t^2+b\sigma_t^2$. Then
$\sigma^2_{t+1}=\omega/(1-b)+a\sum_{j\ge0}b^j\varepsilon_{t-j}^2$ is a function of past returns:
the model is *observation-driven* (Cox 1981). Given $\sigma_1^2$, the log-likelihood is therefore
explicit:
$\sum_t\log[\sigma_t^{-1}g_\nu((r_t-\mu)/\sigma_t)]$, where $g_\nu$ is the standardised $t$
density.

The returns themselves are not Markov of any finite order $k$ when $a,b>0$ (*sketch*). If they
were, $\sigma^2_{t+1}=\mathbb E[(r_{t+1}-\mu)^2\mid\mathfrak F_t]$ would be a function of
$(r_{t-k+1},\dots,r_t)$. Hence so would
$b^k\sigma^2_{t-k+1}=\sigma^2_{t+1}-\omega\sum_{j<k}b^j-a\sum_{j<k}b^j\varepsilon^2_{t-j}$.
But given $\sigma^2_{t-k+1}=v$, the window $(r_{t-k+1},\dots,r_t)$ is a triangular,
positive-Jacobian transformation of $k$ independent innovations with positive densities. So it
has an everywhere positive Lebesgue density for every $v>0$. The conditional laws for different
$v$ are therefore mutually absolutely continuous, and $v$ cannot be a function of the window
unless the law of $\sigma^2$ is degenerate.

The squared-return autocorrelation decays geometrically, $\rho_h=\rho_1(a+b)^{h-1}$ (Francq and
Zakoïan 2019). For $a+b=0.98$, $\rho_{20}/\rho_1=0.98^{19}\approx0.68$. Information about the
process law is spread over lags of order $1/(1-a-b)=50$ days, beyond $L=20$.

*Heston.* The variance $v_t$ is Markov with its own shock, and returns load on a shock that is
correlated with the next variance shock. The state is latent and continuous
(*parameter-driven*) and has no closed-form filter, so the likelihood requires integrating out
the latent variance path and is intractable. This is where a likelihood-free test is needed.

So the "Markov models" of the title are models that are Markov in an augmented state. Among the
models of this paper, exact likelihoods, and with them the likelihood benchmarks (the E2 oracle;
the E3 predictive log-likelihood, PIT and Vuong tests), are available for the i.i.d. models and
GARCH, whose augmented state is a function of observed returns. Heston's latent continuous state
makes its likelihood intractable. Being parameter-driven is not by itself the obstacle:
finite-state Markov-switching models (Hamilton 1989) and linear-Gaussian state-space models
(Kalman filter; Durbin and Koopman 2012) are parameter-driven, yet their likelihoods are exact
through recursive filtering. Proposition 1.3 shows that $L=2$ would suffice to compare two first-order Markov return
processes. To our knowledge, there is no analogue of Proposition 1.3 with a finite $L$ for
hidden-Markov processes with a continuous state. The geometric decay above suggests that $L$
should be of the order of the volatility half-life ($\log 0.5/\log 0.98\approx34$ days for
persistence $0.98$). $L=20$ is a compromise that E5 varies.

### 1.4 Pooling across tickers and normalisation

**Remark 1.6.** In E3, windows of many tickers are pooled after the returns of ticker $i$ are
divided by $s_i$, the standard deviation of its TRAIN returns (DESIGN §2). Conditionally on
TRAIN, $x\mapsto x/s_i$ is a fixed bijection applied to the real *and* synthetic windows of
ticker $i$, so the per-ticker null $P_i=Q_i$ is unchanged.

Suppose ticker $i$ contributes a fraction $\pi_i$ of the real windows and $\pi'_i$ of the
synthetic ones. The pooled laws are the mixtures $\bar P=\sum_i\pi_iP_i^{(s)}$ and
$\bar Q=\sum_i\pi'_iQ_i^{(s)}$. The per-ticker nulls imply the pooled null
$H_0^{\rm pool}:\bar P=\bar Q$ only if $\pi_i=\pi'_i$. **Each ticker must therefore contribute
the same number of real and synthetic windows** (or a fixed common ratio); otherwise the test
detects the composition of the pool, not the quality of the generator (Implementation note 11.5).

$H_0^{\rm pool}$ is weaker than $\bigcap_i\{P_i=Q_i\}$, because per-ticker errors can cancel in
the mixture. The per-ticker tests of E3, with Benjamini–Hochberg (Section 8.2), address this.

---

## 2. IPMs, Wasserstein-1, and the held-out critic statistic

### 2.1 Integral probability metrics and Kantorovich–Rubinstein duality

**Definition 2.1.** For a class $\mathcal F$ of measurable functions, the *integral probability
metric* (IPM; Müller 1997) is $d_{\mathcal F}(P,Q)=\sup_{f\in\mathcal F}|Pf-Qf|$. Examples:
- $\mathcal F=\mathrm{Lip}_1$ gives $W_1$ (Fact 2.2);
- $\mathcal F=\{f:\lVert f\rVert_\infty\le1\}$ gives $2\,\mathrm{TV}$;
- the unit ball of an RKHS gives the MMD (Gretton et al. 2012).

**Fact 2.2 (Kantorovich–Rubinstein duality).** For $P,Q\in\mathcal P_1(\mathbb R^d)$,
$$
W_1(P,Q):=\inf_{\pi\in\Pi(P,Q)}\int\lVert x-y\rVert\,\pi(dx,dy)=\sup_{f\in\mathrm{Lip}_1}(Pf-Qf),
$$
where $\Pi(P,Q)$ is the set of couplings. $W_1$ is a metric on $\mathcal P_1$, so
$W_1(P,Q)=0\iff P=Q$. It metrises weak convergence together with convergence of first moments.
References: Kantorovich and Rubinstein (1958); Villani (2009, Ch. 5–6, including Thm 6.9 for the
metrisation); Dudley (2002, §11.8). Because $\mathrm{Lip}_1$ is symmetric under $f\mapsto-f$, the absolute value in
Definition 2.1 can be dropped.

So, on $\mathcal P_1$, $H_0\iff W_1(P,Q)=0$, and every fixed alternative has $W_1(P,Q)>0$.

**Remark 2.3 (the representation).** The critic does not see raw windows. It is
$\widehat f=\widehat g\circ A$, where $\widehat g$ is the network and $A(x)=D^{-1}(x-c)$ is the
input standardisation; $c$ and the positive diagonal $D$ are computed from the pooled training
half (DESIGN §3.7; per coordinate for univariate windows, per asset channel for multivariate
ones). Conditionally on $\mathcal G$, $A$ is a fixed affine bijection, so
$A_\#P=A_\#Q\iff P=Q$.

The gradient penalty differentiates $\widehat g$ with respect to the *standardised* input, so
the Lipschitz constant it pushes towards 1 is $\mathrm{Lip}(\widehat g)$. This is the Lipschitz
constant of $\widehat f$ for the metric $\lVert x-y\rVert_A:=\lVert D^{-1}(x-y)\rVert$. In raw
coordinates only $\mathrm{Lip}(\widehat f)\le\lVert D^{-1}\rVert_{\rm op}\mathrm{Lip}(\widehat g)$
holds, and for unnormalised daily returns $\lVert D^{-1}\rVert_{\rm op}\approx1/\sigma_{\rm daily}$
is about $50$–$100$. We therefore do **not** absorb $A$ into $f$. We write
$$
P_A:=A_\#P,\qquad Q_A:=A_\#Q,\qquad W_1(P_A,Q_A)=\inf_{\pi\in\Pi(P,Q)}\int\lVert x-y\rVert_A\,\pi(dx,dy),
$$
so $W_1(P_A,Q_A)$ is $W_1(P,Q)$ under the weighted metric $\lVert\cdot\rVert_A$, with
$\lVert D\rVert_{\rm op}^{-1}W_1(P,Q)\le W_1(P_A,Q_A)\le\lVert D^{-1}\rVert_{\rm op}W_1(P,Q)$.
"$W_1$ on the representation" in DESIGN means $W_1(P_A,Q_A)$. Conditionally on $\mathcal G$, the
standardised eval windows $A(X_i)$ and $A(Y_j)$ have laws $P_A$ and $Q_A$, and
$P\widehat f-Q\widehat f=P_A\widehat g-Q_A\widehat g$. Lipschitz statements about the critic
(Proposition 2.5(b)–(c), Remarks 2.6–2.8, Section 4) are made for $\widehat g$ and $P_A,Q_A$;
the level results of Section 3 involve no metric and are unaffected.

### 2.2 The held-out critic statistic

The critic $\widehat f$ is trained on the training halves only, with the WGAN-GP objective
(Gulrajani et al. 2017; DESIGN §3.7):
$$
\widehat J(f)=P^{\rm tr}f-Q^{\rm tr}f-\lambda\,\mathbb E_{\tilde x}\big(\lVert\nabla f(\tilde x)\rVert-1\big)^2,
\qquad \tilde x=uX+(1-u)Y,\ u\sim U(0,1),
$$
where $X,Y$ are drawn from the training halves and $\lambda=10$. The statistic is
$$
T:=P^{\rm ev}\widehat f-Q^{\rm ev}\widehat f=\frac1{n_e}\sum_{i\in I_{\rm ev}}\widehat f(X_i)-\frac1{m_e}\sum_{j\in J_{\rm ev}}\widehat f(Y_j).
\tag{2.1}
$$

**Fact 2.4 (the critic is globally Lipschitz).** The architectures of DESIGN §3.7 are
compositions of globally Lipschitz maps: affine and convolutional layers, LeakyReLU, and
LayerNorm with $\epsilon>0$, whose Jacobian has operator norm at most
$\lVert\gamma\rVert_\infty/\sqrt\epsilon$. So $\mathrm{Lip}(\widehat g)<\infty$ and
$\mathrm{Lip}(\widehat f)\le\lVert D^{-1}\rVert_{\rm op}\mathrm{Lip}(\widehat g)<\infty$, although
the constants are not controlled. Since
$|\widehat f(x)|\le|\widehat f(0)|+\mathrm{Lip}(\widehat f)\lVert x\rVert$,
$\widehat f\in L^p(P)\cap L^p(Q)$ whenever $P,Q\in\mathcal P_p$.

**Proposition 2.5 (conditional unbiasedness; lower bound on $W_1$ of the representation).**
Assume that, conditionally on $\mathcal G$, every eval window $X_i$ has law $P$ and every eval
window $Y_j$ has law $Q$, and that $P,Q\in\mathcal P_1$. Let $\widehat f=\widehat g\circ A$ as in
Remark 2.3. Then:

(a) $\mathbb E[T\mid\mathcal G]=P\widehat f-Q\widehat f=P_A\widehat g-Q_A\widehat g=:\Delta(\widehat f)$.

(b) $\Delta(\widehat f)\le\mathrm{Lip}_{S_A}(\widehat g)\,W_1(P_A,Q_A)\le\mathrm{Lip}(\widehat g)\,W_1(P_A,Q_A)$,
where $\mathrm{Lip}_{S_A}$ is the Lipschitz constant restricted to
$S_A=\operatorname{supp}P_A\cup\operatorname{supp}Q_A$. Both Lipschitz constants are taken in the
standardised coordinates.

(c) If the eval windows are also conditionally independent and $P,Q\in\mathcal P_2$, then
$$
\operatorname{Var}(T\mid\mathcal G)=\frac{\operatorname{Var}_P\widehat f}{n_e}+\frac{\operatorname{Var}_Q\widehat f}{m_e}\le\mathrm{Lip}(\widehat g)^2\Big(\frac{\operatorname{tr}\Sigma_{P_A}}{n_e}+\frac{\operatorname{tr}\Sigma_{Q_A}}{m_e}\Big),\qquad \Sigma_{P_A}=D^{-1}\Sigma_PD^{-1}.
$$

*Proof.* (a) Conditionally on $\mathcal G$, $\widehat f$ is a fixed integrable function
(Fact 2.4). The claim follows from linearity and the substitution lemma for conditional
expectations of functions of independent arguments; the second equality is the change of
variables $u=A(x)$.
(b) $P_A,Q_A\in\mathcal P_1$ because $A$ is affine. If $\mathrm{Lip}_{S_A}(\widehat g)=0$, then
$\widehat g$ is constant on $S_A$ and $\Delta=0$, because $P_A(S_A)=Q_A(S_A)=1$. Otherwise
$h=\widehat g/\mathrm{Lip}_{S_A}(\widehat g)$ is 1-Lipschitz on $S_A$. By McShane's extension
theorem (McShane 1934), it extends to some $\bar h\in\mathrm{Lip}_1(\mathbb R^d)$ equal to $h$ on
$S_A$. Hence $\Delta(\widehat f)=\mathrm{Lip}_{S_A}(\widehat g)(P_A\bar h-Q_A\bar h)$, and Fact 2.2
bounds the bracket by $W_1(P_A,Q_A)$.
(c) For independent $U,U'\sim P_A$ and Lipschitz $g$:
$\operatorname{Var}g(U)=\tfrac12\mathbb E(g(U)-g(U'))^2\le\tfrac12\mathrm{Lip}(g)^2\,\mathbb E\lVert U-U'\rVert^2=\mathrm{Lip}(g)^2\operatorname{tr}\Sigma_{P_A}$.
Apply it with $g=\widehat g$, using $\operatorname{Var}_P\widehat f=\operatorname{Var}_{P_A}\widehat g$.
$\square$

So $T$ is a conditionally unbiased estimator of $\Delta(\widehat f)$. When
$\mathrm{Lip}(\widehat g)\le1$, i.e. when the critic is 1-Lipschitz in the standardised
coordinates, $\Delta(\widehat f)$ is a lower bound on $W_1(P_A,Q_A)$. This is why DESIGN reports
`extra["W1_lower"] = T`: it is a lower-bound reading for $W_1$ of the *standardised* windows, not
for $W_1(P,Q)$ in raw return units, which can be smaller by up to the factor
$\lVert D^{-1}\rVert_{\rm op}$ (Remark 2.3). Three qualifications follow.

**Remark 2.6 ($T$ estimates a lower bound; it is not one).** $T$ can exceed $W_1(P_A,Q_A)$
through sampling error of order $n_e^{-1/2}+m_e^{-1/2}$. A conservative summary is the one-sided
lower confidence bound $T-z_{1-\alpha}\,\widehat{\rm se}$, with
$\widehat{\rm se}^2=s_X^2/n_e+s_Y^2/m_e$. Asymptotically it lies below
$\Delta(\widehat f)\le\mathrm{Lip}(\widehat g)W_1(P_A,Q_A)$ with probability at least $1-\alpha$.

**Remark 2.7 (the gradient penalty is a soft constraint).** The penalty only enforces the
Lipschitz constraint softly, and only in expectation over random interpolates of training
points. Nothing guarantees $\mathrm{Lip}(\widehat g)\le1$: the gradient can exceed 1 away from the
interpolates, and on them by a margin that depends on $\lambda$. The penalty is also two-sided:
it pushes $\lVert\nabla f\rVert$ *towards* 1 rather than below 1 (Petzka et al. 2018 propose a
one-sided version). WGAN critics are known to be poor estimators of $W_1$ itself (Mallasto et al.
2019; Stanczuk et al. 2021). For this reason we use $T$ as a test statistic, and read
"$W_1$ lower bound" as **approximate**. The *level* of the test (Section 3) does not depend on
any Lipschitz property.

**Remark 2.8 (empirical Lipschitz diagnostic).** Work in the standardised coordinates. On a
probe set $\mathcal U$ (standardised pooled eval points $A(z)$, plus random interpolates between
pairs drawn from them), compute
$$
\widehat L_{\rm grad}=\max_{u\in\mathcal U}\lVert\nabla_u\widehat g(u)\rVert,\qquad
\widehat L_{\rm pair}=\max_{u\neq u'\in\mathcal U}\frac{|\widehat g(u)-\widehat g(u')|}{\lVert u-u'\rVert}
=\max_{z\neq z'}\frac{|\widehat f(z)-\widehat f(z')|}{\lVert z-z'\rVert_A}.
$$
The gradient is taken with respect to the standardised input, as in the penalty; a gradient with
respect to raw windows would be larger by a factor of order $\lVert D^{-1}\rVert_{\rm op}$ and is
not the relevant quantity. Both are *lower* bounds on $\mathrm{Lip}(\widehat g)$ (and
$\widehat L_{\rm pair}\le\mathrm{Lip}_{S_A}(\widehat g)$ when $\mathcal U\subset S_A$). Report
$\widehat L$ and $T/\widehat L$:
- If $\widehat L\gg1$, the raw $T$ overstates the "lower bound" on $W_1(P_A,Q_A)$ by up to a
  factor $\widehat L$, and $T/\widehat L$ is the better approximation.
- $T/\widehat L$ is **not** a guaranteed lower bound, because $\widehat L\le\mathrm{Lip}(\widehat g)$.
- The $W_1$ reading of $T$ deserves trust only when $\widehat L\approx1$.

Rescaling by a positive constant that is invariant under permutations of the pooled eval set
leaves the permutation p-value unchanged.

**Remark 2.9 (why hold data out).** The plug-in $W_1(P_n,Q_m)$ is biased upwards and converges
slowly: $\mathbb E\,W_1(P_n,P)$ is of order $n^{-1/d}$ for $d\ge3$ (Dudley 1969; Fournier and
Guillin 2015). For $d=20$ and $n=2000$, $n^{-1/d}\approx0.68$, which swamps plausible effects. The
in-sample training objective $\sup_f(P^{\rm tr}f-Q^{\rm tr}f)$ is biased upwards in the same way.
The held-out $T$ is conditionally unbiased for $\Delta(\widehat f)$ and has an exact null
distribution (Section 3), at the cost of sample splitting (quantified in Proposition 5.6). For
sample-splitting two-sample tests, see Lopez-Paz and Oquab (2017), Kim et al. (2021) and Liu et
al. (2020); for $W_1$-based testing, see Ramdas et al. (2017).

---

## 3. Exact finite-sample level of the fixed-critic permutation test

**Assumption A1 (i.i.d. sampling, independent halves).**
(i) $X_1,\dots,X_n$ are i.i.d. $P$, $Y_1,\dots,Y_m$ are i.i.d. $Q$, and the two samples are
independent.
(ii) The index sets $I_{\rm tr},I_{\rm ev},J_{\rm tr},J_{\rm ev}$ are chosen independently of the
sample values. Random, deterministic and group-wise splits all qualify.
(iii) $\widehat f$ is a measurable function of the training windows and of randomness $\xi$ that
is independent of the data.
(iv) The permutations $\pi_1,\dots,\pi_B$ are drawn independently and uniformly from a finite
group $\mathcal K$ of permutations of $\{1,\dots,N_e\}$, independently of everything else.
$\mathcal K=S_{N_e}$ by default; for stratified permutation, see Remark 3.5(e).

Pool the eval windows as $Z=(Z_1,\dots,Z_{N_e})=(X_{I_{\rm ev}},Y_{J_{\rm ev}})$, the first
$n_e$ coordinates being real, and define
$$
t(z)=\frac1{n_e}\sum_{k\le n_e}\widehat f(z_k)-\frac1{m_e}\sum_{k>n_e}\widehat f(z_k),\qquad (\pi\cdot z)_k=z_{\pi^{-1}(k)} .
$$
Then $T_0:=t(Z)=T$ and $T_b:=t(\pi_b\cdot Z)$. The p-value of DESIGN §3.7 is
$$
p=\frac{1+\#\{b\ge1:T_b\ge T_0\}}{1+B}.
\tag{3.1}
$$

**Assumption A1′ (conditional invariance).** Conditionally on $\mathcal G$, the law of $Z$ is
invariant under $\mathcal K$: $\pi\cdot Z\overset d=Z$ for every $\pi\in\mathcal K$. Under $H_0$,
A1 implies A1′ with $\mathcal K=S_{N_e}$, because $Z$ is then conditionally i.i.d.

**Proposition 3.1 (exact level).** Under A1′ (in particular, under A1 and $H_0$), for every
$\alpha\in(0,1)$, every $B\ge1$, every $n_e,m_e\ge1$ and *every* critic, however trained,
$$
\mathbb P(p\le\alpha\mid\mathcal G)\le\frac{\lfloor\alpha(B+1)\rfloor}{B+1}\le\alpha\quad\text{a.s.},\qquad\text{hence}\qquad\mathbb P(p\le\alpha)\le\alpha .
$$

The proof uses two lemmas. Conditioning on $\mathcal G$ fixes $\widehat f$, hence $t$, so we work
conditionally on $\mathcal G$ throughout.

**Lemma 3.2 (random permutations give an exchangeable vector; Hemerik and Goeman 2018).** Under
A1′ and A1(iv), with $\pi_0=\mathrm{id}$, the vector $(T_0,T_1,\dots,T_B)$ is exchangeable.

*Proof.* Let $h$ be uniform on $\mathcal K$, independent of $(Z,\pi_1,\dots,\pi_B)$. For every
fixed $g\in\mathcal K$, $g\cdot Z\overset d=Z$, and $Z$ is independent of $(h,\pi_{1:B})$. Hence
$(h\cdot Z,\pi_1,\dots,\pi_B)\overset d=(Z,\pi_1,\dots,\pi_B)$, so
$$
(t(\pi_b\cdot Z))_{b=0}^B\overset d=(t(\pi_b\cdot(h\cdot Z)))_{b=0}^B=(t((\pi_bh)\cdot Z))_{b=0}^B,
$$
using that $(\pi,z)\mapsto\pi\cdot z$ is a group action. The tuple $(\pi_0h,\dots,\pi_Bh)$ is
i.i.d. uniform on $\mathcal K$: $h$ is uniform, and given $h$ the $\pi_bh$ are i.i.d. uniform
because right multiplication is a bijection of $\mathcal K$. The tuple is also independent of $Z$.
An i.i.d. tuple is exchangeable, and applying the same map $\sigma\mapsto t(\sigma\cdot Z)$ to each
coordinate preserves exchangeability. $\square$

**Lemma 3.3 (rank bound).** If $(T_0,\dots,T_B)$ is exchangeable and
$R:=\#\{b\in\{0,\dots,B\}:T_b\ge T_0\}$, then $\mathbb P(R\le k)\le k/(B+1)$ for
$k=0,1,\dots,B+1$.

*Proof.* Let $R_c:=\#\{b:T_b\ge T_c\}$. By exchangeability $R_c\overset d=R_0=R$, so
$\mathbb P(R\le k)=\frac1{B+1}\mathbb E\,\#\{c:R_c\le k\}$. Deterministically,
$\#\{c:R_c\le k\}\le k$. Indeed, if the set $C$ of such $c$ had more than $k$ elements, its
element $c^*$ with the smallest $T_{c^*}$ would satisfy $R_{c^*}\ge|C|>k$, a contradiction.
$\square$

*Proof of Proposition 3.1.* $p=R/(B+1)$, so
$\{p\le\alpha\}=\{R\le\lfloor\alpha(B+1)\rfloor\}$. Apply Lemmas 3.2 and 3.3 conditionally on
$\mathcal G$, then take expectations. $\square$

This is the randomisation-test principle of Lehmann and Romano (2005, §15.2), in the
random-permutation form of Dwass (1957), Phipson and Smyth (2010) and Hemerik and Goeman (2018).

**Remark 3.4 (what the level guarantee does and does not need).**
- It needs **no** Lipschitz constraint, **no** convergence of training and **no** continuity. A
  poor or overfitted critic costs power, never level.
- It **does** need conditional invariance of the eval set given the training data. This fails if
  real windows depend on each other, or if eval windows depend on training windows (Section 7).

**Remark 3.5 (weaker nulls, studentisation, groups, seeds).**

(a) *Asymptotic z.* `extra["z"]` $=T/\sqrt{s_X^2/n_e+s_Y^2/m_e}$. Conditionally on
$\mathcal G$, it converges to $N(0,1)$ whenever $P\widehat f=Q\widehat f$ and second moments are
finite (Lindeberg CLT). Under $H_0$, its normal p-value is asymptotically equivalent to (3.1)
(Hoeffding 1952).

(b) *Weaker null.* The permutation test is exact for $P=Q$. For the weaker hypothesis
$H_0':P\widehat f=Q\widehat f$, the unstudentised permutation test can be miscalibrated when
$n_e\neq m_e$ and $\operatorname{Var}_P\widehat f\neq\operatorname{Var}_Q\widehat f$ (Romano 1990).
With the balanced default $n_e=m_e$, the permutation variance matches the true variance
asymptotically. Permuting the *studentised* statistic (Janssen 1997; Chung and Romano 2013) makes
the test asymptotically valid for $H_0'$ while it stays exact under $H_0$. This matters because
Sections 6–7 are about means of $\widehat f$, not about equality of laws.

(c) *Groups.* With tickers as groups and heterogeneous per-ticker laws, the pooled eval vector is
invariant only under *within-group* permutations: A1′ holds for that subgroup under the
per-ticker nulls with independence. Proposition 3.1 then holds with $\mathcal K$ equal to the
within-group permutations (stratified test). The full permutation is not covered.

With $c_g$ real and $c_g$ synthetic eval windows in group $g$, $\sum_g c_g=n_e=m_e$, and group
means $\mu_g$ and variances $\sigma_g^2$, the true null variance is
$\operatorname{Var}T=(2/n_e)\bar\sigma^2_w$ with $\bar\sigma^2_w=\sum_gc_g\sigma_g^2/n_e$. The
full-permutation variance is approximately $(2/n_e)\big(\bar\sigma^2_w+\sum_g(c_g/n_e)(\mu_g-\bar\mu)^2\big)$.
The full permutation is therefore asymptotically **conservative** for balanced groups
(Implementation note 11.1).

(d) *P-value lottery.* The p-value depends on the random split and on $\xi$. Reporting the
smallest p-value over $K$ splits or seeds is invalid. Twice the median of the $K$ p-values
(Meinshausen, Meier and Bühlmann 2009) and twice their mean (Vovk and Wang 2020) are valid under
arbitrary dependence between them.

---

## 4. Consistency

*Standardisation.* The critic is $\widehat f=\widehat g\circ A$, where $A(x)=D^{-1}(x-c)$ is
estimated from the training half (Remark 2.3), so $A$ changes with the sample size. Suppose
$P,Q\in\mathcal P_2$, $n_t/(n_t+m_t)$ converges, and the limiting pooled law has positive
coordinate variances. Then $A\to A^*$ a.s., with $A^*(x)=D_*^{-1}(x-c_*)$ the population
standardisation, and $\lVert D^{-1}\rVert_{\rm op}=O_P(1)$. For every $g\in\mathrm{Lip}_c$,
$$
\big|(P-Q)(g\circ A)-(P-Q)(g\circ A^*)\big|\le c\,(\mathbb E_P+\mathbb E_Q)\lVert A(X)-A^*(X)\rVert
\le c\big(\lVert D^{-1}-D_*^{-1}\rVert_{\rm op}(\mathbb E_P+\mathbb E_Q)\lVert X-c_*\rVert+2\lVert D^{-1}\rVert_{\rm op}\lVert c-c_*\rVert\big),
$$
which tends to 0 a.s., uniformly over $g\in\mathrm{Lip}_c$. We therefore state this section for
the standardised laws $P^*:=A^*_\#P$, $Q^*:=A^*_\#Q$ and the network $\widehat g$, with Lipschitz
constants in the standardised coordinates, and write the proofs for a fixed standardisation. The
estimated $A$ adds $o_P(1)$ terms to Step 1 of Proposition 4.2 and to Corollary 4.3 by the
display above, and, in Lemma 4.1, the supremum over $\{g\circ A:g\in\mathrm{Lip}_c\}$ is at most
$c\lVert D^{-1}\rVert_{\rm op}\big(W_1(P^{\rm tr},P)+W_1(Q^{\rm tr},Q)\big)=o_P(1)$. In Steps 2–3,
$\operatorname{tr}\Sigma_P$ becomes $\operatorname{tr}(D^{-1}\Sigma_PD^{-1})=O_P(1)$. To keep
notation light, in this section $P,Q$ stand for $P^*,Q^*$ and $\widehat f$ for $\widehat g$.

**Assumption C (consistency regime).**
- (C1) Assumption A1 holds, $P,Q\in\mathcal P_2(\mathbb R^d)$, and $P\neq Q$, so
  $W_1(P,Q)>0$.
- (C2) $n_t,m_t,n_e,m_e\to\infty$, and $B\ge\lceil2/\alpha\rceil$ (fixed or growing). Without the
  bound on $B$, (3.1) can never fall below $\alpha$.
- (C3) *Hard Lipschitz constraint (idealisation of the gradient penalty).* There are
  $c<\infty$ and classes $\mathcal F_n\subset\mathrm{Lip}_c$ such that
  $\mathbb P(\widehat f\in\mathcal F_n)\to1$.
- (C4) *Richness.* There are $n_0$ and $f_0\in\bigcap_{n\ge n_0}\mathcal F_n$ with
  $Pf_0-Qf_0=:\eta_0>0$.
- (C5) *Approximate optimisation.* With probability tending to one,
  $(P^{\rm tr}-Q^{\rm tr})\widehat f\ge\sup_{f\in\mathcal F_n}(P^{\rm tr}-Q^{\rm tr})f-\varepsilon_{\rm opt}$
  with $\varepsilon_{\rm opt}\le\eta_0/2$.

**Lemma 4.1 (uniform convergence over Lipschitz classes).** Let $P\in\mathcal P_1(\mathbb R^d)$
and let $P_n$ be the empirical measure of $n$ i.i.d. draws. Then $W_1(P_n,P)\to0$ a.s. and in
$L^1$. By Fact 2.2, $\sup_{f\in\mathrm{Lip}_c}|(P_n-P)f|=c\,W_1(P_n,P)$. So for training halves
of sizes $n_t,m_t\to\infty$,
$$
\Gamma_n:=\sup_{f\in\mathrm{Lip}_c}\big|(P^{\rm tr}-P)f-(Q^{\rm tr}-Q)f\big|\le c\big(W_1(P^{\rm tr},P)+W_1(Q^{\rm tr},Q)\big)\xrightarrow{\ \mathbb P\ }0 .
$$

*Proof.* $P_n\Rightarrow P$ a.s. (Varadarajan 1958) and $\int\lVert x\rVert\,dP_n\to\int\lVert x\rVert\,dP$
a.s. (SLLN). These two facts together are equivalent to $W_1(P_n,P)\to0$ (Villani 2009,
Thm 6.9). Moreover, $W_1(P_n,P)\le n^{-1}\sum_i\lVert X_i\rVert+\mathbb E\lVert X\rVert$, which is
uniformly integrable, so $\mathbb E\,W_1(P_n,P)\to0$. This expectation depends on $n$ only, which
gives convergence in probability along any triangular array of training halves. $\square$

No complexity condition on $\mathcal F_n$ is needed beyond $\mathcal F_n\subset\mathrm{Lip}_c$. The
convergence is slow (Remark 2.9), but consistency needs only a positive margin.

**Proposition 4.2 (consistency of the critic test).** Under Assumption C, the test that rejects
when $p\le\alpha$ satisfies $\mathbb P(\text{reject})\to1$.

*Proof.* Let $\eta=\eta_0/4$ and $A_n=\{\widehat f\in\mathcal F_n,\ \Delta(\widehat f)\ge\eta\}$,
where $\Delta(f)=Pf-Qf$.

*Step 1: $\mathbb P(A_n)\to1$.* On the event in (C3)–(C5), which has probability tending to
one, apply Lemma 4.1 twice:
$$
\Delta(\widehat f)\ge(P^{\rm tr}-Q^{\rm tr})\widehat f-\Gamma_n\ge(P^{\rm tr}-Q^{\rm tr})f_0-\varepsilon_{\rm opt}-\Gamma_n\ge\eta_0-\varepsilon_{\rm opt}-2\Gamma_n\ge\tfrac{\eta_0}2-2\Gamma_n .
$$
Since $\Gamma_n\to0$ in probability, $\Delta(\widehat f)\ge\eta_0/4$ with probability tending to
one.

*Step 2: $T$ is large.* On $A_n$, by conditional Chebyshev and Proposition 2.5(c),
$$
\mathbb P(T\le\eta/2\mid\mathcal G)\le\frac{4c^2}{\eta^2}\Big(\frac{\operatorname{tr}\Sigma_P}{n_e}+\frac{\operatorname{tr}\Sigma_Q}{m_e}\Big)\to0 .
$$

*Step 3: permuted statistics are small.* Let $v_k=\widehat f(Z_k)$, with pooled mean $\bar v$ and
$S^2=N_e^{-1}\sum_k(v_k-\bar v)^2$. For a uniform $\pi$, conditionally on $(\mathcal G,Z)$,
$t(\pi\cdot Z)=(N_e/m_e)\big(n_e^{-1}\sum_{k\in G_1}v_k-\bar v\big)$, where $G_1$ is a uniformly
random $n_e$-subset. Sampling without replacement gives
$$
\mathbb E[t(\pi\cdot Z)\mid\mathcal G,Z]=0,\qquad \operatorname{Var}(t(\pi\cdot Z)\mid\mathcal G,Z)=\frac{N_e^2\,S^2}{n_em_e(N_e-1)}=:V_\pi .
$$
Since $S^2$ minimises the mean squared deviation over centrings, on $A_n$
$$
\mathbb E[S^2\mid\mathcal G]\le c^2(\operatorname{tr}\Sigma_P+\operatorname{tr}\Sigma_Q)+\Delta(\widehat f)^2\le c^2\big(\operatorname{tr}\Sigma_P+\operatorname{tr}\Sigma_Q+W_1(P,Q)^2\big)=:c^2K ,
$$
using $\Delta(\widehat f)\le cW_1$ (Proposition 2.5(b)).

*Step 4: rejection.* Let $k_\alpha=\lfloor\alpha(B+1)\rfloor$. We reject iff
$\#\{b:T_b\ge T_0\}\le k_\alpha-1$. For $B\ge2/\alpha$, $k_\alpha\ge2$ and
$B/k_\alpha\le2/\alpha$. Let $M=\#\{b:T_b\ge\eta/2\}$. On $\{T_0>\eta/2\}$,
$\#\{b:T_b\ge T_0\}\le M$. Given $(\mathcal G,Z)$, Chebyshev and Markov's inequality give
$$
\mathbb P(M\ge k_\alpha\mid\mathcal G,Z)\le\frac{B}{k_\alpha}\cdot\frac{4V_\pi}{\eta^2}\le\frac{8}{\alpha\eta^2}\cdot\frac{N_e^2S^2}{n_em_e(N_e-1)} .
$$
Taking conditional expectations on $A_n$ bounds this by
$\frac{8c^2K}{\alpha\eta^2}\cdot\frac{N_e^2}{n_em_e(N_e-1)}\to0$. Hence
$\mathbb P(\text{no rejection})\le\mathbb P(A_n^c)+\mathbb P(T_0\le\eta/2,A_n)+\mathbb P(M\ge k_\alpha,A_n)\to0$.
$\square$

**Corollary 4.3 ($T$ estimates $W_1$ of the representation under idealised training).** Suppose
(C1)–(C3) hold with $c=1$ (in the standardised coordinates), the classes are rich in the sense
that $\sup_{f\in\mathcal F_n}(Pf-Qf)\to W_1(P,Q)$, and (C5) holds with $\varepsilon_{\rm opt}\to0$
in probability. Then $T\to W_1(P,Q)$ in probability; in the original notation,
$T\to W_1(A^*_\#P,A^*_\#Q)$, the $W_1$ distance under the metric
$\lVert D_*^{-1}(x-y)\rVert$, not $W_1$ in raw return units.

*Proof.* The upper bound is $\Delta(\widehat f)\le W_1$ (Proposition 2.5(b)). The chain in Step 1
with $f_0$ replaced by near-maximisers gives
$\Delta(\widehat f)\ge\sup_{\mathcal F_n}(P-Q)f-\varepsilon_{\rm opt}-2\Gamma_n\to W_1$. Finally,
$T-\Delta(\widehat f)\to0$ by Step 2. $\square$

**Remark 4.4 (how realistic are C3–C5?).**
- *C4* is a population property. It holds if, for some $c$, the $c$-Lipschitz members of
  $\bigcup_n\mathcal F_n$ are dense in bounded 1-Lipschitz functions for uniform convergence on
  compacts. Indeed, $P\neq Q$ gives a bounded 1-Lipschitz $g$ with $(P-Q)g>0$ (bounded Lipschitz
  functions separate measures; Dudley 2002, §11.3). Approximate $g$ within $\epsilon$ on a
  compact $K$ that carries all but $\epsilon$ of the mass of $P+Q$. Outside $K$, use
  $|f-g|(x)\le|f-g|(x_0)+(1+c)\lVert x-x_0\rVert$ and finite first moments. Norm-constrained
  GroupSort networks are dense in $\mathrm{Lip}_1$ on compacts (Anil, Lucas and Grosse 2019). For
  the LeakyReLU or LayerNorm networks of `nets.py` with a soft penalty, we do not prove C4.
- *C3* idealises the gradient penalty (Remark 2.7).
- *C5* is an assumption about stochastic-gradient training that we cannot verify.

The power study (E2) and the Lipschitz diagnostic (Remark 2.8) check these assumptions
empirically. Proposition 4.2 says *what* has to hold for consistency; it does not say that
WGAN-GP training delivers it.

---

## 5. Likelihood connection and the Neyman–Pearson oracle

### 5.1 The Bayes-optimal discriminator and the logit statistic

**Proposition 5.1 (Bayes-optimal logit; Goodfellow et al. 2014, Prop. 1).** Let $\nu=P+Q$,
$p=dP/d\nu$ and $q=dQ/d\nu$. Let $(V,\Lambda)$ satisfy $\Lambda\sim\mathrm{Bernoulli}(\pi)$,
$V\mid\Lambda=1\sim P$ (real) and $V\mid\Lambda=0\sim Q$ (synthetic). Among measurable
$D:\mathbb R^d\to[0,1]$, the binary cross-entropy risk
$-\mathbb E[\Lambda\log D(V)+(1-\Lambda)\log(1-D(V))]$ is minimised by
$D^*=\pi p/(\pi p+(1-\pi)q)$ ($\nu$-a.e.). The minimiser's logit is
$$
\ell^*(v)=\log\frac{p(v)}{q(v)}+\log\frac{\pi}{1-\pi}.
$$

*Proof.* The risk equals $\int-[\pi p\log D+(1-\pi)q\log(1-D)]\,d\nu$. Pointwise in $v$, the
integrand is strictly convex in $D\in[0,1]$ where $p,q>0$, and it is minimised at $D^*$. Where
$q=0<p$ the minimiser is $D=1$, and conversely. $\square$

**Corollary 5.2 (the logit statistic estimates the Jeffreys divergence).** Let
$T_{\rm logit}=P^{\rm ev}\hat\ell-Q^{\rm ev}\hat\ell$ (C2ST with `statistic="logit"`). If
$\hat\ell=\ell^*+\text{const}$, and $\mathrm{KL}(P\Vert Q)$ and $\mathrm{KL}(Q\Vert P)$ are
finite, then
$$
\mathbb E[T_{\rm logit}\mid\mathcal G]=P\log\tfrac pq-Q\log\tfrac pq=\mathrm{KL}(P\Vert Q)+\mathrm{KL}(Q\Vert P)=:J(P,Q),
$$
the Jeffreys divergence (Jeffreys 1946; Kullback and Leibler 1951). The class-prior constant
cancels, so unbalanced training halves do not bias the target.

**Remark 5.3 (scope of Corollary 5.2; the accuracy statistic).**

(i) Unlike the $W_1$ critic, nothing makes $\mathbb E[T_{\rm logit}\mid\mathcal G]$ a *lower*
bound on $J$ for an imperfect $\hat\ell$: rescaling $\hat\ell$ inflates the statistic without
limit. So $T_{\rm logit}$ is a plug-in estimator of $J$, consistent only if $\hat\ell\to\ell^*$ in
$L^1(P)\cap L^1(Q)$. The validity of the test (Proposition 3.1) does not depend on this.

(ii) For balanced eval sets and the decision region $\hat A=\{\hat\ell>0\}$,
$\mathbb E[\mathrm{acc}\mid\mathcal G]=\tfrac12+\tfrac12(P(\hat A)-Q(\hat A))\le\tfrac12(1+\mathrm{TV}(P,Q))$,
with equality for $\hat A=\{p>q\}$. The accuracy C2ST (Lopez-Paz and Oquab 2017; Kim et al. 2021)
therefore estimates a lower bound on $\mathrm{TV}$, exactly as the critic estimates a lower bound
on $W_1$.

(iii) Cheng and Cloninger (2022) study the logit statistic. In E2, `C2ST.fit_logit` lets us
compare $\hat\ell$ with $\ell^*$ on fresh samples (correlation and mean squared error). This
measures how close the learned statistic is to the likelihood-ratio statistic.

### 5.2 The Neyman–Pearson oracle bounds every two-sample test

**Fact 5.4 (Neyman–Pearson lemma; Neyman and Pearson 1933; Lehmann and Romano 2005, Thm 3.2.1).**
Take a simple null and a simple alternative with densities $f_0,f_1$. The test
$\phi^*=\mathbf 1\{f_1/f_0>k\}+\gamma\,\mathbf 1\{f_1/f_0=k\}$, with $k,\gamma$ chosen so that
$\mathbb E_0\phi^*=\alpha$, is most powerful among all (possibly randomised) tests with
$\mathbb E_0\phi\le\alpha$.

Write $\beta_{\rm NP}(F\to F';n,\alpha)$ for the power of this most powerful level-$\alpha$ test
of $F^{\otimes n}$ against $F'^{\otimes n}$.

**Proposition 5.5 (a two-sample test cannot beat the oracle).** Let $P,Q$ have densities with
respect to a common $\sigma$-finite measure. Let $\phi(X_{1:n},Y_{1:m},U)\in[0,1]$ be any test,
where the auxiliary randomness $U$ (splits, critic training, permutations) is independent of the
data.

(a) If $\mathbb E_{P^{\otimes n}\otimes P^{\otimes m}}\phi\le\alpha$, then
$\mathbb E_{P^{\otimes n}\otimes Q^{\otimes m}}\phi\le\beta_{\rm NP}(P\to Q;m,\alpha)$.

(b) If $\mathbb E_{Q^{\otimes n}\otimes Q^{\otimes m}}\phi\le\alpha$, then
$\mathbb E_{P^{\otimes n}\otimes Q^{\otimes m}}\phi\le\beta_{\rm NP}(Q\to P;n,\alpha)$.

So a two-sample test of exact level $\alpha$ (for example the permutation test of
Proposition 3.1) has power at most
$\min\{\beta_{\rm NP}(P\to Q;m,\alpha),\beta_{\rm NP}(Q\to P;n,\alpha)\}$ at $(P,Q)$.

*Proof.* (a) Let $\bar\phi(x,y)=\mathbb E[\phi\mid X=x,Y=y]$, a randomised test of the same size
and power. Consider the simple problem $H_0':(X,Y)\sim P^{\otimes n}\otimes P^{\otimes m}$
against $H_1':(X,Y)\sim P^{\otimes n}\otimes Q^{\otimes m}$. The likelihood ratio is
$\prod_jq(Y_j)/p(Y_j)$, because the $X$-factors cancel. By Fact 5.4 the most powerful
level-$\alpha$ test depends on $Y$ only, and its power is $\beta_{\rm NP}(P\to Q;m,\alpha)$. Since
$\bar\phi$ has level $\alpha$ for $H_0'$, its power is at most that. (b) is the same argument
with the roles of the samples exchanged. $\square$

In what sense the two-sample test "cannot beat the oracle": the oracle knows both densities and
tests one simple hypothesis against one simple alternative. The two-sample test sees only
samples, and its level must hold at *every* null point $P=Q$, in particular at
$P\otimes P$. In bound (a), the $n$ real windows are worthless to the oracle, which already knows
$P$. The lr_oracle_test of DESIGN §3.8 applies $\phi^*$ to the sample whose law varies with
$\delta$, with a Monte Carlo critical value. As a Monte Carlo test with p-value
$(1+\#)/(1+B)$, it has exact level $\alpha$, and its power tends to $\beta_{\rm NP}$ as
$B\to\infty$ (Barnard 1963; Dufour 2006). Plotted against $\delta$, it is the power envelope.

### 5.3 How far below the oracle? Local asymptotics

**Proposition 5.6 (local power of fixed-critic tests).** Let $\{Q_\theta\}_{\theta\in[0,\theta_0)}$
with $Q_0=P$ be differentiable in quadratic mean at $0$, with score $h\in L^2(P)$ and
$I=Ph^2>0$. Let $m\to\infty$ with $n/m\to\kappa\in(0,\infty)$, eval sizes $n_e=\lfloor\rho n\rfloor$
and $m_e=\lfloor\rho m\rfloor$ for some $\rho\in(0,1]$, and local alternatives
$\theta_m=\tau/\sqrt m$ with $\tau>0$.

(a) *Oracle.* $\beta_{\rm NP}(P\to Q_{\theta_m};m,\alpha)\to1-\Phi(z_{1-\alpha}-\tau\sqrt I)$.

(b) *Fixed critic.* Let $f\in L^2(P)$ be fixed, with $\sigma_f^2=\operatorname{Var}_Pf>0$ and
orientation $r_f:=-\operatorname{Corr}_P(f,h)\ge0$ (the critic scores real windows higher). The
test that rejects when $T/\sqrt{s_X^2/n_e+s_Y^2/m_e}>z_{1-\alpha}$, or the permutation test with
the same $f$, has power under $P^{\otimes n}\otimes Q_{\theta_m}^{\otimes m}$ tending to
$$
1-\Phi\Big(z_{1-\alpha}-\tau\sqrt I\;r_f\sqrt{\tfrac{\rho\kappa}{1+\kappa}}\Big).
$$

(c) With $\kappa=1$ and $\rho=1/2$ (DESIGN defaults), the noncentrality of any fixed-critic test
is at most **half** the oracle's. At the alternative where the oracle has power $0.80$
($\alpha=0.05$, noncentrality $1.645+0.842=2.487$), every fixed-critic test has asymptotic power at
most $1-\Phi(1.645-1.244)\approx0.34$.

*Proof sketch.* Under $\theta=0$, the CLT and local asymptotic normality (van der Vaart 1998,
Thm 7.2) give joint asymptotic normality of $\sqrt mT$ and of the log-likelihood ratio
$\log L_m=\tau m^{-1/2}\sum_{j\le m}h(Y_j)-\tau^2I/2+o_P(1)$, with
$$
m\operatorname{Var}T\to\sigma_f^2\frac{1+1/\kappa}{\rho},\qquad
\operatorname{Cov}\big(\sqrt mT,\log L_m\big)\to-\tau\operatorname{Cov}_P(f,h)=\tau r_f\sigma_f\sqrt I .
$$
By Le Cam's third lemma (van der Vaart 1998, Ex. 6.7), under $\theta_m$,
$\sqrt mT\Rightarrow N\big(\tau r_f\sigma_f\sqrt I,\ \sigma_f^2(1+1/\kappa)/\rho\big)$. The
variance estimators are consistent under $\theta=0$, hence under $\theta_m$ by contiguity, and
studentising gives (b). The permutation critical value converges to $z_{1-\alpha}$ under
$\theta=0$ (Hoeffding 1952), hence under $\theta_m$ by contiguity (Le Cam's first lemma). For (a),
$\log L_m\Rightarrow N(\mp\tau^2I/2,\tau^2I)$ under the null and the alternative, respectively.
(c) uses $r_f\le1$ and $\sqrt{1/4}=1/2$. $\square$

The factor $\sqrt{\rho\kappa/(1+\kappa)}$ separates the two costs of not knowing the densities:
- the two-sample cost, $\sqrt{\kappa/(1+\kappa)}$, which tends to 1 as $\kappa\to\infty$, i.e.
  when $P$ is effectively known;
- the splitting cost, $\sqrt\rho$.

The factor $r_f$ is the price of using a sub-optimal direction. With a learned critic, the
argument applies conditionally on the training half with $f=\widehat f$. Because $r_f\le1$, the
bound in (c) also holds for learned critics. Making this rigorous needs a CLT uniform over the
critic class, e.g. $\mathrm{Lip}_c$ with $P\in\mathcal P_{2+\delta}$; this is a sketch.

**Example 5.7 (volatility level, E2 family (a)).** Let $P=N(0,\sigma^2I_L)$ (i.i.d. Gaussian
returns) and $Q_\theta=N(0,(1+\theta)^2\sigma^2I_L)$. The score is
$h(x)\propto\lVert x\rVert^2-L\sigma^2$. For every scale factor, $f(x)=\lVert x\rVert$ is an optimal
Kantorovich potential: the scaling coupling $Y=sX$ gives
$W_1\le|s-1|\,\mathbb E\lVert X\rVert$, and $\lVert x\rVert$ attains it. Its local efficiency is
$r_f^2=\operatorname{Corr}(\chi_L,\chi_L^2)^2=0.876,\ 0.957,\ 0.988$ for $L=1,5,20$. (By
symmetry the population standardisation of Remark 2.3 is a multiple of the identity here, so the
standardised problem is the same up to scale, and $r_f$ is unchanged.) So for
volatility-level deviations, the population $W_1$ critic is nearly as efficient as the
likelihood-ratio direction.

**Heuristic 5.8 (LR critic versus $W_1$ critic).** By Proposition 5.1, the population logit
critic is $\log(p/q_\theta)\approx-\theta h$, the locally optimal direction ($r=1$). The
population $W_1$ critic is the Kantorovich potential, which is in general not proportional to
$h$ ($r<1$). If both are learned well, the LR critic should therefore be locally at least as
powerful. The Lipschitz regularisation of the $W_1$ critic may instead reduce estimation variance
in small samples. Which effect dominates at E2's sample sizes is an empirical question (RQ2); we
make no theoretical claim.

### 5.4 Remarks on implementing the oracle

1. *i.i.d. families* (GBM, Merton, Student-$t$). The window density is the product of daily
   densities, so the oracle is exact.
2. *GARCH windows.* The window density is
   $p(x_{1:L})=\int\prod_tp(x_t\mid\sigma_t^2)\,\Pi(d\sigma_1^2)$, where $\sigma_t^2$ is the
   filtered variance started at $\sigma_1^2$ and $\Pi$ is the law of the variance at the start
   of the window: the stationary law, or the law induced by the simulation's starting state. The
   integral has no closed form. Two options:
   - Monte Carlo over $M$ draws of $\sigma_1^2\sim\Pi$ taken from a long simulated path. This
     is unbiased for $p$, with a log-bias of order $1/M$.
   - Fixing $\sigma_1^2$ at the unconditional variance. This gives a quasi-likelihood: the test
     keeps exact level with a Monte Carlo critical value, but it is **not** NP-optimal, so it is
     not an upper bound.
3. *Dependent windows.* If the windows of a sample are consecutive pieces of one simulated path,
   the joint density is not the product of window densities. The NP statistic is then the
   full-path log-likelihood ratio, which is exact for GARCH given $\sigma_1^2$. A product of
   window densities still gives a valid Monte Carlo test, but not the most powerful one.

Recommendation (Implementation note 11.4): in E2, simulate *independent* windows, each from its
own burned-in path. Then A1 holds exactly for every test, and the oracle is the product of window
densities, integrated over $\sigma_1^2$ for GARCH.

### 5.5 Likelihood benchmarks rank by KL; critics rank by $W_1$ (RQ4)

Whenever finite, the expected out-of-sample log-likelihood of a model $G$ is
$\mathbb E_P\log g=-\mathrm{Ent}(P)-\mathrm{KL}(P\Vert G)$, where $\mathrm{Ent}(P)$ is the
differential entropy of $P$. For
conditional densities of stationary ergodic processes, it is the corresponding relative entropy
rate. So ranking models by predictive log-likelihood, and the Vuong (1989) test, rank by
Kullback–Leibler divergence. The critic ranks by (approximate lower bounds on) $W_1$ of the
window law. Both divergences vanish if and only if the laws are equal, but they can rank
misspecified models differently.

*Numerical example* (computed by quadrature): let $P$ be Student-$t_4$ scaled to unit variance.
- $G_1=N(0,1)$ (right variance, wrong tails):
  $\mathrm{KL}(P\Vert G_1)\approx0.084$, $W_1(P,G_1)\approx0.139$.
- $G_2$ = the same $t_4$ scaled by $1.4$ (right tails, wrong scale):
  $\mathrm{KL}(P\Vert G_2)\approx0.060$, $W_1(P,G_2)=0.4\,\mathbb E_P|X|\approx0.283$.

The likelihood prefers $G_2$ and $W_1$ prefers $G_1$. Disagreement in RQ4 therefore shows that
the two methods measure different divergences, not that either is broken. Agreement is expected
when one model is close to $P$ in every sense.

---

## 6. The relative test

### 6.1 Setting and hypotheses

There are three samples:
- $R_{\rm new}$: $X_1,\dots,X_n$ with law $P_{\rm new}$;
- the synthetic $G$: $Y_1,\dots,Y_m$ with law $Q_G$;
- $R_{\rm past}$: $Z_1,\dots,Z_k$ with law $P_{\rm past}$.

Each is split into train and eval halves, with eval sizes $n_e,m_e,k_e$. Two critics are trained
*separately*, with independent training randomness and the same architecture and
hyperparameters: $\widehat f_G$ on $(X^{\rm tr},Y^{\rm tr})$ and $\widehat f_P$ on
$(X^{\rm tr},Z^{\rm tr})$. Both critics share the $R_{\rm new}$ training half and are evaluated on
the shared $R_{\rm new}$ eval half (DESIGN §3.7). The statistic is $\Delta=T_G-T_P$:
$$
\Delta=\frac1{n_e}\sum_{i\in{\rm ev}}\big[\widehat f_G(X_i)-\widehat f_P(X_i)\big]-\frac1{m_e}\sum_{j\in{\rm ev}}\widehat f_G(Y_j)+\frac1{k_e}\sum_{l\in{\rm ev}}\widehat f_P(Z_l).
\tag{6.1}
$$
Its conditional target is $\delta:=d_G-d_P$, where $d_G=P_{\rm new}\widehat f_G-Q_G\widehat f_G$ and
$d_P=P_{\rm new}\widehat f_P-P_{\rm past}\widehat f_P$ are the critic-induced discrepancies.

**Definition 6.1.** The *critic relative null*, conditional on $\mathcal G$ (now the training
halves of all three samples and both critics' randomness), is
$H_0^{\rm rel}:\ \delta\le0$ against $H_1^{\rm rel}:\ \delta>0$. Its idealised $W_1$ version is
$H_0^{W}:\ W_1(Q_G,P_{\rm new})\le W_1(P_{\rm past},P_{\rm new})$.

### 6.2 Asymptotic validity of the paired bootstrap

**Assumption R.**
- (R1) Conditionally on $\mathcal G$, the three eval samples are mutually independent, and each
  is i.i.d. from $P_{\rm new}$, $Q_G$ and $P_{\rm past}$, respectively.
- (R2) $n_e,m_e,k_e\to\infty$ proportionally ($N:=n_e+m_e+k_e$). Let
  $$
  \sigma_N^2=\frac{\sigma_A^2}{n_e}+\frac{\sigma_B^2}{m_e}+\frac{\sigma_C^2}{k_e},\qquad
  \sigma_A^2=\operatorname{Var}_{P_{\rm new}}(\widehat f_G-\widehat f_P),\ \ \sigma_B^2=\operatorname{Var}_{Q_G}\widehat f_G,\ \ \sigma_C^2=\operatorname{Var}_{P_{\rm past}}\widehat f_P .
  $$
  We require $N\sigma_N^2$ to be bounded away from 0, and the $(2+\epsilon)$-th moments of the
  three summands to be bounded in probability (Lyapunov). Both hold if the critics are fixed
  (training data held fixed) with nonconstant values and finite $(2+\epsilon)$ moments, or if the
  critics are $c$-Lipschitz, the laws are in $\mathcal P_{2+\epsilon}$ and the variances are
  bounded below.
- (R3) The bootstrap draws $X^{{\rm ev}*}$, $Y^{{\rm ev}*}$ and $Z^{{\rm ev}*}$ by independent
  resampling with replacement. $\Delta^*$ uses the **same** $X^{{\rm ev}*}$ draw for both critics
  (paired resampling).

**Proposition 6.2.** Under R1–R3, conditionally on $\mathcal G$:
1. $(\Delta-\delta)/\sigma_N\Rightarrow N(0,1)$;
2. $\sup_x\big|\mathbb P^*(\Delta^*-\Delta\le x)-\mathbb P(\Delta-\delta\le x\mid\mathcal G)\big|\to0$
   in probability;
3. the test that rejects when
   $p^*=(1+\#\{b:\Delta^*_b-\Delta\ge\Delta\})/(1+B)\le\alpha$ satisfies
   $\mathbb P(\text{reject}\mid\mathcal G)=1-\Phi(z_{1-\alpha}-\delta/\sigma_N)+o_P(1)$ as
   $B\to\infty$. This is at most $\alpha+o_P(1)$ on $\{\delta\le0\}$. For fixed critics, the
   limit is $0$, $\alpha$ or $1$ according as $\delta<0$, $\delta=0$ or $\delta>0$;
4. the percentile interval $[q_{0.025}(\Delta^*),q_{0.975}(\Delta^*)]$, with $q_\beta(\Delta^*)$
   the bootstrap quantiles of $\Delta^*$ (this is what `relative.py` reports as `extra["ci"]` by
   default, and always as `extra["ci_boot"]`),
   has asymptotic coverage $0.95$ for $\delta$. So does the basic interval
   $[2\Delta-q_{0.975}(\Delta^*),2\Delta-q_{0.025}(\Delta^*)]$; the two are asymptotically
   equivalent because the limit in (2) is a centred normal, which is symmetric.

*Proof sketch.* (1) $\Delta-\delta$ is a sum of three independent centred sample means; apply
the Lyapunov CLT.
(2) Each bootstrap mean satisfies a conditional CLT with the empirical variance (Bickel and
Freedman 1981; Singh 1981; van der Vaart 1998, Thm 23.4). For triangular arrays, the Lyapunov
condition for the bootstrap draws holds in probability by the law of large numbers for the
empirical $(2+\epsilon)$-th moments. The three bootstrap means are independent given the data.
Pólya's theorem gives uniform convergence because the limit is continuous.
*Pairing:* the $X$-term of $\Delta^*$ is the bootstrap mean of the paired differences
$D_i=\widehat f_G(X_i)-\widehat f_P(X_i)$. Its bootstrap variance estimates
$\sigma_A^2=\operatorname{Var}\widehat f_G+\operatorname{Var}\widehat f_P-2\operatorname{Cov}(\widehat f_G,\widehat f_P)$.
Resampling the two critics' values independently would ignore the covariance, which is typically
positive (both critics score "$R_{\rm new}$-likeness"), and make the test conservative.
(3) By (2), $q^*_{1-\alpha}/\sigma_N\to z_{1-\alpha}$, and
$\mathbb P(\Delta>q^*_{1-\alpha}\mid\mathcal G)=\mathbb P\big((\Delta-\delta)/\sigma_N>q^*_{1-\alpha}/\sigma_N-\delta/\sigma_N\big)$.
(4) Let $q^*_\beta$ be the quantiles of $\Delta^*-\Delta$, so $q_\beta(\Delta^*)=\Delta+q^*_\beta$.
By (2), $q^*_\beta/\sigma_N\to z_\beta$ in probability, and $z_{0.025}=-z_{0.975}$. The percentile
interval contains $\delta$ iff
$-q^*_{0.975}\le\Delta-\delta\le-q^*_{0.025}$, i.e. iff
$(\Delta-\delta)/\sigma_N\in[-z_{0.975},z_{0.975}]+o_P(1)$, an event of probability
$0.95+o(1)$ by (1). The basic interval contains $\delta$ iff
$(\Delta-\delta)/\sigma_N\in[z_{0.025},z_{0.975}]+o_P(1)$, the same event. $\square$

The uncentred variant $p=\#\{b:\Delta_b^*\le0\}/B$ is asymptotically equivalent, by symmetry of
the normal limit. The centred version is better founded (Hall 1992).

### 6.3 What the test does and does not guarantee

**Remark 6.3 (a data-dependent null).** The critics are trained separately, so
$H_0^{\rm rel}$ is a statement about two trained critics. By Proposition 2.5,
$d_G=\mathrm{Lip}(\widehat f_G)W_1(Q_G,P_{\rm new})-s_G$ and
$d_P=\mathrm{Lip}(\widehat f_P)W_1(P_{\rm past},P_{\rm new})-s_P$, with unknown slacks
$s_G,s_P\ge0$. Hence $H_0^W$ can hold while $\delta>0$ (if $s_P>s_G$): the test may reject even
though $G$ is closer in $W_1$. Conversely, it may fail to reject when $G$ is farther. *With
respect to $H_0^W$, the relative test is neither exact nor conservative.* It is "fair" only to
the extent that the two critic problems are equally hard: same architecture, hyperparameters,
training budget and **sample sizes** ($m_t=k_t$, $m_e=k_e$). Otherwise a larger training sample
yields a tighter critic and a larger $d$, confounding sample size with distance (Implementation
note 11.3). Equal resources are necessary but not sufficient: Proposition 6.8 shows that at the
boundary point $Q_G=P_{\rm past}$, where $H_0^W$ holds with equality, the test is strictly
anti-conservative even with equal resources, and the excess does not vanish as the sample sizes
grow.

**Remark 6.4 (guarantee).** Under R1–R3, a rejection means that the critic family, given equal
resources, separates $G$ from $R_{\rm new}$ by more than it separates $R_{\rm past}$ from
$R_{\rm new}$: "$G$ is detectably worse than history". Non-rejection is **not** evidence that
$G$ is at least as good; hence the power requirement in Section 8. The test says nothing directly
about portfolio utility (Section 9).

**Remark 6.5 (relation to Bounliphone et al. 2016).** Bounliphone et al. test
$H_0:\mathrm{MMD}(P_x,P_y)\le\mathrm{MMD}(P_x,P_z)$ for a fixed kernel, with the statistic
$\mathrm{MMD}_u^2(X,Y)-\mathrm{MMD}_u^2(X,Z)$: two dependent U-statistics that share $X$, with
joint asymptotic normality and an explicit covariance. Their null concerns a genuine metric.
Their CLT needs both candidates to differ from the reference, because the MMD U-statistic is
degenerate at equality (Gretton et al. 2012).

The critic version replaces the fixed witness function with learned witnesses evaluated on
held-out data. This has two consequences:
- the statistic is a difference of means of fixed functions, so the joint CLT is elementary and
  does not degenerate when laws coincide;
- the price is the data-dependent null of Remark 6.3.

The MMD version of `relative_test` follows Bounliphone et al. (DESIGN §3.7).

**Remark 6.6 (dependence between $G$ and $R_{\rm past}$).** $G$ is calibrated on $R_{\rm past}$,
so R1 is an idealisation.
- For parametric generators, each $R_{\rm past}$ window affects $\hat\theta$ by $O(1/n)$.
  **Heuristic:** this is asymptotically negligible; a proof would use the influence function of
  $\hat\theta$.
- For the stationary bootstrap generator, synthetic windows are partly copies of $R_{\rm past}$
  windows, and R1 fails. Because both critics score "$R_{\rm new}$-likeness", the copied terms are
  **plausibly** positively correlated, which makes the independent bootstrap conservative
  (heuristic).

The stationary bootstrap generator resamples blocks of $R_{\rm past}$, so $Q_G\approx P_{\rm past}$:
it sits near the boundary of the population null. There, by Proposition 6.8, the probability of
a BAD verdict is *above* $\alpha$, not equal to it, so its E3 verdicts are **not** a calibration
check of the relative test. Its rejection frequency over tickers and seeds is at most a rough
empirical indication of the boundary size, contaminated by the failure of R1.

**Remark 6.7 (serial dependence).** The $R_{\rm new}$ eval windows form a dependent time series.
Replace the i.i.d. resampling of $X^{\rm ev}$ with a block bootstrap of the time-ordered paired
differences. The moving block bootstrap (Künsch 1989) and the stationary bootstrap (Politis and
Romano 1994) are valid for means of strongly mixing sequences (Lahiri 2003). Alternatively, use a
HAC variance with a normal p-value (Section 7). In pooled panels, both critics are trained on the
same $R_{\rm new}$ training half; with a ticker split on common dates, both $d_G$ and $d_P$
inherit the bias of Proposition 7.5, which cancels in $\delta$ only to the extent that both
critics exploit the realised market path equally. The time-block split of Section 7.4 should be
used for the relative test as well.

### 6.4 Size at the boundary $Q_G=P_{\rm past}$

Proposition 6.2 controls the rejection probability conditionally on $\mathcal G$, and only on the
$\mathcal G$-measurable event $\{\delta\le0\}$. What a user needs is the *unconditional*
probability of a BAD verdict for a generator that is as good as history. The most natural such
generator has the same window law as the past, $Q_G=P_{\rm past}$. It lies on the boundary of
$H_0^W$, and of the analogous null for any divergence, since both sides are then equal.

**Proposition 6.8 (unconditional size; anti-conservative at the boundary).** Assume R1–R3 along
a sequence of experiments, let $B\to\infty$, and write $z=z_{1-\alpha}$ with $\alpha<\tfrac12$.

(a) $\mathbb P(\text{reject})=\mathbb E\big[1-\Phi(z-\delta/\sigma_N)\big]+o(1)$.

(b) Suppose in addition that $Q_G=P_{\rm past}$, $m_t=k_t$ and $m_e=k_e$; that
$\widehat f_G=\Psi(X^{\rm tr},Y^{\rm tr},\xi_G)$ and $\widehat f_P=\Psi(X^{\rm tr},Z^{\rm tr},\xi_P)$
for one measurable training map $\Psi$, with $\xi_G,\xi_P$ i.i.d. and independent of the data;
and that the joint law of the data and the index sets is invariant under exchanging the roles of
$(Y,\xi_G)$ and $(Z,\xi_P)$ (true for random splits of equal sizes, and for group splits with a
shared label namespace when $Y$ and $Z$ carry the same groups). Then
$(\delta,\sigma_N)\overset d=(-\delta,\sigma_N)$, and
$$
\mathbb P(\text{reject})=\mathbb E\,\bar g\big(|\delta|/\sigma_N\big)+o(1),\qquad
\bar g(x):=1-\tfrac12\big[\Phi(z-x)+\Phi(z+x)\big].
$$
The function $\bar g$ is continuous and strictly increasing on $[0,\infty)$, with $\bar g(0)=\alpha$
and $\bar g(\infty)=\tfrac12$. Hence $\mathbb P(\text{reject})\ge\alpha+o(1)$. If moreover
$\mathbb P(|\delta|>\varepsilon\sigma_N)\ge\eta$ for some $\varepsilon,\eta>0$ and all large $N$,
then
$$
\liminf\ \mathbb P(\text{reject})\ \ge\ \alpha+\eta\,\big[\bar g(\varepsilon)-\alpha\big]\ >\ \alpha .
$$
So the size at this boundary point tends to $\alpha$ only if $\delta=o_P(\sigma_N)$, i.e. only if
the critic-training noise in $d_G-d_P$ is negligible relative to the evaluation noise
$\sigma_N\asymp N^{-1/2}$.

(c) In particular, if the training halves stay fixed while the eval sizes grow (the fixed-critic
case of Proposition 6.2(3)), then
$\mathbb P(\text{reject})\to\mathbb P(\delta>0)+\alpha\,\mathbb P(\delta=0)=\tfrac12-(\tfrac12-\alpha)\,\mathbb P(\delta=0)$,
which is $\tfrac12$ when $\delta$ has no atom at 0.

*Proof.* (a) By Proposition 6.2(3),
$\mathbb P(\text{reject}\mid\mathcal G)-[1-\Phi(z-\delta/\sigma_N)]\to0$ in probability. The
difference is bounded by 1, so its expectation tends to 0 (dominated convergence).
(b) Let $\omega=(X,Y,Z,\xi_G,\xi_P,\text{index sets})$, and let $\omega'$ be $\omega$ with
$(Y,\xi_G)$ and $(Z,\xi_P)$, and their index sets, exchanged. By assumption
$\omega'\overset d=\omega$. The exchange maps $\widehat f_G$ to $\widehat f_P$ and conversely;
because $Q_G=P_{\rm past}$, it therefore exchanges $d_G$ and $d_P$, leaves
$\sigma_A^2=\operatorname{Var}_{P_{\rm new}}(\widehat f_G-\widehat f_P)$ unchanged, and exchanges
$\sigma_B^2/m_e$ and $\sigma_C^2/k_e$ (using $m_e=k_e$). Hence
$(\delta,\sigma_N)(\omega')=(-\delta,\sigma_N)(\omega)$, and the symmetry follows. For a symmetric
variable $V=\delta/\sigma_N$,
$\mathbb E[1-\Phi(z-V)]=\mathbb E\,\tfrac12\{[1-\Phi(z-|V|)]+[1-\Phi(z+|V|)]\}=\mathbb E\,\bar g(|V|)$.
Next, $\bar g'(x)=\tfrac12[\varphi(z-x)-\varphi(z+x)]>0$ for $x>0$, because $z>0$ gives
$|z-x|<z+x$. The bounds follow from $\bar g\ge\alpha$ on $[0,\infty)$ and $\bar g\ge\bar g(\varepsilon)$
on $[\varepsilon,\infty)$.
(c) Now $\delta$ does not depend on $N$ and $\sigma_N\to0$, so
$1-\Phi(z-\delta/\sigma_N)\to\mathbf 1\{\delta>0\}+\alpha\mathbf 1\{\delta=0\}$; apply (a), dominated
convergence and $\mathbb P(\delta>0)=\mathbb P(\delta<0)$. $\square$

The proof uses only $\mathbb P(\text{reject}\mid\mathcal G)=1-\Phi(z-\delta/\sigma_N)+o_P(1)$ and the
symmetry of $(\delta,\sigma_N)$. So Proposition 6.8 applies to any conditionally valid test of
$H_0^{\rm rel}$ built on $\Delta$, including the HAC-studentised version of Remark 6.7
(`relative.py`, `inference="hac"`), with $\sigma_N$ the corresponding long-run standard deviation.

**Remark 6.9 (consequences and remedies).**

- *What the test controls.* The paired-bootstrap test is valid for the conditional null
  $\delta\le0$, a statement about two trained critics. It is not a level-$\alpha$ test of "$G$ is
  as good as history". For a generator equal in law to the past, the false-BAD rate exceeds
  $\alpha$, and the excess does not vanish as the samples grow proportionally: $\delta$ is the
  difference of two independent replicates of a training-dependent quantity, so it fluctuates at
  the scale of critic training, which is the scale of $\sigma_N$ or larger. Nothing in WGAN-GP
  training (stochastic gradients, non-convexity, early stopping) makes $\delta=o_P(\sigma_N)$. A
  pilot simulation (Implementation note 11.3) gives a rejection rate of about $10\%$ at nominal
  $5\%$, matching the prediction $\mathbb E[1-\Phi(z-\delta/\sigma_N)]$ of part (a).
- *Remedies*, which can be combined:
  1. *Report the boundary size.* In the power-check design $\mathcal C_{\rm ref}$
     (Definition 8.1), also simulate a generator equal in law to $R_{\rm past}$ (an independent
     draw from the reference law), and report the estimated boundary size $\hat\alpha_{\rm bd}$,
     with a Clopper–Pearson interval, next to every verdict. A size-corrected nominal level
     $\alpha'$ with $\hat\alpha_{\rm bd}(\alpha')=\alpha$ can be calibrated in the same simulation;
     it transfers to the data only to the extent that $\mathcal C_{\rm ref}$ reproduces the
     critic-training behaviour (**heuristic**).
  2. *Shrink the training noise in $\delta$.* Use common random numbers for the two critics (same
     initialisation seed and same mini-batch index stream), average $\Delta$ over an ensemble of
     seeds, or cross-fit (exchange train and eval halves and average). These keep the symmetry of
     part (b) and reduce $\operatorname{Var}\delta$, hence the excess, but cannot remove the part
     due to $Y^{\rm tr}\neq Z^{\rm tr}$.
  3. *Add the training variability to the variance.* Retrain both critics on $K$ resampled
     training halves with fresh seeds, estimate $\operatorname{Var}\delta$ from the replicates,
     and studentise $\Delta$ by $(\hat\sigma_N^2+\widehat{\operatorname{Var}}\,\delta)^{1/2}$. At
     the symmetric boundary $\mathbb E\delta=0$, so this targets the unconditional null; its
     validity needs $\delta$ to be approximately normal (**heuristic**). Cost: $2K$ extra critic
     fits.
  4. *A shared critic.* Train one critic $\widehat f$ to separate the $R_{\rm new}$ training
     windows from the pooled $G$ and $R_{\rm past}$ training windows (in equal numbers), and
     compare its means on the $R_{\rm past}$ and $G$ eval windows,
     $\Delta_{\rm sh}=Z^{\rm ev}\widehat f-Y^{\rm ev}\widehat f$ (the $R_{\rm new}$ terms cancel).
     At $Q_G=P_{\rm past}$ the pooled eval windows of $G$ and $R_{\rm past}$ are i.i.d. given
     $\mathcal G$, so the permutation test of Section 3 applied to them has **exact** level at this
     boundary point, for every critic (Proposition 3.1). It targets a different null,
     $Q_G\widehat f\ge P_{\rm past}\widehat f$ for a critic that scores "$R_{\rm new}$-likeness";
     for unequal laws the studentised permutation of Remark 3.5(b) gives asymptotic validity.
  5. *At a minimum*, add to E2/E5 a table of the rejection rate of the relative test at
     $Q_G=P_{\rm past}$, at the E3 sample sizes, with the prediction of part (a) computed from
     fresh draws.

---

## 7. Serial and cross-sectional dependence

### 7.1 Exchangeability fails for real windows

The real windows of one ticker, $(W_{kL})_k$, form a stationary but dependent sequence: volatility
clustering makes window-level features such as realised variance persistent. Synthetic windows
are i.i.d. if each comes from an independent path, and dependent if they come from one long path.
Under $H_0$ (equal window marginals), the pooled eval vector is then not exchangeable, and
Proposition 3.1 does not apply.

**Heuristic 7.1 (size distortion of the naive permutation test).** Let $a_k=\widehat f(X_{(k)})$
be the real eval values in time order. Suppose they have variance $\sigma^2$ and long-run variance
$\omega_a^2=\sigma^2(1+2\sum_{h\ge1}\rho_h)$, and that the synthetic values are i.i.d. with
variance $\sigma^2$, with $n_e=m_e$. Asymptotically, the permutation distribution uses variance
$2\sigma^2/n_e$, while the true variance is $(\omega_a^2+\sigma^2)/n_e$. The rejection rate
therefore tends to
$$
1-\Phi\Big(z_{1-\alpha}\sqrt{2/(2+2\textstyle\sum_h\rho_h)}\Big).
$$
For AR(1)-type persistence $\rho_h=\phi^h$, the long-run factor is
$1+2\sum_h\rho_h=(1+\phi)/(1-\phi)$, and the nominal 5% test rejects about 12% of the time for
$\phi=0.5$ and 17% for $\phi=0.67$. (The formula follows from the CLT; the AR(1) form of $\rho_h$
is an illustration.)

### 7.2 HAC-studentised statistic

Let the real eval values $a_1,\dots,a_{n_e}$ and the synthetic eval values $b_1,\dots,b_{m_e}$ be
ordered in time (`order_x`, `order_y`). Define
$$
z_{\rm hac}=\frac{\bar a-\bar b}{\sqrt{\hat\omega_a^2/n_e+\hat\omega_b^2/m_e}},\qquad
\hat\omega^2=\hat\gamma(0)+2\sum_{h=1}^{H}\Big(1-\frac h{H+1}\Big)\hat\gamma(h),
$$
with the Bartlett kernel and $\hat\gamma(h)=n^{-1}\sum_{k\le n-h}(a_k-\bar a)(a_{k+h}-\bar a)$
(Newey and West 1987). The default bandwidth is $H=\lfloor4(n/100)^{2/9}\rfloor$ (Newey and West
1994). The p-value is $p_{\rm hac}=1-\Phi(z_{\rm hac})$ (DESIGN: `extra["z_hac"]`,
`extra["p_hac"]`).

*Time unit.* The index $k$ counts eval windows in time order, so $h$ and $H$ are in units of
windows (20 trading days at $L=20$), not days. In the implementation (`hac_var_mean`), integer
`order_x`/`order_y` values are placed on a dense grid and lags are counted in grid units.
`order_x`/`order_y` must therefore be **consecutive window indices** (start day divided by $L$)
or ranks of distinct dates, never trading-day indices. With trading-day indices of
non-overlapping windows, consecutive windows are $L$ grid units apart, every lagged product with
$h\le H<L$ vanishes, and $\hat\omega^2$ silently collapses to the (date-clustered) i.i.d.
variance (Implementation note 11.2).

**Assumption D (fixed critic $f$; the weak-dependence regime).**
- (D1) The real window sequence $(X_{(k)})_k$ and the synthetic sequence $(Y_{(k)})_k$ are
  strictly stationary with marginals $P$ and $Q$, independent of each other, and strongly
  ($\alpha$-)mixing with $\sum_h\alpha(h)^{\epsilon/(2+\epsilon)}<\infty$. Moreover
  $\mathbb E|f(X)|^{2+\epsilon}<\infty$ and $\mathbb E|f(Y)|^{2+\epsilon}<\infty$ for some
  $\epsilon>0$. For Lipschitz $f$, the moment condition follows from finite $(2+\epsilon)$-th
  moments of returns.
- (D2) The long-run variances $\omega_a^2,\omega_b^2$ (absolutely convergent series under D1)
  are positive.
- (D3) $H\to\infty$ with $H=o(n^{1/2})$, together with the fourth-cumulant summability of Andrews
  (1991, Prop. 1); or the near-epoch-dependence conditions of de Jong and Davidson (2000) with
  $H=o(n)$.
- (D4) $n_e/(n_e+m_e)\to\kappa\in(0,1)$.

**Proposition 7.2 (asymptotic validity of the HAC test).** Under D1–D4 and
$H_0':Pf=Qf$, $z_{\rm hac}\Rightarrow N(0,1)$, so the test that rejects when
$z_{\rm hac}>z_{1-\alpha}$ has asymptotic level $\alpha$. $H_0$ (equal window marginals) implies
$H_0'$ for every $f$, so the test is asymptotically valid for $H_0$ whatever the critic. The
dependence across windows is a nuisance that $\hat\omega^2$ absorbs.

*Proof.* The CLT for strongly mixing stationary sequences (Ibragimov 1962; see also Herrndorf
1984; Doukhan 1994; Bradley 2005) gives
$\sqrt{n_e}(\bar a-Pf)\Rightarrow N(0,\omega_a^2)$, and likewise for $\bar b$. The two sequences
are independent, so the limits are jointly normal. $\hat\omega^2$ is consistent under D3 (Andrews
1991; de Jong and Davidson 2000). Slutsky's lemma completes the proof. $\square$

**Fact 7.3 (mixing of the models).**
- Under the stationarity condition $a+b<1$, with innovations that have a positive density,
  GARCH(1,1) is $\beta$-mixing, hence $\alpha$-mixing, with geometric rate (Carrasco and Chen
  2002).
- Stochastic-volatility models inherit $\beta$-mixing from the volatility process (Genon-Catalot,
  Jeantheau and Larédo 2000).
- Window-level coefficients satisfy $\alpha_W(h)\le\alpha_r((h-1)L+1)$: a gap of $h$ windows is a
  gap of at least $(h-1)L+1$ days.
- Finite $(2+\epsilon)$ moments for GARCH-$t$ require $\nu>2+\epsilon$ and the corresponding
  moment condition on $(a,b)$ (Francq and Zakoïan 2019).

Proposition 7.2 is stated for a fixed $f$, i.e. conditionally on a fixed training sample with
growing evaluation data. Applying it with $f=\widehat f$ conditionally on $\mathcal G$ requires
the eval sequences to satisfy D1 *conditionally on* $\mathcal G$, in particular to be independent
of $\mathcal G$ (exactly, or up to the coupling error of Lemma 7.4). If the eval data depend on
the training data, $\mathbb E[\bar a-\bar b\mid\mathcal G]$ can be non-zero under $H_0$, and no
variance estimator corrects a bias in the mean (Proposition 7.5). Triangular-array versions for $f=\widehat f_n$ need CLTs uniform over the
critic class (e.g. de Jong 1997 for dependent arrays); we do not pursue them.

### 7.3 Dependence between the training and evaluation halves

All conditional statements above assume that the eval data are independent of $\mathcal G$. In a
time series this fails. Here $\beta(\mathcal A,\mathcal B)=\mathbb E\sup_{C\in\mathcal B}|\mathbb P(C\mid\mathcal A)-\mathbb P(C)|$
is the absolute-regularity coefficient (Bradley 2005).

**Lemma 7.4 (decoupling).** Let $U$ be the training data with $\sigma$-field $\mathcal A$, and $V$
the eval data with $\sigma$-field $\mathcal B$, both with values in Polish spaces. Let $\tilde V$
be a copy of $V$ independent of $U$. Then for every measurable $E$,
$$
\big|\mathbb P((U,V)\in E)-\mathbb P((U,\tilde V)\in E)\big|\le\beta(\mathcal A,\mathcal B).
$$

*Proof.* Let $E_u=\{v:(u,v)\in E\}$. Then
$\mathbb P((U,V)\in E)=\mathbb E\,\mathbb P_{V\mid U}(E_U)$ and
$\mathbb P((U,\tilde V)\in E)=\mathbb E\,\mathbb P_V(E_U)$. The difference is at most
$\mathbb E\sup_C|\mathbb P_{V\mid U}(C)-\mathbb P_V(C)|=\beta(\mathcal A,\mathcal B)$. $\square$

*Consequence.* Suppose training and evaluation data are **contiguous time blocks separated by a
gap of $g$ days**, so that $\beta(\mathcal A,\mathcal B)\le\beta_r(g+1)$, the $\beta$-mixing
coefficient of the (possibly multivariate) return process at lag $g+1$, provided the synthetic
training and eval windows are independent of each other. Lemma 7.4 replaces the eval data by an
independent copy $\tilde V$; this copy keeps the *internal* dependence of $V$. Hence every level
statement whose remaining hypotheses concern only the eval data holds up to an additive
$\beta_r(g+1)$, which decays geometrically for the models of Fact 7.3. This covers
Proposition 7.2; Proposition 3.1 when the eval windows are themselves exchangeable (e.g.
independent series, but not consecutive windows of one series, Section 7.1); and Proposition 6.2
when its eval samples are i.i.d. (for dependent eval windows, its block-bootstrap analogue,
Remark 6.7).

By contrast, a random interleaved split puts dependent neighbouring windows on both sides. The
critic can then learn the *realised* volatility regime of the real training windows (e.g. "high
volatility means real"). Neighbouring real eval windows share that regime, while synthetic windows
do not, so $T>0$ under $H_0$. **Heuristic:** this mechanism makes random splits of a single
series anti-conservative. We recommend temporal block splits with a gap of several volatility
half-lives (Implementation note 11.1). The same mechanism operates across tickers when a pooled
panel is split by ticker on common dates; Proposition 7.5 makes it precise.

### 7.4 Cross-sectional dependence in pooled panels

In E3, real windows of different tickers covering the same dates are dependent through common
factors: the market volatility regime is shared across tickers. Synthetic windows from univariate
generators are independent across tickers. Then $\operatorname{Var}\bar a$ contains
contemporaneous cross-ticker covariances. The permutation distribution misses them. So does a
Newey–West estimator applied to a sequence in which windows with the same time stamp are ordered
arbitrarily.

The appropriate estimator is that of **Driscoll and Kraay (1998)**. For each time index $\tau$,
aggregate $S_\tau=\sum_{k:t(k)=\tau}(a_k-\bar a)$, then apply Newey–West over $\tau$:
$$
\widehat V_{\rm DK}(\bar a)=\frac1{n_e^2}\Big[\sum_\tau S_\tau^2+2\sum_{h=1}^{H}\Big(1-\frac h{H+1}\Big)\sum_\tau S_\tau S_{\tau+h}\Big].
$$
It is consistent as the number of time periods grows, under mixing in time and arbitrary
cross-sectional dependence. With one window per time stamp, it reduces to Newey–West.

*Effective sample size.* For features driven by common factors, precision is governed by the
number of distinct time windows, not by tickers × windows. The TEST period has about 615 trading
days, i.e. about 30 non-overlapping windows of $L=20$. Under the ticker split of DESIGN §3.7 the
eval half contains all of these dates (and so does the training half, which is the problem
treated next); under the time-block split recommended below it contains about 15. HAC estimates
from so few time points are imprecise and biased downwards. The appropriate small-sample
corrections are fixed-$b$ critical values matched to the kernel and bandwidth of the DK/NW
estimator (Kiefer and Vogelsang 2005), or a cluster-robust variance over $G$ time-block clusters
with $t_{G-1}$ critical values (Bester, Conley and Hansen 2011). The two pairings should not be
mixed: the $t_{G-1}$ approximation is derived for the cluster estimator, not for the Bartlett DK
estimator (in the pilot of Implementation note 11.5, $t_{G-1}$ critical values with DK leave the
size at 11% for nominal 5%).

*Train/eval coupling through common dates.* The DK estimator treats a variance problem. Under
the ticker split of DESIGN §3.7 there is also a **bias**. The training and eval halves cover the
same dates and share the realised market path, so the eval data are not independent of
$\mathcal G$, and Proposition 7.2 does not apply conditionally on $\mathcal G$ (Section 7.2).
Lemma 7.4 does not help either, because a ticker split leaves no time gap.

**Proposition 7.5 (bias under ticker splits on common dates).** Let the real windows be
$X_{i\tau}=\psi(s_\tau,\varepsilon_{i\tau})$ for tickers $i$ and dates $\tau=1,\dots,T_d$, where the
factor path $s=(s_\tau)$ is strictly stationary with marginal $\mu$, the idiosyncratic terms
$\varepsilon_{i\tau}$ are i.i.d. and independent of $s$, and $\psi$ is measurable. Write $P_s$ for
the law of $\psi(s,\varepsilon)$, so that every real window has marginal law
$\bar P=\int P_s\,\mu(ds)$. Let the synthetic windows be mutually independent, independent of
the real data, with marginal $\bar Q$, and assume the pooled null $\bar Q=\bar P$. Split by
ticker, independently of the data, into training tickers $I_{\rm tr}$ and eval tickers
$I_{\rm ev}$, each ticker contributing one window per date. Let $\widehat f$ be
$\mathcal G$-measurable, where $\mathcal G$ is generated by the training windows (real and
synthetic), the split and $\xi$. Then:

(a) Conditionally on $(\mathcal G,s)$, the real eval windows are independent with
$X_{i\tau}\sim P_{s_\tau}$, and
$$
\mathbb E[T\mid\mathcal G,s]=\big(\hat P_s-\bar Q\big)\widehat f,\qquad
\hat P_s:=\frac1{T_d}\sum_{\tau=1}^{T_d}P_{s_\tau}:
$$
the conditional mean of $T$ is the difference between the critic's means under the *realised*
mixture $\hat P_s$ and under the population law.

(b) $\mathbb E\,T=\mathbb E\big[(\hat P_s-\bar Q)\widehat f\big]$. This vanishes if $\widehat f$ is
independent of $s$. In general it is a covariance between the critic and the realised factor
path, and it need not vanish.

(c) (*Idealised critic.*) Suppose the training half reveals $s$ (as with many training tickers
per date), and $\widehat f=\log(d\hat P_s/d\bar Q)$ is the Bayes logit (Proposition 5.1) of the
population version of the training problem, "real windows of these dates against synthetic
windows". Then $\mathbb E[T\mid\mathcal G,s]=J(\hat P_s,\bar Q)\in(0,\infty]$ whenever
$\hat P_s\neq\bar Q$. For Gaussian scale mixtures, $\psi(s,\varepsilon)=s\,\varepsilon$ with
$\varepsilon\sim N(0,I_L)$, this holds a.s. as soon as $\mu$ has no atoms.

*Proof.* (a) The real eval windows are functions of $s$ and of
$(\varepsilon_{i\tau})_{i\in I_{\rm ev}}$. The latter are independent of $(s,\mathcal G)$, because
$\mathcal G$ depends on the real data only through $s$ and $(\varepsilon_{i\tau})_{i\in I_{\rm tr}}$.
So, given $(\mathcal G,s)$, the $X_{i\tau}$ with $i\in I_{\rm ev}$ are independent with laws
$P_{s_\tau}$. The synthetic eval windows are independent of $(\mathcal G,s)$ with law $\bar Q$.
Since $\widehat f$ is $\mathcal G$-measurable and each eval ticker has one window per date,
$\mathbb E[P^{\rm ev}\widehat f\mid\mathcal G,s]=\hat P_s\widehat f$ and
$\mathbb E[Q^{\rm ev}\widehat f\mid\mathcal G,s]=\bar Q\widehat f$.
(b) Take expectations in (a). If $\widehat f\perp s$, condition on $\widehat f$ and use
$\mathbb E\,P_{s_\tau}g=\bar Pg=\bar Qg$ for each fixed $g$ and each $\tau$.
(c) By (a),
$\mathbb E[T\mid\mathcal G,s]=\hat P_s\log\frac{d\hat P_s}{d\bar Q}-\bar Q\log\frac{d\hat P_s}{d\bar Q}=\mathrm{KL}(\hat P_s\Vert\bar Q)+\mathrm{KL}(\bar Q\Vert\hat P_s)$,
which is positive when the laws differ. For the Gaussian scale mixture, the characteristic
function of a mixture $\int P_s\,\rho(ds)$ at $t$ is
$\int e^{-\sigma^2\lVert t\rVert^2/2}\,\rho_2(d\sigma^2)$, the Laplace transform at
$\lVert t\rVert^2/2$ of the law $\rho_2$ of $s^2$ under $\rho$. So the mixture determines
$\rho_2$. For $\hat P_s$, $\rho=T_d^{-1}\sum_\tau\delta_{s_\tau}$ and $\rho_2$ is atomic; for
$\bar Q=\bar P$, $\rho=\mu$ and $\rho_2$ has no atoms when $\mu$ has none. Hence
$\hat P_s\neq\bar Q$. $\square$

Proposition 7.5 is the cross-sectional analogue of the interleaved-split mechanism of
Section 7.3. With many training tickers per date, the training half reveals the realised factor
path. A critic that separates real from synthetic training windows then learns "windows typical
of the realised TEST dates are real". The real eval windows share those dates, and the synthetic
windows do not. The result is a bias, $\mathbb E[T\mid\mathcal G]\neq0$ (typically $>0$) under the
pooled null. DK and HAC variances estimate the dispersion of $T$ *around*
$\mathbb E[T\mid\mathcal G]$, so they cannot remove it: the DK z is shifted by roughly bias/se,
and the DK p-value is anti-conservative as well. The permutation p-value, whose null variance
also ignores the cross-sectional dependence, is worse. A pilot simulation (Implementation
note 11.5) shows both effects. E1 cannot reveal this bias: when both samples are real tickers on
common dates, both share the realised path, and (a) gives $\hat P_s$ on both sides.

*Recommendations for pooled real-data tests.*

(a) **Time-block split** (recommended). Split the TEST dates into a training block and an eval
block that are disjoint and separated by a gap of $g$ days, with all tickers on both sides, and
apply the same date split to the synthetic windows (this matters when the generator starts from
the fitted end state). Lemma 7.4 then applies with $\beta_r(g+1)$ for the panel viewed as a
multivariate process: rejection probabilities differ by at most $\beta_r(g+1)$ from those of a
design with independent eval data, for which Proposition 7.5(b) gives no bias. The DK z with
fixed-$b$ critical values (or a time-block cluster variance with $t_{G-1}$ critical values)
handles the remaining cross-sectional and serial dependence within the eval block; its residual
small-sample distortion should be measured by the simulated placebo of Section 7.6. At $L=20$,
the TEST period gives about 14 training dates, a one-window gap and about 15 eval dates.

(b) Alternatively, **train the critic on a separate period**, e.g. real TRAIN windows against
synthetic windows simulated for the TRAIN dates, separated from TEST by a gap, and evaluate on
TEST. This keeps all TEST dates for evaluation. The level argument is as in (a); power requires
the generator's deviation to be present in the training period too.

**For pooled real-data tests, the permutation p-value is anti-conservative. The DK p-value, with
fixed-$b$ critical values (or the time-block cluster alternative), should be the primary
real-data p-value only when the critic is trained on dates that are disjoint from, and separated
by a gap from, the eval dates.
Under the ticker split of DESIGN §3.7, every p-value, DK included, is biased towards rejection;
E3 p-values computed that way should not be reported as tests.**

### 7.5 Studentised permutation tests

Chung and Romano (2013) show that permuting a *studentised* difference of means gives a test that
is exact when the samples are i.i.d. from the same law, and asymptotically valid for equality of
means when the laws differ (i.i.d. samples). Their result concerns unequal laws, not serial
dependence. DESIGN §3.7 cites Chung and Romano (2013) for the robustness of `z_hac` to serial
dependence between windows; that citation is incorrect. The paper should cite Newey and West
(1987), Andrews (1991) and Driscoll and Kraay (1998) for `z_hac`, and Chung and Romano (2013)
only for the studentised permutation under unequal i.i.d. laws (Remark 3.5(b); Implementation
note 11.8).

**Heuristic 7.6.** Permuting the HAC-studentised statistic should be asymptotically valid under
D1–D4. Permutation destroys the time structure, so the permuted studentised statistic is
asymptotically $N(0,1)$ by a combinatorial CLT (Hoeffding 1952). The observed one is $N(0,1)$ by
Proposition 7.2. We have not proved this, and the paper should not claim it. It does offer a
hybrid: exact under exchangeability, and plausibly robust under dependence.

### 7.6 The empirical placebo (E1) as complementary evidence

E1 compares real data with real data where $H_0$ is plausible. It is complementary evidence, but
it cannot certify size in E3:
- *Random ticker halves in the same period share dates.* Common shocks then correlate the two
  samples positively, and $\operatorname{Var}(\bar a-\bar b)=\operatorname{Var}\bar a+\operatorname{Var}\bar b-2\operatorname{Cov}(\bar a,\bar b)$
  is *smaller* than under independence. The permutation test is conservative there even if it is
  anti-conservative in E3, where the synthetic sample is independent of the realised market path.
- *The train/eval coupling of Proposition 7.5 is invisible in E1.* Both samples are real tickers
  on the same dates, so both share the realised path, and the bias cancels. E1 can look well
  calibrated while E3, under the same ticker split, is not.
- *Repeated random splits of one history are dependent.* The rejection frequency estimates a
  rejection rate conditional on the realised market path, not the unconditional size.
- E1 is still useful. Gross failures show up, and it calibrates the magnitude of $T$ under
  realistic cross-ticker heterogeneity.

*Recommended complement* (Implementation note 11.5): a simulated placebo that reproduces the E3
dependence asymmetry. Simulate the "real" panel from a model with cross-sectional dependence:
CCC-GARCH-$t$ fitted to the panel, or `MVStationaryBootstrap` with blocks common to all assets.
Simulate the "synthetic" windows independently per ticker from the *same* marginal law: the
univariate GARCH-$t$ margins, or per-ticker independent stationary bootstraps. The per-ticker
nulls then hold exactly or approximately. The placebo must use **exactly** the split, grouping,
ordering and sample sizes used in E3. The rejection rates of the permutation, HAC and DK p-values
then measure their size distortion directly, including the bias of Proposition 7.5 when E3 uses a
ticker split.

---

## 8. Decision rule and multiple testing

### 8.1 The verdict

**Definition 8.1 (power check).** A power check is a pre-registered pair
$(\mathcal C_{\rm ref},\delta^*)$:
- $\mathcal C_{\rm ref}$ is a reference simulation design that matches the empirical one: the
  sample sizes of $R_{\rm past}$, $R_{\rm new}$ and $G$, $L$, the split, the critic
  hyperparameters, and, where possible, the dependence structure. By default, $R_{\rm past}$ and
  $R_{\rm new}$ are drawn from one reference law (e.g. GARCH-$t$ fitted to the panel), so that
  $\delta\approx0$ for the past.
- $\delta^*$ is a pre-specified deviation applied to the generator, e.g. synthetic volatility
  off by 20%, or Gaussian instead of $t_4$ tails.

Let $\beta(\delta^*)=\mathbb P_{\mathcal C_{\rm ref},\delta^*}(p_{\rm rel}\le\alpha)$. Estimate it
from $R$ replications by $\hat\beta$, with one-sided Clopper–Pearson lower bound $\beta_L$
(Clopper and Pearson 1934). The same design, with the generator replaced by an independent draw
from the law of $R_{\rm past}$ ($\delta^*=0$), gives the boundary size
$\alpha_{\rm bd}=\mathbb P_{\mathcal C_{\rm ref},0}(p_{\rm rel}\le\alpha)$, estimated by
$\hat\alpha_{\rm bd}$ with a Clopper–Pearson interval; by Proposition 6.8 it exceeds $\alpha$.

**Definition 8.2 (verdict; DESIGN §1).** With $\alpha=0.05$ and $\beta_{\min}=0.8$:
$$
V=\begin{cases}
\textsf{BAD} & \text{if } p_{\rm rel}\le\alpha,\\
\textsf{GOOD} & \text{if } p_{\rm rel}>\alpha\ \text{and}\ \hat\beta\ge\beta_{\min}\ \ (\text{recommended: }\beta_L\ge\beta_{\min}),\\
\textsf{INCONCLUSIVE} & \text{otherwise.}
\end{cases}
$$
The absolute test is reported alongside. The combination "absolute test rejects, relative test
does not" means "distinguishable from the future, but no worse than history". It is the expected
real-data outcome and a finding in itself.

**Proposition 8.3 (what the verdict controls).**

(a) Under R1–R3 (Proposition 6.2), $\mathbb P(V=\textsf{BAD}\mid\mathcal G)\le\alpha+o_P(1)$ on
the $\mathcal G$-measurable event $\{\delta\le0\}$, and only there. This is a statement about the
two trained critics. It does **not** bound the false-BAD rate of a generator that is as good as
history: if $Q_G=P_{\rm past}$ with equal resources, Proposition 6.8 gives
$\mathbb P(V=\textsf{BAD})=\mathbb E\,\bar g(|\delta|/\sigma_N)+o(1)>\alpha$ unless
$\delta=o_P(\sigma_N)$. A generator equal in law to the past is labelled BAD more often than
$\alpha$; the simulated boundary size $\hat\alpha_{\rm bd}$ of Definition 8.1 says how much more.

(b) Suppose the relative test's rejection probability at the true configuration is at least
$\beta(\delta^*)$. Then $\mathbb P(V=\textsf{GOOD})\le\mathbb P(p_{\rm rel}>\alpha)\le1-\beta(\delta^*)\le0.2$,
up to Monte Carlo error in $\hat\beta$.

*Proof.* (a) $\{V=\textsf{BAD}\}=\{p_{\rm rel}\le\alpha\}$. (b) $\{V=\textsf{GOOD}\}\subset\{p_{\rm rel}>\alpha\}$.
$\square$

Part (b) is only as strong as its premise, which cannot be verified: the true deviation must be
*at least as detectable* as $\delta^*$ under $\mathcal C_{\rm ref}$. **GOOD therefore means "no
detectable deviation of the pre-specified size and type", never "equal to the truth".** The power
check is a design-stage calculation, like a sample-size calculation. With $R=100$ replications
the standard error of $\hat\beta$ is about $0.04$, so $R\ge200$ and the $\beta_L$ criterion are
recommended.

**Remark 8.4 (equivalence-test alternative).** A GOOD verdict with a formal error guarantee
follows from an equivalence (non-inferiority) test (Schuirmann 1987; Wellek 2010). Test
$H_0^{\rm eq}:\delta\ge\varepsilon$ against $H_1^{\rm eq}:\delta<\varepsilon$, and declare GOOD if
the upper $(1-\alpha)$ bootstrap confidence bound for $\delta$ lies below $\varepsilon$. By
Proposition 6.2, $\mathbb P(\textsf{GOOD}\mid\mathcal G)\le\alpha+o_P(1)$ on $\{\delta\ge\varepsilon\}$
(again a statement about the trained critics; by the argument of Proposition 6.8, the
unconditional error rate at a configuration where $\mathbb E\delta=\varepsilon$ can exceed
$\alpha$ when $\delta$ fluctuates at the scale of $\sigma_N$). A natural
margin is the expected critic discrepancy at the pre-specified deviation, estimated in the
power-check simulation. **Heuristic:** this transfers across configurations only to the extent
that critic training behaves similarly. We suggest reporting it as a robustness column.

**Remark 8.5 (seeds).** The verdict inherits the p-value lottery of Remark 3.5(d). Aggregate over
splits and seeds with a valid combination, e.g. twice the median p-value, before applying
Definition 8.2.

### 8.2 Multiple testing across tickers

E3 runs per-ticker tests ($K=100$ tickers, $L=5$) with p-values $p_1,\dots,p_K$:
- The Benjamini–Hochberg procedure (1995) at level $q$ controls the false discovery rate of
  **BAD** verdicts at $qK_0/K$ under independence or positive regression dependence (PRDS;
  Benjamini and Yekutieli 2001, Thm 1.2).
- One-sided tests whose statistics are jointly Gaussian with nonnegative correlations are PRDS
  (Benjamini and Yekutieli 2001, §3.1). Per-ticker statistics are **plausibly** positively
  dependent through the market factor, but this is not proved.
- Report BH as primary and the Benjamini–Yekutieli correction (level
  $q/\sum_{k\le K}k^{-1}\approx q/5.19$ for $K=100$) as robustness.
- Storey's (2002) $\hat\pi_0$ summarises the fraction of tickers for which the generator is not
  detectably worse.

GOOD is a non-rejection, not a discovery, so no FDR statement applies to it. Per-ticker
$\textsf{GOOD}$ also needs per-ticker power at per-ticker sample sizes (about 60 eval windows at
$L=5$), so most per-ticker verdicts will be BAD or INCONCLUSIVE. This should be said in advance.

---

## 9. Markowitz utility

### 9.1 Set-up and utility losses

Let $r_t\in\mathbb R^N$, with real TRAIN sample $R_{\rm tr}$ ($T_0\times N$) and real TEST sample
$R_{\rm te}$ ($T_1\times N$). Let $(\mu,\Sigma)$ be the TEST-period moments: either population
moments, or the realised $(\hat\mu_{\rm te},S_{\rm te})$ used by `portfolio.evaluate`, in which
case all statements below are exact identities for the measured quantities. The feasible set
$\mathcal W$ is $\{w:\mathbf 1'w=1\}$, intersected with $\{w\ge0\}$ for long-only portfolios.
- GMV: $w_{\rm gmv}(\hat\Sigma)=\arg\min_{w\in\mathcal W}w'\hat\Sigma w$, which equals
  $\hat\Sigma^{-1}\mathbf 1/(\mathbf 1'\hat\Sigma^{-1}\mathbf 1)$ when unconstrained.
- Mean-variance: $w_{\rm mv}(\hat\mu,\hat\Sigma)=\arg\max_{w\in\mathcal W}w'\hat\mu-\tfrac\gamma2w'\hat\Sigma w$.

*TRTR versus TSTR.* $\hat w_{\rm real}=w(\text{moments}(R_{\rm tr}))$ and
$\hat w^{(b)}_{\rm syn}=w(\text{moments}(Y^{(b)}))$, where $Y^{(b)}$, $b\le B$, are synthetic
samples of length $T_0$ from $G$ calibrated on $R_{\rm tr}$.

*Losses.* With $V(w)=w'\Sigma w$, $w^*=w_{\rm gmv}(\Sigma)$ and $V^*=V(w^*)$:
$$
\mathcal L_{\rm gmv}(w)=V(w)-V^*\ \ \big(\text{or } V(w)/V^*-1\big),\qquad
\mathcal L_{\rm mv}(w)={\rm CE}(w^*_{\rm mv})-{\rm CE}(w),\quad {\rm CE}(w)=w'\mu-\tfrac\gamma2w'\Sigma w .
$$
The TSTR utility gap (E4) is
$\mathrm{Gap}_{\rm gmv}=B^{-1}\sum_bV(\hat w_{\rm syn}^{(b)})-V(\hat w_{\rm real})$, and likewise
for CE and Sharpe ratios.

### 9.2 GMV utility depends on second moments only

**Lemma 9.1 (quadratic loss identities).**
(a) Unconstrained GMV: for every $w$ with $\mathbf 1'w=1$,
$V(w)-V^*=(w-w^*)'\Sigma(w-w^*)$.
(b) For a closed convex $\mathcal W$ and $w\in\mathcal W$: $V(w)-V^*\ge(w-w^*)'\Sigma(w-w^*)$.
(c) Mean-variance with only the budget constraint:
${\rm CE}(w^*_{\rm mv})-{\rm CE}(w)=\tfrac\gamma2(w-w^*_{\rm mv})'\Sigma(w-w^*_{\rm mv})$; with
further convex constraints, "$\ge$".

*Proof.* (a) $\Sigma w^*=\mathbf 1/(\mathbf 1'\Sigma^{-1}\mathbf 1)$, so
$w^{*\prime}\Sigma(w-w^*)=\mathbf 1'(w-w^*)/(\mathbf 1'\Sigma^{-1}\mathbf 1)=0$. Expand
$V(w)=V^*+2w^{*\prime}\Sigma(w-w^*)+(w-w^*)'\Sigma(w-w^*)$.
(b) First-order optimality of $w^*$ on $\mathcal W$ gives $w^{*\prime}\Sigma(w-w^*)\ge0$.
(c) The first-order condition is $\mu-\gamma\Sigma w^*-\nu\mathbf 1=0$ for a multiplier $\nu$.
For feasible $w$,
${\rm CE}(w)-{\rm CE}(w^*)=(w-w^*)'(\gamma\Sigma w^*+\nu\mathbf 1)-\tfrac\gamma2[(w-w^*)'\Sigma(w-w^*)+2(w-w^*)'\Sigma w^*]$,
which equals $-\tfrac\gamma2\lVert w-w^*\rVert_\Sigma^2$ because $\mathbf 1'(w-w^*)=0$. With
constraints, use $\nabla{\rm CE}(w^*)'(w-w^*)\le0$. $\square$

Applied with $\Sigma=S_{\rm te}$, (a) gives an exact decomposition of the measured E4 gap
(unconstrained GMV):
$$
V_{\rm te}(\hat w_{\rm syn})-V_{\rm te}(\hat w_{\rm real})=\lVert\hat w_{\rm syn}-w^*_{\rm te}\rVert^2_{S_{\rm te}}-\lVert\hat w_{\rm real}-w^*_{\rm te}\rVert^2_{S_{\rm te}},
$$
a difference of Mahalanobis distances to the ex-post optimal GMV portfolio.

**Proposition 9.2 (sensitivity of GMV to $\hat\Sigma$).** Consider unconstrained GMV with
$\Sigma,\hat\Sigma$ symmetric positive definite. Let $A=\Sigma^{-1/2}\hat\Sigma\Sigma^{-1/2}$ (similar
to $\Sigma^{-1}\hat\Sigma$), $\kappa=\lambda_{\max}(A)/\lambda_{\min}(A)$, and
$M=\Sigma^{-1}-\Sigma^{-1}\mathbf 1\mathbf 1'\Sigma^{-1}/(\mathbf 1'\Sigma^{-1}\mathbf 1)$.

(a) *Scale invariance.* $w(c\hat\Sigma)=w(\hat\Sigma)$ for every $c>0$; in particular
$V(w(c\Sigma))=V^*$.

(b) *Kantorovich bound.* $1\le V(w(\hat\Sigma))/V^*\le(1+\kappa)^2/(4\kappa)$, and the bound is
sharp.

(c) *Relative operator-norm error.* Let $\varepsilon:=\inf_{c>0}\lVert cA-I\rVert_{\rm op}$, which
equals $(\kappa-1)/(\kappa+1)<1$. Then $V(w(\hat\Sigma))/V^*-1\le\varepsilon^2/(1-\varepsilon^2)$, a
restatement of (b). In particular, if the relative covariance error satisfies
$\lVert\Sigma^{-1/2}(\hat\Sigma-\Sigma)\Sigma^{-1/2}\rVert_{\rm op}=\varepsilon_0<1$, then the
relative GMV loss is at most $\varepsilon_0^2/(1-\varepsilon_0^2)$: it is *quadratic* in the
relative covariance error.

(d) *First order.* $w(\Sigma+E)-w^*=-MEw^*+O(\lVert E\rVert^2)$ and
$V(w(\Sigma+E))-V^*=(Ew^*)'M(Ew^*)+O(\lVert E\rVert^3)$. Here $M\succeq0$ with
$\ker M=\operatorname{span}(\mathbf 1)$, so the quadratic term vanishes if and only if $w^*$ is
also the GMV portfolio of $\Sigma+E$.

*Proof.* (a) is immediate.
(b) Let $u=\Sigma^{-1/2}\mathbf 1$. Then
$V(w(\hat\Sigma))/V^*=(u'A^{-2}u)(u'u)/(u'A^{-1}u)^2$. The Kantorovich inequality states
$(x'Bx)(x'B^{-1}x)\le\frac{(M_B+m_B)^2}{4M_Bm_B}(x'x)^2$ for symmetric positive definite $B$ with
spectrum in $[m_B,M_B]$ (Kantorovich 1948; Horn and Johnson 2013). Apply it with $B=A^{-1}$ and
$x=B^{1/2}u$, using $M_B/m_B=\kappa$. The lower bound is optimality of $w^*$. Equality holds when
$x=B^{1/2}u=A^{-1/2}u$ (not $u$ itself) has equal mass on the extreme eigenvectors $e_{\min}$,
$e_{\max}$ of $A$ and none elsewhere, i.e.
$(u'e_{\min})^2/\lambda_{\min}=(u'e_{\max})^2/\lambda_{\max}$. For every $\Sigma$ (hence $u$),
$N\ge2$ and $\kappa\ge1$, one can choose $A$ with condition number $\kappa$ and two eigenvectors
spanning a plane that contains $u$ at the required angle, and set
$\hat\Sigma=\Sigma^{1/2}A\Sigma^{1/2}$; so the bound is sharp (Example 9.6 gives an explicit
instance).
(c) $\lVert cA-I\rVert_{\rm op}$ is minimised at $c=2/(\lambda_{\max}+\lambda_{\min})$, with value
$(\kappa-1)/(\kappa+1)$, and $\varepsilon^2/(1-\varepsilon^2)=(\kappa-1)^2/(4\kappa)$. If
$\lVert A-I\rVert_{\rm op}=\varepsilon_0<1$, the spectrum of $A$ lies in
$[1-\varepsilon_0,1+\varepsilon_0]$, so $\kappa\le(1+\varepsilon_0)/(1-\varepsilon_0)$, and
$(1+\kappa)^2/(4\kappa)$ is increasing in $\kappa\ge1$.
(d) Differentiating $w=\Sigma^{-1}\mathbf 1/(\mathbf 1'\Sigma^{-1}\mathbf 1)$ gives
$dw=-\Sigma^{-1}dE\,w+s(w'dE\,w)w=-M\,dE\,w$ with $s=\mathbf 1'\Sigma^{-1}\mathbf 1$. Then
Lemma 9.1(a) and $M\Sigma M=M$ give the quadratic form. Finally, $Mx=0\iff x\propto\mathbf 1$, and
$Ew^*\propto\mathbf 1\iff(\Sigma+E)w^*\propto\mathbf 1\iff w^*=w(\Sigma+E)$. $\square$

(The numerical sanity checks of (b)–(d) were done on random matrices with $N=5$ and are not part
of the paper.)

**Corollary 9.3 (what drives synthetic GMV utility).** Let $\hat\Sigma_{\rm syn}$ be the sample
covariance of a synthetic sample of length $T_s$ from a stationary ergodic generator with
covariance $\Sigma_G$.

(i) $\hat\Sigma_{\rm syn}\to\Sigma_G$ a.s. as $T_s\to\infty$ (ergodic theorem). Hence
$V(w(\hat\Sigma_{\rm syn}))/V^*\to V(w(\Sigma_G))/V^*\le(1+\kappa_G)^2/(4\kappa_G)$, with
$\kappa_G=\operatorname{cond}(\Sigma^{-1}\Sigma_G)$. In the large-sample limit, **the GMV
utility of a generator depends on it only through its covariance matrix, and only up to scale**.
Tails, volatility clustering, jumps and serial dependence matter only insofar as they change
$\Sigma_G$.

(ii) **Heuristic.** At finite $T_s$, tails and clustering enter through the fluctuation
$\eta=\hat\Sigma_{\rm syn}-\Sigma_G=O_P(T_s^{-1/2})$. Its asymptotic covariance is the long-run
covariance of $(r_tr_t'-\Sigma_G)w^*$, which is inflated by fourth moments (tails) and by
volatility persistence (clustering). By Proposition 9.2(d),
$$
\mathbb E\,\mathcal L_{\rm gmv}\approx\big((\Sigma_G-\Sigma)w^*\big)'M\big((\Sigma_G-\Sigma)w^*\big)+T_s^{-1}\operatorname{tr}(M\Omega_G),
$$
where $\Omega_G$ is the long-run covariance of $r_t(r_t'w^*)$. This neglects higher-order terms
and mean estimation.

(iii) **Heuristic (noise doubling).** If $G$ is calibrated so that $\Sigma_G\approx\hat\Sigma_{\rm tr}$
(exact for `MVGaussian`) and one synthetic sample of length $T_0$ is used, then
$\hat\Sigma_{\rm syn}-\Sigma=(\hat\Sigma_{\rm tr}-\Sigma)+(\hat\Sigma_{\rm syn}-\Sigma_G)$. The two
error terms are conditionally uncorrelated and of comparable size. Even a well-specified generator
then roughly *doubles* the estimation-noise part of the TRTR loss. Pooling $B$ replicas
($T_s=BT_0$) divides the extra term by $B$. The same argument applies with more force to
$\hat\mu_{\rm syn}$ in mean-variance portfolios (Implementation note 11.6).

**Proposition 9.4 (no free lunch from synthetic data).** Let $Y=g(R_{\rm past},U)$ with $U$
independent of $(R_{\rm past},\vartheta_{\rm new})$, where $\vartheta_{\rm new}$ is any TEST-period
quantity, such as $\Sigma$, $(\mu,\Sigma)$ or $R_{\rm te}$. Let $\pi$ be any portfolio rule with
values in a convex $\mathcal W$, and let $\ell(w;\vartheta)$ be convex in $w$. Then the rule
$\bar w(R_{\rm past})=\mathbb E[\pi(Y)\mid R_{\rm past}]$ is feasible, uses only $R_{\rm past}$,
and satisfies
$\ell(\bar w;\vartheta_{\rm new})\le\mathbb E[\ell(\pi(Y);\vartheta_{\rm new})\mid R_{\rm past},\vartheta_{\rm new}]$.

*Proof.* Feasibility follows from convexity of $\mathcal W$. Apply Jensen's inequality
conditionally on $(R_{\rm past},\vartheta_{\rm new})$, using
$U\perp(R_{\rm past},\vartheta_{\rm new})$. $\square$

For GMV, $V(w)=w'\Sigma w$ is convex, and so is $-{\rm CE}$. Hence: (i) no TSTR rule beats the
best rule based on the real past (a Blackwell-type statement; Blackwell 1953); and (ii) averaging
weights over replicas beats averaging the losses of replica-wise portfolios. A synthetic pipeline
can beat a *specific* TRTR rule, such as the sample-covariance GMV, only by acting as a
regulariser: a structured covariance, as in CCC models, acts like shrinkage.

### 9.3 Critic verdicts and portfolio utility can disagree

**Example 9.5 (BAD for the critic, perfect for GMV).** Let returns be i.i.d. $N(0,\Sigma)$ and
let $G$ be i.i.d. $N(0,c\Sigma)$ with $c\neq1$. Windows satisfy $X\overset d=\Sigma^{1/2}Z$ and
$Y\overset d=\sqrt c\,X$ (column-wise), so
$W_1(P,Q)=|\sqrt c-1|\,\mathbb E\lVert X\rVert_F>0$: the scaling coupling gives "$\le$", and
$f=\lVert\cdot\rVert_F$ attains it. The standardised laws also differ, the standardisation being a
bijection, so under Assumption C the critic test rejects with probability tending to one
(Proposition 4.2), and the relative test tends to reject whenever the past is closer. Yet
by Proposition 9.2(a) the GMV loss of $\Sigma_G=c\Sigma$ is exactly zero.

**Example 9.6 (nearly perfect in $W_1$, large GMV loss).** Let $N=2$ with
$\Sigma=v_1v_1'+\eta^2v_2v_2'$ and $\Sigma_G=v_1v_1'+s\eta^2v_2v_2'$. Here
$v_1=(\cos\vartheta,\sin\vartheta)$, $v_2=(-\sin\vartheta,\cos\vartheta)$ and
$\tan\vartheta=(1-\eta\sqrt s)/(1+\eta\sqrt s)$. Here
$A=\Sigma^{-1/2}\Sigma_G\Sigma^{-1/2}=v_1v_1'+s\,v_2v_2'$, so $\kappa=1/s$, and the angle is chosen
so that $A^{-1/2}u$, with $u=\Sigma^{-1/2}\mathbf 1$, loads equally on $v_1$ and $v_2$ (the
equality condition in the proof of Proposition 9.2(b)). Then
- $V(w(\Sigma_G))/V^*=(1+s)^2/(4s)$, the Kantorovich bound $(1+\kappa)^2/(4\kappa)$, attained; it
  is $1.5625$ for $s=1/4$, independently of $\eta$. (With $\tan\vartheta=(1-\eta)/(1+\eta)$
  instead, $u$ itself loads equally on $v_1$ and $v_2$, and the ratio is
  $2(1+s^2)/(1+s)^2=1.36$, below the bound.)
- for i.i.d. Gaussian windows of length $L$,
  $W_1\le W_2=\sqrt L\,\eta\,(1-\sqrt s)\to0$ as $\eta\to0$ (Gaussian $W_2$ formula: Dowson and
  Landau 1982; Olkin and Pukelsheim 1982; additivity of $W_2^2$ over independent coordinates;
  the value does not depend on $\vartheta$). The per-asset standardisation of Remark 2.3 (not to
  be confused with the matrix $A$ of Proposition 9.2) does not change this: $W_1$ of the
  standardised windows is at most $\lVert D^{-1}\rVert_{\rm op}W_2$, and
  $\lVert D^{-1}\rVert_{\rm op}\to\sqrt2$ as $\eta\to0$ (both asset variances tend to $1/2$; it is
  about $1.49$ at $\eta=0.1$).

The covariance error sits in the low-variance spread direction, which GMV weights heavily (and
with extreme long-short positions: for $\eta=0.1$, $w^*=(-1.48,2.48)$ and
$w(\Sigma_G)=(-4.49,5.49)$). $W_1$ discounts this direction by $\sqrt{\lambda_2}=\eta$. So the
critic *distance*, the quantity E4 correlates with utility loss, ranks such a generator as nearly
perfect.

**Heuristic:** the training objective, in standardised units with a 1-Lipschitz network, gains
at most $W_1\le\lVert D^{-1}\rVert_{\rm op}\sqrt L\eta(1-\sqrt s)\approx\sqrt{2L}\,\eta(1-\sqrt s)$
from this direction. When that gain is small relative to the $O(n_t^{-1/2})$ sampling noise of the
objective, training is unlikely to find the direction, and the test is unlikely to reject.

**Example 9.7 (univariate critics are blind to correlation).** A generator with exact univariate
window laws and independent assets satisfies every per-ticker null of E3: each univariate test
has level $\alpha$ at most. Its GMV portfolio can nevertheless be arbitrarily bad, since
$\Sigma_G=\operatorname{diag}(\Sigma)$.

**Example 9.8 (drift errors are undetectable yet decisive for mean-variance portfolios).** In
the Gaussian i.i.d. idealisation with a known $\sigma$, consider a synthetic drift error $\delta$
for one asset. By Proposition 5.5(b) (null point: both samples from the synthetic law), no
level-$\alpha$ two-sample test that uses $T_1$ real TEST days can have power above
$1-\Phi(z_{1-\alpha}-|\delta|\sqrt{T_1}/\sigma)$, however many synthetic draws it uses. Take
$\delta=\sigma/\sqrt{T_0}$, the standard error of the TRAIN mean, and $T_1/T_0\approx615/1258$.
The power is then at most $1-\Phi(1.645-0.70)\approx0.17$.

Errors of this size dominate mean-variance utility (Merton 1980; Best and Grauer 1991; Chopra and
Ziemba 1993), and by Lemma 9.1(c) the loss is
$\tfrac1{2\gamma}(\hat\mu-\mu)'M(\hat\mu-\mu)$ when $\hat\Sigma=\Sigma$. Calibrated generators
reproduce the TRAIN mean, so they cannot remove this error (Proposition 9.4).

**Consequences for RQ5.** Critic distance and portfolio utility are related but not monotonically:
- GMV utility is a function of $\Sigma_G$ up to scale (Corollary 9.3);
- the critic is sensitive to scale, tails and clustering, to which GMV is blind (Example 9.5);
- GMV is sensitive to relative errors in low-variance directions, which $W_1$ discounts
  (Example 9.6);
- mean-variance utility is dominated by drift errors that no test can detect at these sample
  sizes (Example 9.8).

A generator can therefore be BAD in the critic sense and useful for GMV, and vice versa. Whether
verdicts *predict* utility on real data is an empirical question, which is RQ5. The theory
predicts (**heuristic**) a stronger association for GMV, through covariance accuracy, than for
mean-variance portfolios. It also predicts that univariate verdicts carry no information about
the correlation structure that drives GMV.

---

## 10. Summary: what is proved and what is heuristic

| Result | Status | Key assumptions | Used in |
|---|---|---|---|
| Prop. 1.2: windows identify dependence up to lag $L-1$ | proved | stationarity | choice of $L$, E5 |
| Prop. 1.3: pairs determine a stationary Markov chain | proved | both chains Markov(1) | the "Markov models" of the title |
| Prop. 2.5: $\mathbb E[T\mid\mathcal G]=\Delta(\widehat f)\le\mathrm{Lip}(\widehat g)\,W_1(A_\#P,A_\#Q)$ (standardised coordinates) | proved | eval independent of train, $\mathcal P_1$ | `W1_lower` |
| Prop. 3.1: exact level of the fixed-critic permutation test | proved | conditional exchangeability (A1′) | RQ1, E2 |
| Prop. 4.2: consistency | proved | C1–C5, including the idealised hard Lipschitz constraint C3 | RQ2 |
| Cor. 4.3: $T\to W_1(A^*_\#P,A^*_\#Q)$ | proved | idealised C3 with $c=1$ (standardised coordinates), rich classes, $\varepsilon_{\rm opt}\to0$ | interpretation |
| Prop. 5.1, Cor. 5.2: Bayes logit and Jeffreys divergence | proved (standard) | perfect discriminator | `lrcritic` |
| Prop. 5.5: no two-sample test beats the NP oracle | proved | densities; level $\le\alpha$ at the null point | E2 oracle |
| Prop. 5.6: local power factor $\le\sqrt{\rho\kappa/(1+\kappa)}$ | proved for fixed $f$; sketch for learned $f$ | DQM | E2 interpretation |
| Heur. 5.8: LR critic locally more efficient than $W_1$ critic | heuristic | — | E2 prediction |
| Prop. 6.2: paired bootstrap valid for the conditional relative null; percentile CI | proof sketch | R1–R3 | RQ3 |
| Rem. 6.3: no guarantee for the $W_1$ relative null | stated | — | interpretation |
| Prop. 6.8: unconditional size $>\alpha$ at $Q_G=P_{\rm past}$ unless $\delta=o_P(\sigma_N)$; $\to\tfrac12$ for fixed critics | proved (given Prop. 6.2) | R1–R3; equal resources; exchangeable training of the two critics | verdict interpretation, boundary-size table (E2/E5) |
| Heur. 7.1: size distortion of the naive permutation test | heuristic (Gaussian, AR(1)) | — | E1/E3 caveat |
| Prop. 7.2: HAC test valid under mixing | proved from standard results (fixed critic) | D1–D4; applied to $\widehat f$ only if eval data are independent of $\mathcal G$ | E3 real data (time-block split) |
| Lemma 7.4: train/eval decoupling through $\beta$-mixing | proved | Polish spaces; time gap; eval-only hypotheses still needed | split design |
| Prop. 7.5: bias of $T$ under ticker splits on common dates | proved (factor model) | common factor, independent synthetic windows | E3 split design; against DK as a fix |
| Heur. 7.6: HAC-studentised permutation | heuristic | — | robustness |
| Prop. 8.3: verdict error control | proved under premises; BAD part conditional on $\{\delta\le0\}$ only (false-BAD rate at $Q_G=P_{\rm past}$ exceeds $\alpha$, Prop. 6.8); GOOD part conditional on an unverifiable premise | — | verdict table |
| Lemma 9.1, Prop. 9.2: GMV loss identity, scale invariance, Kantorovich bound | proved | positive definite matrices | RQ5 |
| Cor. 9.3(i) / (ii)–(iii) | proved / heuristic | ergodicity | E4 design |
| Prop. 9.4: no free lunch | proved | convex loss | RQ5 |
| Ex. 9.5–9.8: critic and utility disagree | proved (9.6: the distance statement; the learnability remark is heuristic) | Gaussian idealisations | RQ5 |

---

## 11. Implementation notes (for module owners; not for the paper)

These recommendations go beyond DESIGN.md. Adopting them is the lead author's decision.

**11.1 `twosample/critic.py` and `c2st.py`.**
- When `groups_*` are given, permute *within groups* (stratified). This makes Proposition 3.1
  exact under the per-group nulls; the full permutation is conservative (Remark 3.5(c)).
- Offer a studentised-permutation option (Remark 3.5(b)).
- Report a Lipschitz diagnostic `extra["lip_hat"]` (gradient norm, with respect to the
  *standardised* input, on interpolates of the standardised pooled eval set) and `T / lip_hat`
  (Remark 2.8). Document `W1_lower` as a reading on the standardised scale (Remark 2.3).
- Report the lower confidence bound $T-1.645\,\widehat{\rm se}$ (Remark 2.6).
- Split train/eval into *contiguous time blocks with a gap*, not at random, for a single real
  time series (Section 7.3) **and** for pooled panels on common dates (Proposition 7.5). A
  `split="time"` option that takes the date of each window and a gap would serve both.

**11.2 HAC (`z_hac`, `p_hac`).**
- If `order_x` has ties (several tickers per date), sum within each time index before applying
  Newey–West (Driscoll–Kraay, Section 7.4). A Newey–West estimator over a tie-broken ordering
  misses the cross-sectional covariance. Use the Newey–West (1994) bandwidth.
- *Units.* `order_x`/`order_y` must be consecutive window indices (start day divided by $L$) or
  ranks of distinct dates, **not** trading-day indices: `hac_var_mean` places integer orders on a
  dense grid and counts lags in grid units (Section 7.2). Pilot (AR(1) window values, $\phi=0.6$,
  30 windows, 4000 replications): true $\operatorname{Var}(\bar a)=0.198$; with window indices the
  estimator averages $0.098$ (3 lags); with $20\times$ window indices it averages $0.047$, exactly
  the i.i.d. $s^2/n$, because every lagged product is zero. Either document this in
  `hac_var_mean` and the experiment scripts, or have `hac_var_mean` use ranks of distinct dates.
- The pilot also shows that Newey–West with about 30 time points underestimates the variance by
  half even in the right units. With few time points, use fixed-$b$ critical values for the
  DK/NW statistic, or a time-block cluster variance with $t_{G-1}$ critical values; do not pair
  the Bartlett DK estimator with $t_{G-1}$ (Section 7.4).
- The DK z is valid only if the critic is trained on data independent of the eval data
  (Proposition 7.5): with ticker splits on common dates it is anti-conservative too.

**11.3 `twosample/relative.py`.**
- Use the centred bootstrap p-value $(1+\#\{\Delta^*_b-\Delta\ge\Delta\})/(1+B)$. `extra["ci"]`
  is the percentile interval, which Proposition 6.2(4) now describes.
- Subsample $G$ to the size and per-ticker composition of $R_{\rm past}$, so that both critics
  face equally hard problems (Remark 6.3), with the same hyperparameters.
- Offer a block-bootstrap option for time-ordered $R_{\rm new}$ eval windows (Remark 6.7).
- *Boundary size* (Proposition 6.8). Pilot with the shipped `relative_test` logic and
  `CriticTest(n_steps=400)`: $P_{\rm new}=N(0,I_{10})$, $Q_G=P_{\rm past}=N(0,1.2^2I_{10})$, 600
  windows each, 400 replications. Rejection rate $40/400=0.100$ at nominal 5% (exact 95% CI
  $0.072$–$0.134$) and $65/400=0.163$ at 10%. From $5\times10^4$ fresh draws per replication,
  $\delta/\sigma_N$ has mean $-0.05$, standard deviation $0.74$ and $\mathbb P(\delta>0)=0.48$,
  and the prediction $\mathbb E[1-\Phi(z_{1-\alpha}-\delta/\sigma_N)]$ of Proposition 6.8(a) is
  $0.086$ (5%) and $0.142$ (10%). The existing slow test `test_boundary_size_critic_slow` (40
  replications, lenient bound) cannot separate 5% from 10%; if it is kept, it should assert the
  anti-conservativeness rather than near-nominal size.
- Options, in order of cost (Remark 6.9): common random numbers for the two critics (same torch
  seed and mini-batch index stream); a seed ensemble or cross-fitting; a retraining variance
  term added to $\sigma_N^2$; a shared-critic variant with an exact permutation test at
  $Q_G=P_{\rm past}$.

**11.4 E2.**
- Simulate independent windows, each from its own burned-in path, so that A1 holds exactly and
  the window-product oracle is NP-optimal.
- For GARCH, integrate the window density over $\sigma_1^2$ by Monte Carlo (Section 5.4). If
  $\sigma_1^2$ is fixed instead, call the result a "quasi-oracle".
- Report $\operatorname{Corr}(\hat\ell,\ell^*)$ from `C2ST.fit_logit`.

**11.5 E1 and E3.**
- Use equal numbers of real and synthetic windows per ticker (Remark 1.6).
- Generate synthetic windows from independent paths.
- Use a **time-block split** of the TEST dates with a gap (about 14 training dates, a one-window
  gap, about 15 eval dates at $L=20$), or train the critic on a separate period, for every pooled
  absolute and relative test (Section 7.4). Do not use the ticker split for pooled tests.
- Add the simulated dependent-null placebo of Section 7.6, with **exactly** the split, grouping,
  ordering and sample sizes of E3.
- Treat the DK p-value with fixed-$b$ critical values as primary for pooled real-data tests, and
  only under the time-block (or separate-period) split; report its placebo size alongside.
- *Pilot* (100 tickers × 30 dates × $L=20$; real window $s_\tau u_{i\tau}z$ with a market factor
  $\log s_\tau\sim N(0,0.6^2)$ shared across tickers on a date, $\log u_{i\tau}\sim N(0,0.3^2)$,
  $z\sim N(0,I_{20})$; synthetic windows with independent $s',u',z'$, so the pooled null holds
  exactly; `CriticTest(n_steps=400, n_perm=500)`; 200 replications; rejection rates at 5%):

  | split | permutation | DK, normal | DK, fixed-$b$ (approx.) | mean DK $z$ |
  |---|---|---|---|---|
  | ticker split (DESIGN §3.7) | 0.450 | 0.215 | 0.135 | 0.91 |
  | ticker split, critic trained on an independent panel (control) | 0.175 | 0.070 | 0.055 | 0.10 |
  | time-block split (train dates 0–13, gap 14, eval 15–29) | 0.310 | 0.120 | 0.070 | 0.25 |

  DK ticker split against control: Fisher one-sided $p=2\times10^{-5}$; paired mean shift of the
  DK $z$: $0.81$ ($t=6.7$). The control's excess is DK's small-$T$ distortion; the rest is the
  bias of Proposition 7.5. The time-block split removes the bias; with only 15 eval dates the DK
  variance is too small (sd of $z$ about $1.5$), and fixed-$b$ critical values (Kiefer–Vogelsang
  Bartlett polynomial, $b=(H+1)/T$) bring the rate to $0.07$ (exact 95% CI about $0.04$–$0.11$).
  $t_{14}$ critical values with the Bartlett DK estimator give $0.11$. The permutation p-value is
  anti-conservative under every split, as expected under cross-sectional dependence.

**11.6 E4.**
- Besides replica-wise portfolios, report portfolios built from moments pooled over the $B$
  replicas, or weights averaged over replicas (Corollary 9.3(iii), Proposition 9.4).
- Report $\kappa_G=\operatorname{cond}(S_{\rm te}^{-1}\hat\Sigma_{\rm syn})$ and the Kantorovich bound
  as diagnostics.

**11.7 Verdict.** Require the Clopper–Pearson lower bound $\beta_L\ge0.8$, with $R\ge200$
power-check replications. Aggregate p-values over seeds by twice the median. Report the simulated
boundary size $\hat\alpha_{\rm bd}$ (Definition 8.1, Proposition 6.8) with its interval next to
every verdict, and describe BAD as controlled only conditionally on the trained critics
(Proposition 8.3(a)).

**11.8 Corrections and additions to DESIGN.md (for the lead author; DESIGN.md not edited).**
- §3.7 `z_hac`: the citation of Chung and Romano (2013) for robustness to serial dependence is
  incorrect. Cite Newey and West (1987), Andrews (1991) and Driscoll and Kraay (1998) for
  `z_hac`; cite Chung and Romano (2013) only for the studentised permutation under unequal i.i.d.
  laws (Remark 3.5(b), Section 7.5).
- §3.7 group split: for pooled panels on common dates, the ticker split couples the training and
  eval halves through the realised market path and biases every p-value towards rejection
  (Proposition 7.5). E3 should use a time-block split with a gap (or separate-period training).
- §3.7 `order_x`/`order_y`: specify the unit, consecutive window indices or date ranks, not
  trading days (Section 7.2).
- §3.7 critic statistic: "lower bound on W1 on the representation" means $W_1(A_\#P,A_\#Q)$ in
  standardised coordinates (Remark 2.3), not $W_1$ in return units.
- §1 decision rule / §3.7 relative test: the false-BAD rate at $Q_G=P_{\rm past}$ exceeds
  $\alpha$ (Proposition 6.8); the verdict needs the boundary size reported alongside (11.7).
- §3.7 relative test, "bootstrap 95% CI": it is the percentile interval (Proposition 6.2(4)).

---

## References

Cited author–year. The literature agent maintains `paper/references.bib`; entries marked
*(verify)* are ones whose bibliographic details I am less certain of.

- Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica* 59, 817–858.
- Anil, C., Lucas, J., Grosse, R. (2019). Sorting out Lipschitz function approximation. *ICML*.
- Barnard, G.A. (1963). Discussion of "The spectral analysis of point processes" by M.S. Bartlett. *JRSS B* 25, 294.
- Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B* 57, 289–300.
- Benjamini, Y., Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Ann. Statist.* 29, 1165–1188.
- Best, M.J., Grauer, R.R. (1991). On the sensitivity of mean-variance-efficient portfolios to changes in asset means. *Rev. Financ. Stud.* 4, 315–342.
- Bester, C.A., Conley, T.G., Hansen, C.B. (2011). Inference with dependent data using cluster covariance estimators. *J. Econometrics* 165, 137–151.
- Bickel, P.J., Freedman, D.A. (1981). Some asymptotic theory for the bootstrap. *Ann. Statist.* 9, 1196–1217.
- Blackwell, D. (1953). Equivalent comparisons of experiments. *Ann. Math. Statist.* 24, 265–272.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *J. Econometrics* 31, 307–327.
- Bounliphone, W., Belilovsky, E., Blaschko, M.B., Antonoglou, I., Gretton, A. (2016). A test of relative similarity for model selection in generative models. *ICLR*.
- Bradley, R.C. (2005). Basic properties of strong mixing conditions. A survey and some open questions. *Probab. Surveys* 2, 107–144.
- Carrasco, M., Chen, X. (2002). Mixing and moment properties of various GARCH and stochastic volatility models. *Econometric Theory* 18, 17–39.
- Cheng, X., Cloninger, A. (2022). Classification logit two-sample testing by neural networks for differentiating near manifold densities. *IEEE Trans. Inf. Theory* 68. *(verify pages)*
- Chopra, V.K., Ziemba, W.T. (1993). The effect of errors in means, variances, and covariances on optimal portfolio choice. *J. Portfolio Management* 19(2), 6–11.
- Chung, E., Romano, J.P. (2013). Exact and asymptotically robust permutation tests. *Ann. Statist.* 41, 484–507.
- Clopper, C.J., Pearson, E.S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26, 404–413.
- Cox, D.R. (1981). Statistical analysis of time series: some recent developments. *Scand. J. Statist.* 8, 93–115.
- de Jong, R.M. (1997). Central limit theorems for dependent heterogeneous random variables. *Econometric Theory* 13, 353–367.
- de Jong, R.M., Davidson, J. (2000). Consistency of kernel estimators of heteroscedastic and autocorrelated covariance matrices. *Econometrica* 68, 407–423.
- Doukhan, P. (1994). *Mixing: Properties and Examples.* Lecture Notes in Statistics 85, Springer.
- Dowson, D.C., Landau, B.V. (1982). The Fréchet distance between multivariate normal distributions. *J. Multivariate Anal.* 12, 450–455.
- Driscoll, J.C., Kraay, A.C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Rev. Econ. Stat.* 80, 549–560.
- Dudley, R.M. (1969). The speed of mean Glivenko–Cantelli convergence. *Ann. Math. Statist.* 40, 40–50.
- Dudley, R.M. (2002). *Real Analysis and Probability.* Cambridge University Press.
- Dufour, J.-M. (2006). Monte Carlo tests with nuisance parameters: a general approach to finite-sample inference and nonstandard asymptotics. *J. Econometrics* 133, 443–477.
- Durbin, J., Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press.
- Dwass, M. (1957). Modified randomization tests for nonparametric hypotheses. *Ann. Math. Statist.* 28, 181–187.
- Fournier, N., Guillin, A. (2015). On the rate of convergence in Wasserstein distance of the empirical measure. *Probab. Theory Relat. Fields* 162, 707–738.
- Francq, C., Zakoïan, J.-M. (2019). *GARCH Models*, 2nd ed. Wiley.
- Genon-Catalot, V., Jeantheau, T., Larédo, C. (2000). Stochastic volatility models as hidden Markov models and statistical applications. *Bernoulli* 6, 1051–1079.
- Goodfellow, I. et al. (2014). Generative adversarial nets. *NeurIPS*.
- Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., Smola, A. (2012). A kernel two-sample test. *JMLR* 13, 723–773.
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A. (2017). Improved training of Wasserstein GANs. *NeurIPS*.
- Hall, P. (1992). *The Bootstrap and Edgeworth Expansion.* Springer.
- Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57, 357–384.
- Hemerik, J., Goeman, J.J. (2018). Exact testing with random permutations. *TEST* 27, 811–825.
- Herrndorf, N. (1984). A functional central limit theorem for weakly dependent sequences of random variables. *Ann. Probab.* 12, 141–153.
- Heston, S.L. (1993). A closed-form solution for options with stochastic volatility. *Rev. Financ. Stud.* 6, 327–343.
- Hoeffding, W. (1952). The large-sample power of tests based on permutations of observations. *Ann. Math. Statist.* 23, 169–192.
- Horn, R.A., Johnson, C.R. (2013). *Matrix Analysis*, 2nd ed. Cambridge University Press.
- Ibragimov, I.A. (1962). Some limit theorems for stationary processes. *Theory Probab. Appl.* 7, 349–382.
- Janssen, A. (1997). Studentized permutation tests for non-i.i.d. hypotheses and the generalized Behrens–Fisher problem. *Statist. Probab. Lett.* 36, 9–21.
- Jeffreys, H. (1946). An invariant form for the prior probability in estimation problems. *Proc. R. Soc. Lond. A* 186, 453–461.
- Kallenberg, O. (2002). *Foundations of Modern Probability*, 2nd ed. Springer.
- Kantorovich, L.V. (1948). Functional analysis and applied mathematics. *Uspekhi Mat. Nauk* 3, 89–185.
- Kantorovich, L.V., Rubinstein, G.S. (1958). On a space of completely additive functions. *Vestnik Leningrad Univ.* 13, 52–59.
- Kiefer, N.M., Vogelsang, T.J. (2005). A new asymptotic theory for heteroskedasticity-autocorrelation robust tests. *Econometric Theory* 21, 1130–1164.
- Kim, I., Ramdas, A., Singh, A., Wasserman, L. (2021). Classification accuracy as a proxy for two-sample testing. *Ann. Statist.* 49, 411–434.
- Kullback, S., Leibler, R.A. (1951). On information and sufficiency. *Ann. Math. Statist.* 22, 79–86.
- Künsch, H.R. (1989). The jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17, 1217–1241.
- Lahiri, S.N. (2003). *Resampling Methods for Dependent Data.* Springer.
- Lehmann, E.L., Romano, J.P. (2005). *Testing Statistical Hypotheses*, 3rd ed. Springer.
- Liu, F., Xu, W., Lu, J., Zhang, G., Gretton, A., Sutherland, D.J. (2020). Learning deep kernels for non-parametric two-sample tests. *ICML*.
- Lopez-Paz, D., Oquab, M. (2017). Revisiting classifier two-sample tests. *ICLR*.
- Mallasto, A., Montúfar, G., Gerolin, A. (2019). How well do WGANs estimate the Wasserstein metric? arXiv:1910.03875. *(verify)*
- McShane, E.J. (1934). Extension of range of functions. *Bull. Amer. Math. Soc.* 40, 837–842.
- Meinshausen, N., Meier, L., Bühlmann, P. (2009). p-values for high-dimensional regression. *JASA* 104, 1671–1681.
- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *J. Financ. Econ.* 3, 125–144.
- Merton, R.C. (1980). On estimating the expected return on the market. *J. Financ. Econ.* 8, 323–361.
- Müller, A. (1997). Integral probability metrics and their generating classes of functions. *Adv. Appl. Probab.* 29, 429–443.
- Newey, W.K., West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55, 703–708.
- Newey, W.K., West, K.D. (1994). Automatic lag selection in covariance matrix estimation. *Rev. Econ. Stud.* 61, 631–653.
- Neyman, J., Pearson, E.S. (1933). On the problem of the most efficient tests of statistical hypotheses. *Phil. Trans. R. Soc. A* 231, 289–337.
- Olkin, I., Pukelsheim, F. (1982). The distance between two random vectors with given dispersion matrices. *Linear Algebra Appl.* 48, 257–263.
- Petzka, H., Fischer, A., Lukovnikov, D. (2018). On the regularization of Wasserstein GANs. *ICLR*.
- Phipson, B., Smyth, G.K. (2010). Permutation p-values should never be zero. *Stat. Appl. Genet. Mol. Biol.* 9(1), Art. 39.
- Politis, D.N., Romano, J.P. (1994). The stationary bootstrap. *JASA* 89, 1303–1313.
- Ramdas, A., García Trillos, N., Cuturi, M. (2017). On Wasserstein two-sample testing and related families of nonparametric tests. *Entropy* 19(2), 47.
- Romano, J.P. (1990). On the behavior of randomization tests without a group invariance assumption. *JASA* 85, 686–692.
- Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *J. Pharmacokinet. Biopharm.* 15, 657–680.
- Singh, K. (1981). On the asymptotic accuracy of Efron's bootstrap. *Ann. Statist.* 9, 1187–1195.
- Stanczuk, J., Etmann, C., Kreusser, L.M., Schönlieb, C.-B. (2021). Wasserstein GANs work because they fail (to approximate the Wasserstein distance). arXiv:2103.01678. *(verify)*
- Storey, J.D. (2002). A direct approach to false discovery rates. *JRSS B* 64, 479–498.
- van der Vaart, A.W. (1998). *Asymptotic Statistics.* Cambridge University Press.
- Varadarajan, V.S. (1958). On the convergence of sample probability distributions. *Sankhyā* 19, 23–26.
- Villani, C. (2009). *Optimal Transport: Old and New.* Springer.
- Vovk, V., Wang, R. (2020). Combining p-values via averaging. *Biometrika* 107, 791–808.
- Vuong, Q.H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica* 57, 307–333.
- Wellek, S. (2010). *Testing Statistical Hypotheses of Equivalence and Noninferiority*, 2nd ed. Chapman & Hall/CRC.
