# Notation and conventions for the paper

Binding for every section writer. Macros are defined in `paper/main.tex`; use them instead of
retyping (a later change of notation is then one line). If you need a new symbol or macro, ask the
outline owner; do not define macros inside section files. Notation follows `docs/theory.md`
wherever it is unambiguous; deviations are listed in the last section, with the reason.

## 1. Conventions

**Returns and periods.**
- Prices are $S_{it}$ (asset or ticker $i$, trading day $t$). Daily log-returns
  $r_{it}=\log S_{it}-\log S_{i,t-1}$, physical measure, one time step = one trading day. No
  annualisation anywhere except the Sharpe ratio (factor $\sqrt{252}$).
- $N$ = number of tickers / assets in a cross-section (100 in the E1/E3/E5 panel, 30 in an E4
  universe). Index $i=1,\dots,N$.
- \TRAIN{} = 2018-01-02 to 2022-12-30, $\Ttr=1258$ days ("real past", calibration).
  \TEST{} = 2023-01-03 to 2025-06-16, $\Tte=615$ days ("real new", evaluation). The data file's
  nominal end date is 2025-06-17; the last trading day is 2025-06-16. Write "2025-06-16".
  E5 alternative split: \TRAIN{} 2018-01-02 to 2021-12-31, \TEST{} 2022-01-03 to 2025-06-16.
- Per-ticker scale $s_i$ = standard deviation of ticker $i$'s \TRAIN{} returns. Real *and*
  synthetic returns of ticker $i$ are divided by the same $s_i$ before pooling.

**Windows.**
- A window is $L$ consecutive daily returns, non-overlapping (step $L$); default $L=20$
  (E3 per-ticker tests and E4: $L=5$; E5 varies $L\in\{5,10,20,60\}$). Univariate window
  $\in\R^{L}$, multivariate window $\in\R^{N\times L}$; $d$ = dimension ($L$ or $NL$).
- Window counts at $L=20$ for the 100-ticker panel: 30 per ticker in \TEST{} (3000), 62 per
  ticker in \TRAIN{} (6200). E4 at $(N,L)=(30,5)$: 123 windows per sample.
- Each window carries a **date rank** $\tau$ = rank of its first day on the common trading
  calendar (\TRAIN{} days $0,\dots,\Ttr-1$, \TEST{} days $\Ttr,\dots$). Synthetic windows over
  the \TEST{} horizon get the \TEST{} ranks. HAC lags are counted in windows, never in days.
- Each ticker contributes the same number of real and synthetic windows (theory Remark 1.6).

**Samples and laws (always this role assignment).**
- $X_1,\dots,X_n$: real windows of the comparison sample. In the absolute test on real data
  $X=\Rnew$ (real \TEST{} windows). Law $P$; in relative statements $\Pnew$.
- $Y_1,\dots,Y_m$: synthetic windows from the generator $G$. Law $Q$; in relative statements
  $\QG$.
- $Z_1,\dots,Z_k$: real past windows $\Rpast$ (real \TRAIN{} windows). Law $\Ppast$.
- Densities $p,q$ of $P,Q$ when they exist (E2 oracle).
- "Generator" = fitted model or resampling scheme $G$; "synthetic sample" = its simulated windows.
- In E2, "real" and "synthetic" are labels only: $X\sim P$, $Y\sim Q$ with $P,Q$ known.

**Split and training.**
- Training / evaluation halves: index sets $\Itr,\Iev$ (real), $\Jtr,\Jev$ (synthetic); sizes
  $n_t,n_e$ and $m_t,m_e$ (relative test: $k_t,k_e$ for $\Rpast$). Default 50/50.
- $\Gsig$ = $\sigma$-field of the training halves, the index sets and the training randomness
  $\xi$ (initialisation, mini-batches). Conditioning on $\Gsig$ fixes the critic.
- Protocol "iid" (independent windows, E2): random split, permutation p-values (exact).
- Protocol "dependent" (real data, long simulated paths: E1, E3, E4, E5, E2 part g):
  $K=10$ contiguous **time blocks** of equal length of each sample's own date range; whole
  blocks are assigned at random to the training or evaluation half (cut closest to half of the
  windows). Blocks interleave in time and there is **no gap** between training and evaluation
  blocks (code: `time_block_labels`, `train_eval_split`). Learned tests report the
  Driscoll-Kraay p-value $\phac$; MMD and \SigMMD{} the wild-bootstrap p-value $\pwild$; KS is
  reported and marked invalid.

**Statistics, levels, reporting.**
- Level $\alpha=0.05$ unless stated; tables also show $\alpha\in\{0.01,0.10\}$.
- One-sided tests: large $T$ (or $\Danc$) is evidence against the null. KS is two-sided.
- Rejection rates are reported as "rate [Wilson 95% interval]" (boundary sizes also
  Clopper-Pearson). Rates to 2 or 3 decimals as in the generated tables; never retype a number
  that is not in a generated table (`paper/tables/*.tex`); otherwise write `\todo{...}`.
- Seed combination: $\pmed=\min\{1,\,2\med_s p_s\}$ over the synthetic seeds $s$ (valid under
  arbitrary dependence; theory Remark 3.5(d)).
- Test names in text and tables: \WC{} (Wasserstein critic), \LRC{} (C2ST logit statistic),
  C2ST (accuracy), MMD, \SigMMD{}, KS, LR oracle. Write "the \WC{} test", "the anchored relative
  test", "the two-critic relative test", "the time-block split", "the Driscoll-Kraay (HAC)
  p-value", "the wild bootstrap".
- Verdicts: \GOOD, \BAD, \INCONCL, \INVALID{} (sans-serif macros).
- Models in text: GBM, Merton, GARCH-$t$ (or \GARCHt{} on first use), Heston, stationary
  bootstrap; multivariate: MV Gaussian, MV Student-$t$, CCC-GARCH-$t$, MV bootstrap,
  factor GARCH-$t$ (labels as in the generated tables).
- Experiments are called E1-E5 and E2-dep (the dependence-only families), as in DESIGN.

**Statements.** One shared counter numbered within sections (Proposition 4.3, Lemma A.2).
Environments: theorem, proposition, lemma, corollary, fact (standard result with a reference),
definition, assumption, example, remark. Heuristics are remarks whose title contains
"(heuristic)". Proofs of Section 4 and 8.1 statements go to Appendix A unless they are 5 lines
or less. Always `\cref` (`\Cref` at the start of a sentence), never "Proposition~\ref"; it
prints the environment name ("Proposition 4.3", "Appendix A", "Table 2"). Equations: `\eqref`
in running text ("(3.2)"); `\cref{eq:...}` prints "equation (3.2)". Labels: `sec:`, `app:`,
`def:`, `ass:`, `prop:`, `lem:`, `cor:`, `fact:`, `rem:`, `ex:`, `eq:`, `tab:`, `fig:`; the full
registry is in OUTLINE.md §12.

## 2. Symbol table

### 2.1 General probability and analysis

| Symbol | Macro | Meaning |
|---|---|---|
| $\E,\ \Prob$ | `\E`, `\Prob` | expectation, probability |
| $\Var,\Cov,\Corr$ | `\Var`, `\Cov`, `\Corr` | variance, covariance, correlation (operators) |
| $\Law(\cdot)$ | `\Law` | law of a random element; $\Law(G)$ = window law of the generator $=\QG$ |
| $\one\{A\}$, $\one$ | `\one` | indicator of $A$; vector of ones (portfolios) |
| $\R$, $\Nat$ | `\R`, `\Nat` | reals, naturals ($N$ is reserved for the number of assets) |
| $\norm{x}$, $\abs{x}$ | `\norm{x}`, `\abs{x}` | Euclidean (Frobenius) norm, absolute value; `\norm*{...}` scales |
| $\W(P,Q)$ | `\W` | Wasserstein-1 distance (Euclidean ground metric) |
| $\Lip(f)$, $\Lip_c$ | `\Lip` | Lipschitz constant; class $\{f:\Lip(f)\le c\}$ |
| $\Pspace{p}(\R^d)$ | `\Pspace{p}` | Borel probability measures with finite $p$-th moment |
| $\KL(P\Vert Q)$, $\Jeff(P,Q)$ | `\KL`, `\Jeff` | Kullback-Leibler; Jeffreys $\KL(P\Vert Q)+\KL(Q\Vert P)$ |
| $\TV$, $\MMD$ | `\TV`, `\MMD` | total variation; maximum mean discrepancy |
| $Pf$ | (none) | $\int f\,dP$ |
| $\Phi$, $\varphi$, $z_{1-\alpha}$ | (none) | standard normal cdf, density, $1-\alpha$ quantile |
| $\eqd$, $\dto$, $\pto$, $\indep$ | `\eqd`, `\dto`, `\pto`, `\indep` | equality in law; convergence in law; in probability; independence |
| $\argmin,\argmax,\diag,\Tr,\med,\cond,\supp$ | same names | operators (trace is `\Tr`) |
| i.i.d. | `\iid` | text abbreviation |

### 2.2 Data, periods, samples

| Symbol | Macro | Meaning | Code |
|---|---|---|---|
| $S_{it}$, $r_{it}$ | | price, daily log-return | `data.log_returns` |
| $N$ | | tickers / assets in the cross-section (100 panel; 30 per E4 universe) | `config.N_PANEL` |
| \TRAIN, \TEST | `\TRAIN`, `\TEST` | calibration and evaluation periods | `config.TRAIN/TEST` |
| $\Ttr$, $\Tte$ | `\Ttr`, `\Tte` | their lengths in days (1258, 615) | |
| $s_i$ | | \TRAIN{} std of ticker $i$ (fixed scale for real and synthetic) | `data.train_vol` |
| $L$, $d$ | | window length (default 20); window dimension ($L$ or $NL$) | `config.WINDOW_L` |
| $\tau$ | | date rank of a window's first day (time stamp for HAC / wild bootstrap) | `order_x`, `order_y` |
| $\Rnew$, $\Rpast$ | `\Rnew`, `\Rpast` | real \TEST{} windows, real \TRAIN{} windows | |
| $G$ | | generator (fitted model or resampling scheme) | `models` |
| $X_i$, $Y_j$, $Z_l$ | | real-new, synthetic, real-past windows | `X`, `Y`/`G`, `R_past` |
| $n$, $m$, $k$ | | sizes of the $X$, $Y$, $Z$ samples | |
| $P$, $Q$ | | laws of one real / one synthetic window (absolute test) | |
| $\Pnew$, $\QG$, $\Ppast$ | `\Pnew`, `\QG`, `\Ppast` | window laws in the relative test | |
| $p$, $q$ | | densities of $P$, $Q$ (E2 oracle) | `window_loglik` |

### 2.3 Split, critic and statistics

| Symbol | Macro | Meaning | Code |
|---|---|---|---|
| $\Itr,\Iev,\Jtr,\Jev$ | `\Itr` etc. | training / evaluation index sets (real; synthetic) | `train_eval_split` |
| $n_t,n_e,m_t,m_e,k_t,k_e$ | (none) | training / evaluation sizes ($n_e$ = real eval size) | `n_eval_x` etc. |
| $\Ptr f,\Pev f,\Qtr f,\Qev f$ | `\Ptr`, `\Pev`, `\Qtr`, `\Qev` | empirical means over the halves | |
| $K$ | | number of time blocks (default 10) | `time_blocks` |
| $\Gsig$, $\xi$ | `\Gsig` | training $\sigma$-field; training randomness | |
| $A$, $m_A$, $\Lambda$ | | input standardisation $A(x)=\Lambda^{-1}(x-m_A)$, mean / positive diagonal scale of the pooled training half (per coordinate; per channel for $(N,L)$ windows) | `Standardizer` |
| $P_A$, $Q_A$ | | standardised laws $A_\#P$, $A_\#Q$; "W1 lower bound" refers to $\W(P_A,Q_A)$ | |
| $\fhat=\ghat\circ A$ | `\fhat`, `\ghat` | trained critic; network part | `Witness` |
| $\lgp$ | `\lgp` | gradient-penalty weight (10) | `gp_lambda` |
| $T$ | | critic statistic $\Pev\fhat-\Qev\fhat$ | `TestResult.statistic` |
| $\Dcrit(f)$ | `\Dcrit` | critic discrepancy $Pf-Qf$; $\E[T\mid\Gsig]=\Dcrit(\fhat)$ | |
| $\lhat$, $\lstar$ | `\lhat`, `\lstar` | learned logit; Bayes logit $\log(p/q)$ (+ prior constant) | `C2ST.fit_logit` |
| $\Tlogit$, $\Tacc$ | `\Tlogit`, `\Tacc` | \LRC{} statistic $\Pev\lhat-\Qev\lhat$; C2ST balanced accuracy | `c2st.py` |
| $\TMMD$, $\Tsig$, $\TKS$ | `\TMMD`, `\Tsig`, `\TKS` | unbiased $\MMD_u^2$; MMD on signature features; KS on pooled returns | `mmd.py`, `signature.py`, `ks.py` |
| $\varsigma$ | | Gaussian-kernel bandwidth (median heuristic $\varsigma_{\mathrm{med}}$) | `median_heuristic` |
| $\Sig$ | `\Sig` | truncated signature (depth 4) of the time-augmented cumulative-return path | `signature_features` |
| $\TLR$, $\bNP$ | `\TLR`, `\bNP` | oracle statistic $\sum_j\log q(Y_j)/p(Y_j)$; Neyman-Pearson power | `lr_oracle_test` |
| $B$ | | permutations (1000 learned; 500 kernel) or bootstrap replicates (500) | `n_perm`, `n_boot` |
| $T_b$ | | statistic after the $b$-th label permutation | |
| $\pperm$, $\phac$, $\pwild$ | `\pperm`, `\phac`, `\pwild` | permutation, Driscoll-Kraay (normal), wild-bootstrap p-values | `p_perm`, `p_hac`, `p_wild` |
| $z$, $\zhac$ | `\zhac` | difference of means over its i.i.d. / HAC standard error | `z`, `z_hac` |
| $\gamma(h)$, $\LRV$ | `\LRV` | autocovariance at lag $h$ (in windows); long-run variance $\sum_h\gamma(h)$ | `hac_var_mean` |
| $H$ | | Bartlett truncation lag (Andrews AR(1) plug-in; Newey-West rule as robustness) | `hac_lags` |
| $\mathcal S_\tau$ | | Driscoll-Kraay date sum $\sum_{k:\tau(k)=\tau}(a_k-\bar a)$ | |
| $a_k$, $b_k$ | | critic values on real / synthetic eval windows, in time order | |
| $\zeta_\tau$, $\ell_w$ | | wild-bootstrap AR(1) multipliers (shared by windows of date $\tau$); block length | `ar1_multipliers` |

### 2.4 Relative tests and decision rule

| Symbol | Macro | Meaning | Code |
|---|---|---|---|
| $\Danc$ | `\Danc` | anchored statistic: mean of $\fhat$ on $\Rpast$ eval windows minus mean on $G$ eval windows ($\fhat$ trained on $\Rnew$ vs pool $\{G_{\trn},\Rpast_{,\trn}\}$) | `relative_test(method="anchored")` |
| $\delta$ | | anchored conditional target $\Ppast\fhat-\QG\fhat$; null $\delta\le0$ | |
| $\Dtwo$ | `\Dtwo` | two-critic statistic $\Dcrit_G-\Dcrit_P$ (secondary) | `method="two_critic"` |
| $\Dcrit_G$, $\Dcrit_P$, $\delta^{\mathrm{2c}}$ | | critic discrepancies of the two critics; two-critic target $\Dcrit_G-\Dcrit_P$ | |
| $\sigma_N$ | | evaluation standard deviation of $\Dtwo$ (theory Prop. 6.2) | |
| $p_s$, $\pmed$ | `\pmed` | p-value of synthetic seed $s$; $\min\{1,2\med_s p_s\}$ | `median_pvalue` |
| $\Cdesign$, $\vartheta^*$ | `\Cdesign` | reference design and pre-specified deviation of the power check (synthetic volatility $\times1.2$) | `power_check` |
| D1, D2 | | power designs: per-ticker GARCH-$t$ with stationary $\Rpast^*$; real \TRAIN{} volatility path | `e3 power`, `realvol` |
| $\hat\beta$, $\betaL$, $\bmin$ | `\betaL`, `\bmin` | estimated power of the rule; Clopper-Pearson lower bound; threshold 0.8 | |
| $\abd$ | `\abd` | boundary size (rejection rate at $\Law(G)=\Law(\Rpast)$) | `boundary_size` |
| $V$ | | verdict \GOOD{} / \BAD{} / \INCONCL{} (\INVALID{} for the bootstrap) | `pipeline.verdict` |

### 2.5 Models

| Symbol | Meaning |
|---|---|
| $\mu,\sigma$ | GBM drift and volatility (daily) |
| $\lambda_J,\mu_J,\sigma_J$ | Merton jump intensity (per day), jump mean and sd |
| $\omega,a,b,\nu$ | GARCH(1,1)-$t$: $r_t=\mu+\sigma_tz_t$, $\sigma_t^2=\omega+a\varepsilon_{t-1}^2+b\sigma_{t-1}^2$, $z_t$ standardised Student-$t_\nu$; persistence $a+b$ (cap 0.995 with variance targeting) |
| $v_t,\kappa_v,\theta_v,\xi_v,\rho_v$ | Heston variance, mean-reversion rate, long-run variance, vol of variance, leverage correlation (daily units); $c=\xi_v/\sqrt{2\kappa_v\theta_v}\le\sqrt{1.5}$ |
| $m_t$, $\beta_i$, $e_{it}$ | factor GARCH-$t$: $r_{it}=\mu_i+\beta_im_t+e_{it}$, equal-weight panel factor $m_t$ |
| $\vartheta$ | E2 deviation parameter of a family (volatility factor $1+\vartheta$; mixing fraction $\vartheta$); family (b) is indexed by $\nu$, family (c) by $a+b$, E2-dep block by $L_b$ |
| $L_b$ | E2-dep block length inside a window ($L_b\in\{20,10,5,2,1\}$) |

### 2.6 Likelihood benchmarks and stylized facts

| Symbol | Meaning | Code |
|---|---|---|
| $g(r_t\mid\mathcal F_{t-1})$ | one-step predictive density of a likelihood model, filter started at the end of \TRAIN | `loglik_obs(r, r_history)` |
| $\mathrm{LS}(G)$ | mean out-of-sample predictive log-score on \TEST{} | `predictive_loglik` |
| $u_t$ | PIT values $G(r_t\mid\mathcal F_{t-1})$ | `pit` |
| Berkowitz, PIT-KS | LR test of $\Phi^{-1}(u_t)$ i.i.d. $N(0,1)$ vs Gaussian AR(1); KS of $u_t$ vs $U(0,1)$ | `berkowitz_test`, `pit_ks_test` |
| AG | Amisano-Giacomini test (HAC Vuong statistic on log-score differences, no parameter penalty) | `vuong_test(hac=True)` |
| $\hat\rho_{\abs{r}}(h)$ | sample ACF of $\abs{r_t}$ at lag $h$ days | `stylized.acf` |
| $\hat\alpha_{\mathrm{Hill}}$ | Hill tail index (5% tails) | `stylized.hill` |

### 2.7 Portfolios (Section 8)

| Symbol | Macro | Meaning |
|---|---|---|
| $w$, $\mathcal W$ | | weights, feasible set $\{\one'w=1\}$ (and $w\ge0$ long-only) |
| $\hat\mu,\hat\Sigma$, $\SigLW$ | `\SigLW` | sample mean and covariance; Ledoit-Wolf (identity target) |
| $\wgmv$, $\wmv$ | `\wgmv`, `\wmv` | GMV portfolio; mean-variance $\argmax_w w'\hat\mu-\tfrac\gamma2w'\hat\Sigma w$ |
| $\gamma$ | | risk aversion, $\gamma\in\{1,5,20\}$ ($\gamma=5$ for CE of GMV and 1/N) |
| $V(w)$ | | out-of-sample variance $w'\Sigma w$ |
| $\CE$, $\SR$ | `\CE`, `\SR` | daily certainty equivalent (mean $-\tfrac\gamma2$ var); annualised Sharpe ratio |
| TRTR, TSTR | | train on real / on synthetic, test on real |
| $B$ (local) | | synthetic replicas per generator (20) |
| $\kappa_G$ | | $\cond(S_{\mathrm{te}}^{-1}\hat\Sigma_{\mathrm{syn}})$; Kantorovich bound $(1+\kappa_G)^2/(4\kappa_G)$ |

## 3. Local symbols (may be reused inside one statement, defined there)

Greek letters are scarce. Inside a single proposition, example or proof a writer may reuse the
following, provided the statement defines them: $\kappa$ ($\lim n/m$ in the local-power result;
condition number in Section 8), $\rho$ (evaluation fraction in the local-power result), $\pi$
(class prior; mixture weights), $\eta,\varepsilon$ (margins), $h$ (score function in the
local-power result; lag elsewhere), $\theta$ (parameter of a local family $Q_\theta$), $\phi$
(AR(1) coefficient in the naive-permutation heuristic), $\iota$ (class label in the Bayes-logit
result). Never reuse $\alpha$, $\beta$ (level, power), $T$, $\Danc$, $\delta$, $L$, $N$, $n$, $m$,
$k$, $K$, $B$ globally.

## 4. Deviations from `docs/theory.md` and DESIGN (with reasons)

1. **Prices $S_{it}$, not $P_t$.** theory.md §0 writes $r_t=\log P_t-\log P_{t-1}$, which clashes
   with the window law $P$.
2. **$\Danc$ is the anchored statistic; the two-critic statistic is $\Dtwo$.** In theory.md §6
   $\Delta$ is the two-critic statistic and the anchored one is $\Delta_{\rm sh}$ (Remark 6.9(4)).
   The anchored test is primary (DESIGN §1, Change log), so it gets the plain symbol, matching
   the code (`statistic` of `relative_test`, default `method="anchored"`). The two-critic targets
   $d_G,d_P$ of theory.md become $\Dcrit_G,\Dcrit_P$.
3. **Critic discrepancy $\Dcrit(f)=Pf-Qf$**, written $\Delta(\hat f)$ in theory.md Prop. 2.5,
   renamed to free $\Delta$ (item 2).
4. **Standardisation $A(x)=\Lambda^{-1}(x-m_A)$**, written $D^{-1}(x-c)$ in theory.md Remark 2.3,
   because $D$ is the critic discrepancy and $c$ the Lipschitz constant.
5. **Long-run variance $\LRV$** instead of theory.md's $\omega^2$ (clash with the GARCH intercept
   $\omega$). Driscoll-Kraay date sums $\mathcal S_\tau$ instead of $S_\tau$ (clash with prices).
6. **Day counts $\Ttr,\Tte$** instead of theory.md §9's $T_0,T_1$ ($T$ is the statistic);
   $T_{\mathrm{tr}}$ and $T_{\mathrm{te}}$ also index the Section 8 sample moments
   ($S_{\mathrm{te}}$ = \TEST{} sample covariance, as in theory.md).
7. **Heston parameters $\kappa_v,\theta_v,\xi_v,\rho_v$** (code: `kappa, theta, xi, rho`),
   subscripted because $\kappa,\theta,\rho,\xi$ have other meanings. Merton intensity $\lambda_J$
   and gradient-penalty weight $\lgp$ are subscripted for the same reason.
8. **Deviation parameters $\vartheta$ (E2) and $\vartheta^*$ (power check)** instead of DESIGN's
   "delta" and theory.md's $\delta^*$, because $\delta$ is the anchored conditional target. E2-dep
   block length $L_b$ instead of the script's `b` (clash with GARCH $b$).
9. **Number of time blocks $K$** (code `time_blocks`), per-ticker multiple-testing count written
   $N$ (theory.md §8.2 uses $K$ for tickers). Monte Carlo draws of the GARCH initial variance:
   $M$ (512 in E2), not the scripts' `K`.
10. **Kernel bandwidth $\varsigma$** (code `bandwidth`, "h" in DESIGN), because $h$ is a lag.
11. **Weak convergence $\dto$** ($\rightsquigarrow$, van der Vaart) instead of theory.md's
    $\Rightarrow$, to avoid confusion with implication in displayed arguments.
12. **HAC bandwidth.** theory.md §7.2 gives the Newey-West (1994) rule as default; the code
    default (and DESIGN Change log) is the Andrews (1991) AR(1) plug-in. The paper describes the
    Andrews plug-in; Newey-West is the E5 robustness option.
