# Literature review and citation guide

Companion to `paper/references.bib` (128 entries). Written for the authors of
`paper/sections/*.tex`: what each reference contributes, where our paper differs from the closest
work, and which keys back which claim.

**Verification.** Every key in `references.bib` was checked on 2026-11-14 against an
authoritative record. Entries with a DOI were checked with Crossref (`api.crossref.org/works/<DOI>`).
NeurIPS, PMLR (ICML/AISTATS), ICLR, JMLR and arXiv entries were checked on the proceedings
listing (`proceedings.neurips.cc`, `proceedings.mlr.press`), the ICLR accepted-paper lists
(`iclr.cc`), `jmlr.org` or the arXiv API. The one journal article without a DOI (Liu and Rubin
1995, *Statistica Sinica*) was checked on the journal's own index. Page ranges that Crossref cuts
to the first page were completed from Semantic Scholar, RePEc/IDEAS or the publisher page. Fields
that could not be confirmed (for example NeurIPS page numbers before 2020) are left out, not
guessed. When the online and print years differ, the print year is used. The file compiles with
`natbib`/`plainnat` and BibTeX gives 0 warnings. Statements about what a paper does were checked
against its full text where this file quotes it (arXiv PDFs of Bai et al., Bühler et al.,
Gulrajani et al., Wang-Gao-Xie, Bounliphone et al., Cetingoz-Lehalle, Storchan et al.,
Franco-Pedroso et al.) and otherwise against the published abstract (Crossref or RePEc).

Legend: the bold key is the BibTeX key. "Us" means our paper.

---

## 1. GANs, integral probability metrics, optimal transport

- **goodfellow2014generative**: the GAN minimax game. Its Proposition 1 gives the optimal
  discriminator `D* = p_data / (p_data + p_g)`, so `logit D* = log p_data/p_g`. This is the
  identity behind our `lrcritic` (C2ST with the logit statistic) and its link to the
  likelihood-ratio oracle.
- **arjovsky2017wasserstein**: WGAN. The critic maximises `E_P f - E_Q f` over 1-Lipschitz
  `f`, which by Kantorovich-Rubinstein duality is W1. Our critic statistic is the held-out value of
  this objective.
- **gulrajani2017improved**: WGAN-GP, a gradient penalty on interpolates in place of weight
  clipping, with lambda = 10 and Adam with learning rate 1e-4. Their recommended Adam betas are
  (0, 0.9) (Algorithm 1 and Section 5); they use (0.5, 0.9) only in the LSUN robustness
  experiment (Appendix E). Our critic uses lambda = 10, lr = 1e-4 and betas (0.5, 0.9), so write
  "lambda and learning rate as recommended by Gulrajani et al.; betas (0.5, 0.9) as in their
  Appendix E", never "their default betas". The penalty is soft, so the Lipschitz constant is
  not enforced exactly (see mallasto2019well).
- **muller1997integral**: defines integral probability metrics `d_F(P,Q) = sup_{f in F} |E_P f - E_Q f|`
  and their generating classes. Our statistic is an empirical IPM over the critic class.
- **sriperumbudur2012empirical**: empirical estimation and consistency rates of IPMs (W1,
  Dudley, MMD). Reference for "an IPM over a restricted class is estimable at parametric rates,
  W1 itself is not".
- **villani2009optimal**: the reference text for optimal transport and Kantorovich-Rubinstein
  duality.
- **nguyen2010estimating**: variational (convex risk minimisation) estimation of f-divergences
  and of the likelihood ratio. Gives the theory behind "a classifier's logit estimates the log
  density ratio".
- **arora2017generalization**: the "neural net distance". With a finite-capacity discriminator
  the GAN objective is not W1 or JS but a weaker IPM that generalises from samples. Use it to
  state honestly what our statistic measures.
- **fournier2015rate**: the empirical measure converges in W1 at rate `n^{-1/d}` in dimension
  `d >= 3`. This is why unrestricted W1 between windows of length L = 20 would be useless as a test
  statistic, and why we restrict the critic class.
- **ramdas2017wasserstein**: Wasserstein two-sample tests and their relation to energy-distance
  and kernel tests. Prior work on W-based testing, without learned critics.
- **mallasto2019well** (arXiv): studies how well WGAN critics with approximate Lipschitz
  constraints estimate the true W1. The more accurate c-transform estimator is not the best
  generator objective, so a good critic is not a good W1 estimate. Supports our wording that T is "an IPM over the trained-critic class, a lower bound on
  `L * W1` for the critic's actual Lipschitz constant L", not "an estimate of W1".

## 2. Two-sample testing, permutation inference, multiple testing, HAC

- **gretton2012kernel**: the kernel MMD two-sample test (unbiased MMD^2, median-heuristic
  bandwidth, permutation or asymptotic null). This is our `MMDTest` baseline.
- **lopezpaz2017revisiting**: classifier two-sample test (C2ST). Train a classifier on one split
  and test whether held-out accuracy differs from 1/2. They proposed it for evaluating GANs. This
  is our `C2ST` baseline with `statistic="accuracy"`.
- **liu2020learning**: MMD with a deep kernel trained on a training split to maximise a power
  proxy, tested on the held-out split by permutation. They show C2ST is a special case. The
  strongest learned kernel baseline. We do not implement it, so say so if asked.
- **cheng2022classification**: statistic = difference of held-out mean logits of a trained
  network classifier, with power guarantees when the densities lie near low-dimensional
  manifolds. Our `lrcritic` *is* this statistic, so credit it to them.
- **kim2021classification**: accuracy-based tests are consistent, but in the Gaussian
  mean-shift case they are only `1/sqrt(pi)` as efficient as Hotelling-type tests. Accuracy
  throws magnitude away. This motivates witness/mean-difference statistics over accuracy.
- **bounliphone2016test**: the relative similarity test. H0: model 1 is at least as close as
  model 2 to a reference sample, using the difference of two MMDs and their joint asymptotic
  normality (shared reference sample). Our MMD relative test follows it, and inherits its
  conditions. Their Problem 1 assumes three independent i.i.d. samples and requires
  `Px != Py` and `Px != Pz`: the joint normal limit needs both MMDs to be non-degenerate. Our
  windows are serially dependent, and the configuration in which G, R_past and R_new all have the
  same law is degenerate (the first-order projection vanishes). There the normal approximation
  does not apply, and `relative.py` documents that its paired bootstrap is not reliable either
  (the test is then typically conservative). The paper should report that configuration
  separately, as conservative, or exclude it from size tables. Outside it, note that both
  Bounliphone's variance formula and the i.i.d. paired bootstrap ignore serial dependence; for
  time-ordered real windows report the HAC version (`relative.py` gives `z_hac` when `order_*`
  is passed).
- **chung2013exact**: for *independent i.i.d. samples*, a two-sample permutation test is exact
  when P = Q. When the null is only equality of a parameter (for example E_P f = E_Q f for a fixed
  f) and P differs from Q in other ways (unequal variances, unequal sample sizes), the plain
  permutation test can fail even asymptotically; permuting a *studentised* statistic restores
  asymptotic validity and keeps exactness when P = Q. It says nothing about serial dependence.
  Where it applies to us: a studentised permutation test of `E_P f = E_Q f` with the critic f
  fixed, when windows are i.i.d. (as in E2 with independent paths) but the null of interest is
  equal critic means rather than P = Q, so the two laws may differ in other ways (for example
  unequal variances of f). Our code permutes the *unstudentised* difference,
  which is exact for H0: P = Q under exchangeability; cite Chung and Romano only if a studentised
  permutation p-value is added. It does **not** justify `z_hac`, which is a normal approximation
  with a HAC variance (see the HAC entry below).
- **janssen1997studentized**: the earlier result that studentised permutation tests are
  asymptotically valid for the Behrens-Fisher problem (unequal variances). Cite with
  chung2013exact.
- **romano2022permutation**: studentised permutation tests under weak dependence (mixing and
  moment conditions), exact for i.i.d. data and asymptotically valid for testing zero
  autocorrelation of a stationary series. It is the closest permutation-type result under
  time-series dependence, but it is about autocorrelation, not two-sample mean equality. Cite it
  only as "related", if a referee asks for a permutation argument under dependence.
- **lehmann2022testing** (4th ed.): randomisation and permutation tests, exactness, the
  Neyman-Pearson lemma (our LR oracle).
- **hemerik2018exact**: Monte Carlo permutation p-values `(1 + #{T_b >= T}) / (1 + B)` give
  exact level with randomly drawn permutations.
- **phipson2010permutation**: the same `+1` correction, and why permutation p-values should never
  be zero.
- **benjamini1995controlling**: BH FDR control for the per-ticker tests in E3.
- **benjamini2001control**: BH is valid under positive regression dependence and needs a log
  correction (BY) under arbitrary dependence. Cite it when a referee asks about cross-ticker
  dependence (common market factor) in the per-ticker tests.
- **kirchler2020two**: location tests on learned neural representations, with a held-out split
  when no auxiliary data are available. An early "learned features, then a simple mean test on
  held-out data" design.
- **kubler2022witness**: the *witness two-sample test*. Learn a real-valued witness on a training
  split, then test equality of its means on the held-out split, with a permutation null. **This
  is the general template our critic test instantiates** (with a WGAN-GP neural witness).
- **wang2022two** (AISTATS 2022): two-sample test with the *kernel projected Wasserstein*
  distance. A nonlinear (kernel) projector into a low-dimensional space is trained on a training
  split to maximise the Wasserstein distance between the projected samples, and a permutation
  test is run on the held-out split with the projector fixed (their Algorithm 1, "Permutation
  two-sample test using the KPW distance"). **The split-then-permute structure of a Wasserstein
  test is theirs, not ours.** Difference: their statistic is the empirical W1 between projected
  samples in dimension d, with a kernel projector; ours is the held-out mean difference of a
  neural Lipschitz critic trained with a gradient penalty (a one-dimensional witness, no
  empirical OT on the eval split). **wang2021two** (ISIT 2021) is the companion paper with a
  linear projection.
- **hu2026neural** (arXiv, Jan 2026): "Neural Wasserstein two-sample tests". Projection-1-
  Wasserstein with learned projections and a deep-network approximately 1-Lipschitz witness,
  max-type aggregation, asymptotically pivotal calibration (no resampling). The closest paper in
  the *witness* (a deep approximately 1-Lipschitz network), while wang2022two is the closest in
  the *calibration* (held-out permutation). Both must be cited.
- **lloyd2015statistical**: model criticism with the MMD witness, comparing data with samples from
  a fitted model to find where it fails. A conceptual ancestor of "use a two-sample test to
  criticise a generator".
- **chwialkowski2014wild**: kernel tests under temporal dependence, where the i.i.d. permutation
  null fails and a wild bootstrap is used. Relevant to our serially dependent windows.
- **jitkrittum2018informative**: linear-time relative goodness-of-fit tests between two models
  (Rel-UME, Rel-FSSD), with interpretable locations where one model fits better. Prior relative
  test besides Bounliphone et al.
- **gerber2024likelihood**: likelihood-free hypothesis testing. Decide which of two simulators
  produced the data, with `n` simulated and `m` real samples, and a sharp `n*m` sample-complexity
  trade-off. The theoretical framing for our relative test with simulated G.
- **HAC and `z_hac`.** `z_hac` is `T / se_hac`, compared with N(0,1). Its justification is a CLT
  for the mean of f(window) over a weakly dependent (mixing) sequence plus a consistent long-run
  variance, not a permutation argument. Keys:
  - **newey1987simple**: the Bartlett-kernel HAC estimator (used in `critic.py`, `relative.py`
    and `likelihood.vuong_test`).
  - **andrews1991heteroskedasticity**: consistency of kernel HAC estimators under
    fourth-cumulant / mixing conditions (used by `docs/theory.md`, condition D3), and the
    data-dependent AR(1) plug-in bandwidth. As of 2026-11-14 the AR(1) plug-in for the Bartlett
    kernel is the **default** lag in `critic.hac_var_mean` (`hac_lags=None` or `"andrews"`), so
    it is the bandwidth behind `z_hac` of the critic, C2ST, MMD and relative tests.
  - **newey1994automatic**: the rule-of-thumb lag `floor(4 (T/100)^(2/9))`, used by
    `likelihood.newey_west_variance` (Vuong test with `hac=True`) and by `hac_lags="nw"` in
    `critic.py`. Cite whichever rule the reported numbers actually used.
  - **driscoll1998consistent**: summing centred values within each date before Newey-West, which
    makes the variance robust to cross-sectional dependence between windows of different tickers
    at the same date (the pooled E3 setting). The code calls this the Driscoll-Kraay estimator.
  - **herrndorf1984functional**: (functional) CLT for strongly mixing sequences, the limit theorem
    behind the normal approximation.
  - **carrasco2002mixing**: GARCH(1,1) and a class of stochastic-volatility models are
    geometrically beta-mixing with finite moments under easy-to-check conditions. This is what
    makes the mixing assumption credible for windows of GARCH-type returns. For our
    Euler-discretised Heston chain it is plausible but not checked.
- **hall1991two**: two guidelines for bootstrap tests: resample in a way that reflects the null
  (centre the bootstrap distribution at the estimate), and bootstrap a pivotal (studentised)
  statistic. `relative.py` follows the first guideline (centred p-value). It bootstraps the
  unstudentised Delta, so it does not follow the second; say so, or add a studentised version.
- **diebold1995comparing**: comparing two forecasts through a loss differential with a HAC
  t-statistic. Our relative test is the distributional analogue (loss = critic distance to
  R_new).

## 3. Path signatures

- **lyons1998differential**: rough paths. The signature as the canonical feature map of a path.
- **chevyrev2016primer** (arXiv 2016; revised as a Springer chapter, 2025): practitioner primer
  covering time augmentation, lead-lag and truncation. Matches the transforms in `signature.py`.
- **kiraly2019kernels**: signature kernels for sequential data.
- **chevyrev2022signature**: (robust) signature moments characterise laws of stochastic
  processes. Gives a signature MMD and a two-sample test for laws of paths. **Caveat for us:** our
  `SigMMD` uses truncated depth-4 signature features with a Gaussian kernel and no tensor
  normalisation, so it does not inherit their characteristic property. Call it a "truncated
  signature-feature MMD in the spirit of Chevyrev and Oberhauser (2022)".
- **issa2023nonadversarial**: trains neural SDE market generators with signature-kernel scoring
  rules instead of an adversarial critic. Shows that signature MMDs are already used *as training
  losses* in finance, which is a reason to evaluate with an independent test.

## 4. Synthetic financial data and evaluation of generative models

- **wiese2020quant**: Quant GANs (TCN generator) for S&P 500 returns. Evaluation by stylized facts
  and distributional metrics, with no formal test at a controlled level.
- **takahashi2019modeling**: GANs reproduce heavy tails, volatility clustering and leverage.
  Evaluation is visual or by stylized-fact statistics.
- **kondratyev2019market**: the "market generator" (restricted Boltzmann machine). Industry
  motivation for synthetic market data. Also published in *Risk* (Cutting Edge, Feb 2020).
- **buehler2020data**: signature-based (conditional) VAE market generator for small-data
  regimes, trained on S&P 500 paths. They argue explicitly that "two-sample tests may be better
  suited as performance evaluation metrics than distributional metrics and divergences"
  (Section 2.2), and evaluate the generator with the Chevyrev-Oberhauser signature-MMD two-sample
  test (with the distribution-free threshold `c_alpha = 4 sqrt(-log(alpha) / n)`, valid for
  every null law and hence conservative) and with marginal KS tests (Section 4.1.1: e.g. weekly
  paths pass the
  signature test up to the 99.998% level, KS p-value 0.57). There is no size or power analysis.
  **So formal two-sample testing of market generators is not new**: do not write that the field
  only "eyeballs stylized facts". What is new in ours is a learned critic test whose level and
  power are validated (against ground truth and the LR oracle), a relative null and a link to
  utility.
- **bai2026calibrating** (ACM TOMACS, online 2025; arXiv 2105.12893, 2021): calibration of
  non-identifiable simulation models, including the ABIDES limit-order-book market simulator, by
  *eligibility sets* with frequentist guarantees. Simulated and real outputs are compressed by a
  feature extractor (auto-encoder, GAN discriminator output or hidden layer, **WGAN critic
  output** or hidden layer), and features are compared by Bonferroni-corrected KS statistics
  (SKS) or mean differences (SSMD, ESMD) against computable critical values. They find the critic
  output the most conservative feature and recommend hidden layers. **Prior use of a WGAN critic
  to validate a financial simulator against real data**; see the competitors section.
- **francopedroso2018ets** (arXiv): the "ETS challenges". Realism of simulated financial return
  series posed as a real-versus-simulated *classification* competition; the best features and
  classifiers indicate what the simulators miss. Classifier accuracy only, no null distribution.
- **storchan2021learning** (arXiv): MAS-GAN. A discriminator trained to tell real from simulated
  price and volume series is used as the objective to calibrate a multi-agent market simulator.
  Discriminator output as a realism score, with no test.
- **vyetrenko2020get**: "Get real". Stylized-fact realism metrics for limit-order-book market
  simulators, applied to real and simulated market data. The descriptive, stylized-fact side of
  simulator validation.
- **ni2021sig**: Sig-Wasserstein GANs (ICAIF 2021). Replace the critic with a signature-space W1
  surrogate. **liao2024sig**: the extended conditional version in *Mathematical Finance*. Cite
  both when saying "signature W1 is used for training".
- **yoon2019time**: TimeGAN. Introduces the *discriminative score* (post-hoc 2-layer LSTM
  classifier, held-out classification error) and the *predictive score* (train next-step
  predictor on synthetic, MAE on real). Includes a daily Google stock dataset.
- **esteban2017real** (arXiv): RCGAN. Introduces *train on synthetic, test on real* (TSTR) and
  TRTS.
- **assefa2020generating**: J.P. Morgan AI Research overview of synthetic data in finance, its
  uses (back-testing, sharing, privacy) and evaluation pitfalls.
- **alaa2022faithful**: sample-level fidelity, diversity and generalisation metrics. These are
  descriptive scores with no null distribution.
- **theis2016note**: log-likelihood, sample quality and other criteria can disagree sharply, so
  evaluation must match the intended use. Motivates RQ4 and RQ5.
- **cont2026tail**: Tail-GAN. The discriminator uses a tail-risk (VaR/ES of portfolios) based
  loss. An example of a use-targeted metric to contrast with our use-agnostic test.
- **athey2024using**: WGAN-generated data that mimics real datasets, used to rank econometric
  estimators in Monte Carlo studies (*Journal of Econometrics*). The econometric precedent for
  "is the synthetic data good enough for a downstream decision?".
- **stenger2024evaluation**: survey of evaluation measures for synthetic time series. Cite it for
  how heterogeneous evaluation practice is.
- **buehler2019deep**: deep hedging. A main consumer of market generators.
- **bailey2017probability**: backtest overfitting. Motivates synthetic data for back-testing.
- **cetingoz2026synthetic** (*Quantitative Finance* 2026; arXiv 2501.03993): whether synthetic
  returns help build Markowitz and minimum-variance portfolios. They prove that the initial
  sample size limits what a generator can add, warn against generating far more data than was
  available (it can bias the estimated statistic), show that generic generators fit the
  high-variance principal components that matter little for minimum-variance portfolios, compare
  with a block bootstrap, and argue for use-targeted evaluation. See Section 8 and competitors.

## 5. Stylized facts and return models

- **cont2001empirical**: the stylized facts we measure in `stylized.py`: heavy tails,
  volatility clustering, leverage, aggregational Gaussianity, absence of linear autocorrelation.
- **mandelbrot1963variation**, **fama1965behavior**: heavy tails of speculative returns. Fama
  also documents near-zero return autocorrelation.
- **black1973pricing**: GBM / log-normal benchmark (our `GBM`).
- **merton1976option**: jump-diffusion (our `Merton`). The daily return density is an exact
  Poisson mixture of normals.
- **engle1982autoregressive**, **bollerslev1986generalized**: ARCH and GARCH.
  **bollerslev1987conditionally**: GARCH with Student-t innovations (our `GARCHt`).
- **heston1993closed**: stochastic-volatility model (our `Heston`, simulated under P).
  **lord2010comparison**: the full-truncation Euler scheme (`v^+ = max(v, 0)` in drift and
  diffusion) we use, and its comparison with other biased schemes.
- **bollerslev1990modelling**: constant conditional correlation (our `CCCGARCHt`).
  **joe2005asymptotic**: two-stage estimation (inference functions for margins: univariate MLE
  first, dependence parameters second) and its asymptotic efficiency; this is how `CCCGARCHt` is
  fitted. **engle2002dynamic**: DCC. Cite it as the natural extension we do not implement.
- **liu1995ml**: ML estimation of the multivariate t by EM, ECM and ECME; our `MVStudentT` uses
  ECME (degrees of freedom updated on the observed likelihood).
- **hill1975simple**: Hill tail-index estimator (5% tails in `stylized.summary`).
- **hansen2005forecast**: GARCH(1,1) is hard to beat out of sample. This justifies GARCH-t as the
  strong parametric benchmark.
- **aitsahalia2007maximum**: (approximate) maximum likelihood for stochastic-volatility models.
  **Wording for us:** Heston has no closed-form transition density for returns alone. Its
  likelihood needs filtering or approximation, so it is "not available in closed form", not
  "intractable in principle". The stationary bootstrap has no parametric likelihood at all.

## 6. Likelihood-based and likelihood-free evaluation and estimation

- **diebold1998evaluating**: PIT `u_t = F_t(r_t)` is i.i.d. U(0,1) under a correct density
  forecast (our `pit_ks_test`).
- **berkowitz2001testing**: `Phi^{-1}(u_t)` and an LR test of N(0,1) i.i.d. against a Gaussian
  AR(1) (our `berkowitz_test`).
- **vuong1989likelihood**: LR test for non-nested, possibly misspecified models (our
  `vuong_test`). Our relative test is the likelihood-free analogue.
- **amisano2007comparing**: weighted out-of-sample log-likelihood ratio tests for density
  forecasts (focus on tails). Cite for predictive log-likelihood comparisons.
- **giacomini2006tests**: conditional predictive ability. It evaluates the *forecasting method*
  (model plus estimation window) under non-stationarity, which is the same philosophy as our
  relative test (generator calibrated on R_past against R_past itself).
- **gneiting2007strictly**: the log score is strictly proper. This justifies predictive
  log-likelihood as the likelihood benchmark.
- **mcfadden1989method**, **duffie1993simulated**: simulated method of moments. Duffie-Singleton
  is specifically for Markov models of asset prices (our Heston SMM fit).
- **gourieroux1993indirect**, **smith1993estimating**: indirect inference, the simulation-based
  estimator for models with intractable likelihoods (CREST).
- **gallant1996which**: EMM. Match the score of an auxiliary model, which answers "which
  moments to match". The critic can be read as a *learned* moment chosen to be maximally
  discriminating.
- **gourieroux1997simulation**: monograph on simulation-based econometric methods (CREST).
- **fermanian2004nonparametric**: nonparametric simulated maximum likelihood, using a kernel
  density of simulated draws as the likelihood. An alternative route to likelihood benchmarks for
  Heston.
- **gutmann2018likelihood**: likelihood-free inference by classification, where classifier
  accuracy serves as the discrepancy.
- **cranmer2015approximating** (arXiv): calibrated classifiers approximate likelihood ratios.
  Supports using the learned logit as an LR proxy in ground-truth studies
  (`C2ST.fit_logit` against the true log-LR).
- **cranmer2020frontier**: survey of simulation-based inference.
- **kaji2023adversarial**: adversarial estimation in *Econometrica*. See the competitors section.
- **kaji2023metropolis**: Metropolis-Hastings in which a trained classifier's output replaces the
  intractable likelihood ratio. A second discriminator-based inference method from the same
  line of work; cite with kaji2023adversarial when saying discriminators are accepted as
  inferential objects.
- **bernton2019approximate**: ABC with the Wasserstein distance between observed and simulated
  data (Robert and coauthors), a minimum-distance likelihood-free method.
- **cherief2022finite**: finite-sample guarantees for minimum-MMD estimation that are
  robust to misspecification and allow dependent data.

### 6b. Specification tests for financial and econometric models

The econometric precedent a CREST referee will look for first.

- **aitsahalia1996testing**: tests parametric diffusion models of the short rate by comparing the
  model-implied marginal density with a kernel density estimate.
- **corradi2005bootstrap**: bootstrap specification tests for diffusions. In the multifactor
  case (including stochastic volatility, with no closed-form likelihood) the test compares the
  empirical CDF of the actual data with the empirical CDF of data *simulated from the fitted
  model*. The limit depends on serial dependence and on parameter-estimation error (PEE), and an
  empirical-process block bootstrap that accounts for PEE gives the critical values.
- **bhardwaj2008simulation**: the conditional-distribution version (a simulated conditional
  Kolmogorov test), with block-bootstrap critical values; more power than marginal tests.
- **hong2005nonparametric**: transition-density (PIT-based) kernel specification test for
  continuous-time and discrete-time dynamic models (GARCH, SV, jump-diffusions), robust to serial
  dependence. Needs the model's transition density, which our likelihood-free setting lacks.
- **corradi2011predictive**: simulation-based predictive densities for one-factor and SV
  diffusions, and Diebold-Mariano / White-type tests comparing the accuracy of *several possibly
  misspecified* models. The econometric counterpart of our relative test. It builds on the
  nonparametric simulated ML of fermanian2004nonparametric.
- **bierens1990consistent**: a conditional moment test made consistent against all alternatives
  by weighting with `exp(t' Phi(x))` for a fixed class of `t`.
- **stinchcombe1998consistent**: consistent specification tests over a class of test functions
  indexed by a nuisance parameter present only under the alternative (logistic and other
  "comprehensively revealing" functions, neural hidden units included), calibrated
  asymptotically.
- **lee1993testing**: neural-network tests for neglected nonlinearity (hidden units with random
  weights as test functions), compared with alternative tests. An early neural test function in
  econometrics.

## 7. Bootstrap for dependent data

- **kunsch1989jackknife**: moving-block bootstrap.
- **politis1994stationary**: stationary bootstrap, with geometric block lengths (our
  `StationaryBootstrap` generator and `MVStationaryBootstrap`).
- **politis2004automatic** with **patton2009correction**: automatic block-length rule. Always cite
  the correction together with the original.
- **lahiri2003resampling**: monograph on resampling for dependent data.

## 8. Portfolio choice and estimation risk

- **markowitz1952portfolio**: mean-variance portfolio selection.
- **michaud1989markowitz**: mean-variance optimisers act as "estimation-error maximisers".
- **chopra1993effect**: errors in means matter roughly an order of magnitude more than errors in
  covariances. Explains why GMV is the cleanest TSTR target.
- **ledoit2004well**: linear shrinkage covariance estimator (our `ledoit_wolf`).
- **demiguel2009optimal**: 1/N is hard to beat out of sample (our `equal_weights` benchmark).
- **kan2007optimal**: analytic out-of-sample utility loss from parameter uncertainty. Gives the
  theoretical yardstick for "utility loss".
- **jagannathan2003risk**: long-only constraints act as shrinkage. Justifies reporting long-only
  mean-variance next to unconstrained GMV.
- **ledoit2008robust**: HAC/bootstrap test for the difference of two Sharpe ratios. Use it for
  synthetic-trained against real-trained portfolios in E4.
- **cetingoz2026synthetic**: synthetic data for Markowitz and minimum-variance portfolios. Their
  sample-size result sets expectations for E4. Moments estimated from B replicas of a generator
  fitted on TRAIN are a randomised function of TRAIN, so they carry no more information than
  TRAIN; Cetingoz and Lehalle make this limit precise and show that very large B can add bias.
  Synthetic portfolios should therefore not be expected to beat portfolios estimated on real
  TRAIN, except where a parsimonious generator acts as a structured, shrinkage-like estimator.
  Frame E4's utility loss accordingly. Our question is different: whether a distributional
  verdict *predicts* the utility gap across universes. They do not test that.
- **brown1992survivorship**, **shumway1997delisting**: survivorship and delisting bias. Cite in
  the data-limitation sentence (current S&P 500 constituents only).

---

## Closest competitors and our contribution

Our method: a WGAN-GP critic trained alone (no generator) on a training split of real versus
synthetic windows; statistic `T = mean f(X_ev) - mean f(Y_ev)` on the held-out split; permutation
p-value with f fixed; HAC-studentised variant over time-ordered eval windows; a *relative* version
`Delta = D(G, R_new) - D(R_past, R_new)` with paired bootstrap over the shared R_new eval windows;
validation against the Neyman-Pearson LR oracle for Markov models; and a downstream Markowitz
TSTR study.

**What is new, stated precisely.** None of these is new on its own:
- a witness learned on one split and tested on the other (Kübler et al. 2022);
- a Wasserstein test with split-then-permute calibration (Wang, Gao and Xie 2022);
- a WGAN critic used to validate a financial simulator against real data (Bai et al. 2026);
- a formal two-sample test used to evaluate a market generator (Bühler et al. 2020);
- a specification test that compares real data with data simulated from the fitted financial
  model, with dependence handled by a block bootstrap (Corradi and Swanson 2005).

The contribution is the combination, with its validation:
- (i) level and power of a WGAN-GP critic test, checked against ground truth and against the
  Neyman-Pearson LR oracle where the likelihood is tractable, including under serial dependence
  (permutation against HAC calibration);
- (ii) a relative null whose benchmark is the real past, for non-stationary markets;
- (iii) agreement of critic verdicts with likelihood-based evaluation (predictive log-likelihood,
  Berkowitz, Vuong) on S&P 500 data;
- (iv) a test of whether the verdict predicts Markowitz utility.

Never write "the first formal test of synthetic financial data" or "the first use of a critic for
validation".

1. **Kübler et al. (2022) witness test; Kirchler et al. (2020); Cheng and Cloninger (2022).** The
   generic construction (learn a witness on one split, test the mean difference on the other,
   permutation null) is **not** ours. Kübler et al. state it in general, Kirchler et al. use mean
   tests on learned features, and Cheng and Cloninger use exactly the held-out logit difference
   (our `lrcritic`). Our critic test is an instance of this template whose witness is trained with
   the WGAN-GP objective (an IPM/W1 dual objective) instead of a power proxy or cross-entropy.
   Claim the instance and its properties, not the template.
2. **Lopez-Paz and Oquab (2017), C2ST.** Their statistic is 0-1 accuracy with a binomial/normal
   null, developed for i.i.d. data. We (i) replace accuracy by a real-valued witness mean
   difference, which keeps magnitude; Kim et al. (2021) quantify the power cost of accuracy.
   (ii) We handle dependent financial windows with group (ticker) splits and HAC studentisation.
   (iii) We add the relative test and LR-oracle benchmarks. C2ST stays a baseline.
3. **Liu et al. (2020), MMD-D.** Deep kernels trained to maximise a test-power criterion. Our
   critic is trained with the WGAN objective, not for power, and gives a one-dimensional witness
   that can be read as a learned moment (the "W1 lower bound" reading). We do **not** show it is more
   powerful than MMD-D. MMD-D is not among our baselines, which a referee may flag (see
   limitations).
4. **Wasserstein two-sample tests: Wang, Gao and Xie (2021, 2022); Hu and Lin (2026); Ramdas et
   al. (2017).** Wang et al. (AISTATS 2022) already train a (kernel) projector on a training split
   to maximise the Wasserstein distance between the projected samples, then run a permutation
   test on the held-out split with the projector fixed. **The split-then-permute structure of the
   Wasserstein test is not ours.** Differences: their statistic is the empirical W1 between
   projected samples (kernel projector in 2022, linear in 2021); ours is the held-out mean
   difference of a neural critic trained with a gradient penalty. That is a one-dimensional
   witness with no optimal-transport computation on the eval split, and it has an HAC version
   for dependent windows. Hu and Lin (2026) are the closest in the *witness* (a deep,
   approximately 1-Lipschitz network), but they use learned projections, max-type aggregation and
   asymptotically pivotal calibration for i.i.d. high-dimensional data. None of them has
   time-series windows, a relative null or likelihood benchmarks.
5. **Bai, Balch, Chen, Dervovic, Lam and Vyetrenko (2026), WGAN critic for simulator
   calibration.** They build *eligibility sets* (parameter values not rejected) for
   non-identifiable simulators, including the ABIDES limit-order-book market simulator. Real and
   simulated outputs pass through a feature extractor, one option being the output (or last
   hidden layer) of a trained WGAN critic. The features are compared by Bonferroni-corrected KS
   statistics or mean differences against computable critical values. Their guarantees are
   asymptotic, for i.i.d. outputs, with simulation size growing faster than the real sample. So
   a WGAN critic has already been used, with frequentist guarantees, to validate a financial
   simulator against real data. Differences:
   - Goal: they calibrate (invert many tests to get a parameter set); we give a verdict on one
     fitted generator.
   - Statistic: their critic is a fixed, pre-trained feature map aggregated by KS or
     mean-difference statistics, and they report its output as the most conservative feature. Ours
     is trained on a separate split of the two samples being compared, and its held-out mean
     difference *is* the statistic, calibrated by permutation (exact under exchangeability) or HAC.
   - We add serial-dependence handling, a relative null and the LR-oracle power benchmark.
   Present ours as different from theirs, not as the first use of a critic for validation.
6. **Bühler, Horvath, Lyons, Perez Arribas and Wood (2020); Chevyrev and Oberhauser (2022),
   signature MMD as a test for market generators.** Bühler et al. argue for two-sample tests as
   evaluation metrics. They evaluate their S&P 500 market generator with the Chevyrev-Oberhauser
   signature-MMD test (distribution-free, hence conservative, threshold) and marginal KS tests,
   without a size or power analysis. **Formal two-sample testing of market generators is not new.**
   We use a truncated signature-feature MMD only as a *baseline*. We do not claim
   characteristicness, and our contribution is not a new signature test. What we add is the
   validation they lack: size under H0 (placebos, ground truth) and power against
   finance-relevant deviations, relative to the LR oracle. A "pass" of a conservative test is weak
   evidence without a power check, which is the point of our decision rule.
7. **Relative tests: Bounliphone et al. (2016); Jitkrittum et al. (2018); Gerber and Polyanskiy
   (2024); Corradi and Swanson (2011).** Relative similarity and likelihood-free tests compare
   *two models* against one reference. Corradi and Swanson (2011) is the econometric counterpart:
   they compare the predictive densities of several possibly misspecified diffusion models, built
   by simulation, with Diebold-Mariano / White-type tests. New in ours:
   - (i) one of the two "models" is the **real past** R_past, so H0
     `d(G, R_new) <= d(R_past, R_new)` asks "is G at least as good a proxy for the future as
     history is?" under non-stationarity;
   - (ii) the distance is a learned critic IPM with paired bootstrap over shared R_new eval
     windows, not MMD with joint asymptotic normality or a predictive-density loss.

   **Caveats.**
   - The two critics are trained separately, so Delta compares IPMs over two *data-dependent*
     classes. Use identical architectures and training budgets for both and say so.
   - Bounliphone et al.'s asymptotics assume independent i.i.d. samples and non-degenerate MMDs
     (both candidates differ from the reference). Our windows are dependent, and the
     configuration with G, R_past and R_new all equal in law is degenerate. Report it separately,
     as conservative, or exclude it (see Section 2).
8. **Simulation-based specification tests of financial models: Aït-Sahalia (1996); Corradi and
   Swanson (2005); Bhardwaj, Corradi and Swanson (2008); Hong and Li (2005).** Testing a financial
   model by comparing the data with simulations from the fitted model is an established
   econometric idea. It covers stochastic-volatility models with no closed-form likelihood, which
   is our Heston case. Corradi and Swanson compare the empirical CDFs of actual and simulated
   data; Bhardwaj et al. compare conditional distributions. Both get critical values from a block
   bootstrap that accounts for serial dependence and parameter-estimation error (PEE).
   Differences:
   - (a) *Null.* They test the *model class*: the data come from G_theta at the pseudo-true theta,
     estimated by theta-hat, so estimation error enters the limit and must be corrected. We test
     the *fitted generator*: `H0: Law(G_theta-hat) = Law(R_new)`, with theta-hat fitted on TRAIN
     and held fixed, evaluated out of period on TEST.
   - (b) *Statistic.* Theirs is a fixed CDF (Kolmogorov-type) functional; ours is a critic
     learned on a training split.
   - (c) *Calibration.* Theirs is a block bootstrap with a PEE correction; ours is permutation or
     HAC, conditional on theta-hat.
   - (d) We add a relative null. Their multiple-model predictive-density comparison (Corradi and
     Swanson 2011) is the counterpart of our relative test (item 7).
9. **Kaji, Manresa and Pouliot (2023), adversarial estimation.** (Read in full, arXiv v3, Nov
   2022.) They *estimate* structural parameters by `min_theta max_D` of the cross-entropy
   classification objective. A logistic discriminator gives optimally weighted SMM, the oracle
   discriminator gives MLE (efficient when simulation size m >> n), and a sieve or neural
   discriminator can attain efficiency. Under misspecification the target is the
   Jensen-Shannon projection. Inference on theta is by bootstrap. Data are i.i.d., and Section 3.3
   says explicitly that "the theoretical results in this paper do not cover time series data".
   They mention the Wasserstein loss only as an alternative (footnote 1, Section 3.2). They do
   **not** propose a specification test based on the discriminator value. New in ours:
   - we *test* a calibrated generator rather than estimate it;
   - we use the WGAN-GP critic (IPM) rather than cross-entropy (JS);
   - we work on dependent time-series windows;
   - we have a relative null and a downstream utility study.

   **Parameter estimation, stated exactly.** Calibration (TRAIN) and evaluation (TEST) are
   separated in time, and we condition on theta-hat. For the null
   `H0: Law(G_theta-hat) = Law(R_new)` about the *fitted generator*, no PEE correction is needed:
   theta-hat is fixed, the synthetic sample is fresh simulation, and the statistic uses only TEST
   windows. This is **not** a test of correct specification of the model class (the null of
   Corradi and Swanson, and the natural null in an estimation setting). For that null, estimation
   error in theta-hat matters, and our test would over-reject. A rejection therefore does not
   separate misspecification from estimation error or from parameter drift between periods; the
   relative test is designed to absorb drift. Conditioning on TRAIN is exact if R_new is
   independent of TRAIN. With serial dependence (volatility persistence across 2022-12-30) it
   holds only approximately, through mixing.

   The shared insight to cite: their logistic-to-SMM and oracle-to-MLE bridge is the estimation
   counterpart of our C2ST-logit and LR-oracle comparison (RQ4).
10. **Uncalibrated discriminator scores: TimeGAN discriminative score (Yoon et al., 2019); ETS
    challenges (Franco-Pedroso et al., 2018); MAS-GAN (Storchan, Vyetrenko and Balch, 2021).**
    - TimeGAN reports a post-hoc LSTM classifier's held-out error.
    - The ETS challenges pose the realism of simulated financial series as a real-versus-simulated
      classification competition.
    - Storchan et al. use a real-versus-fake discriminator's output on price and volume series as
      the objective to calibrate a multi-agent market simulator.

    All three use classifier accuracy or discriminator output with no null distribution, no
    p-value and no level or power analysis. They do not account for serial dependence or for
    non-stationarity between training and evaluation periods. Our C2ST is the calibrated version
    of these scores, and our critic test the IPM version. We can state that a "small"
    discriminative score is uninterpretable without the power check that our decision rule
    requires.
11. **Train on synthetic, test on real (Esteban et al., 2017; TimeGAN predictive score);
    Cetingoz and Lehalle (2026).** TSTR *is* our RQ5 utility measure (Markowitz portfolios built
    on synthetic moments, evaluated on real TEST). Cetingoz and Lehalle study exactly this use.
    They prove that the initial sample size limits what synthetic data can add to Markowitz and
    minimum-variance portfolios, warn against generating far more data than was available, compare
    a generator with a block bootstrap and argue for use-targeted evaluation. Their result implies
    that E4's synthetic portfolios should not be expected to beat portfolios estimated on real
    TRAIN. The new question in ours is whether a distributional test verdict *predicts* TSTR
    utility across universes (Spearman with universe-level bootstrap), rather than using TSTR as
    the metric itself. They do not ask it.

**Overclaim guard (for the theory section).**
- *Exactness.* The permutation p-value is exact conditional on the training half when the pooled
  eval windows are exchangeable under H0 (lehmann2022testing, hemerik2018exact). Equality of the
  marginal window law is not enough when real eval windows are serially dependent (volatility
  clustering across adjacent non-overlapping windows) and synthetic windows come from independent
  paths. The `critic.py` docstring reports 15-17% rejections at nominal 5% for GARCH(1,1)-t
  windows with a+b = 0.98. State exactness under exchangeability (i.i.d. windows, e.g. in E2 with
  independent paths) and use the HAC-studentised statistic for real data. Justify `z_hac` by a CLT
  for mixing sequences plus a consistent HAC variance (herrndorf1984functional,
  carrasco2002mixing; newey1987simple, andrews1991heteroskedasticity, driscoll1998consistent).
  Do not justify it with chung2013exact, whose result is for independent samples.
- *Independence of the critic from the eval windows.* With a random or ticker-grouped split,
  eval windows are interleaved in time with, or contemporaneous to, training windows. Under serial
  or cross-sectional dependence, f is then not exactly independent of the eval windows, and every
  "conditional on the training half" argument (permutation or HAC) is approximate. A time-blocked
  split with a gap makes the dependence vanish under mixing.
- *Parameter estimation.* "No PEE correction" holds for the null about the fitted generator only
  (competitors item 9), not for correct specification of the model class (item 8).
- *Relative test, degenerate case.* The configuration with G, R_past and R_new all equal in law
  violates the non-degeneracy that Bounliphone et al. assume. Report it as conservative or exclude
  it.
- *Novelty.* Do not claim the first formal test of market generators (buehler2020data), the first
  critic-based validation of a financial simulator (bai2026calibrating), the first split-then-permute
  Wasserstein test (wang2022two), or the first simulation-based specification test
  (corradi2005bootstrap).
- *"W1 lower bound".* GP is a soft constraint, so T is an IPM over the trained-critic class (a
  lower bound on `L * W1` for the critic's true Lipschitz constant L). It is not an estimate of W1
  (arora2017generalization, mallasto2019well).
- *Heston "intractable".* Write "no closed-form likelihood" (aitsahalia2007maximum).

## Positioning for a CREST audience

For a CREST / ENSAE-Polytechnique readership, present the critic test as a **specification test
for simulation-based models**. Two econometric traditions meet here:
- *Simulation-based estimation:* indirect inference (gourieroux1993indirect,
  smith1993estimating), simulated moments for Markov asset-price models (duffie1993simulated) and
  simulation-based econometrics (gourieroux1997simulation; fermanian2004nonparametric).
- *Specification testing of financial models:* density- and transition-density tests of
  diffusions (aitsahalia1996testing, hong2005nonparametric). The closest are the simulation-based
  tests of Corradi and Swanson (corradi2005bootstrap, bhardwaj2008simulation). They compare
  actual data with data simulated from the fitted model and use a block bootstrap for dependence
  and estimation error. Say early, and exactly, how our null differs (fitted generator, out of
  period, competitors item 8).

In that language the critic is a *data-chosen test function*. EMM picks the auxiliary score as
the moment (gallant1996which). Consistent specification tests use a whole class of weight
functions to be consistent against all alternatives: exponential weights (bierens1990consistent),
or classes indexed by a nuisance parameter present only under the alternative, neural hidden units
included (stinchcombe1998consistent; neural test functions in lee1993testing). The critic test is
a **learned, sample-split relative of the Bierens and Stinchcombe-White tests**:
- *Function class.* Theirs is fixed, and the statistic takes a sup or an integral over it. Ours
  picks one test function within a Lipschitz neural class on a training split.
- *Calibration.* Theirs is asymptotic. Ours is permutation (exact under exchangeability) or HAC
  on the held-out split.
- *Null.* They test conditional moment restrictions; we test equality of two laws of windows.
  The analogy is in the test-function class.

It follows that the absolute test inherits the power of the learned moment, not the consistency
of Bierens-type tests. Do not claim consistency against all alternatives without an argument that
training finds a discriminating f.

Kaji, Manresa and Pouliot (2023) (and kaji2023metropolis) show this audience already accepts
discriminators as econometric objects. Our paper is the *testing* counterpart for dependent
financial data. Pitch the relative test as a Vuong / Diebold-Mariano / Giacomini-White style
comparison (vuong1989likelihood, diebold1995comparing, giacomini2006tests), with Corradi and
Swanson's comparison of several misspecified simulated models (corradi2011predictive) as the
closest econometric relative. Its benchmark is the forecasting *method* "use the real past",
which is the honest bar under non-stationarity. The likelihood benchmarks (diebold1998evaluating,
berkowitz2001testing, amisano2007comparing) let a referee check the learned test against tools
they trust. Minimum-distance results with Wasserstein or MMD under misspecification and
dependence (bernton2019approximate, cherief2022finite) are the natural bridge to the estimation
side. Keep the machine-learning vocabulary minimal. Define "critic" once as "a neural-network test
function chosen to maximise a mean difference under a Lipschitz penalty".

## Citation map

| Claim we make | Keys to cite |
|---|---|
| Synthetic financial data is used for back-testing, data sharing, hedging | assefa2020generating, kondratyev2019market, buehler2019deep, bailey2017probability |
| GAN-type market generators exist and are mostly evaluated informally (stylized facts, scores) | wiese2020quant, takahashi2019modeling, ni2021sig, liao2024sig, yoon2019time, cont2026tail, stenger2024evaluation, vyetrenko2020get |
| Formal two-sample tests already used to evaluate market generators (no size or power analysis) | buehler2020data, chevyrev2022signature |
| Discriminator or critic outputs already used to judge or calibrate financial simulators (no calibrated null) | bai2026calibrating, storchan2021learning, francopedroso2018ets, yoon2019time |
| Evaluation metrics for generative models disagree and must match the use | theis2016note, alaa2022faithful |
| Stylized facts of returns | cont2001empirical, mandelbrot1963variation, fama1965behavior |
| GAN discriminator optimum is the density ratio; logit = log-LR | goodfellow2014generative, nguyen2010estimating, cranmer2015approximating |
| WGAN critic, Kantorovich-Rubinstein duality | arjovsky2017wasserstein, villani2009optimal |
| Gradient penalty and its hyper-parameters (betas (0.5, 0.9) are their Appendix E setting, not their default) | gulrajani2017improved |
| Our statistic is an empirical IPM | muller1997integral, sriperumbudur2012empirical |
| Finite critics measure a weaker "neural net distance", not W1 | arora2017generalization, mallasto2019well |
| Unrestricted empirical W1 suffers from the curse of dimensionality | fournier2015rate |
| Wasserstein-based two-sample tests (prior) | ramdas2017wasserstein, wang2021two, wang2022two, hu2026neural |
| Split-then-permute Wasserstein test (learned projector, held-out permutation) | wang2022two |
| Train-split / held-out witness test template | kubler2022witness, kirchler2020two, lopezpaz2017revisiting |
| Learned-logit (LR critic) two-sample statistic | cheng2022classification |
| Accuracy-based tests lose power | kim2021classification |
| Deep-kernel MMD tests | liu2020learning |
| MMD test and median heuristic | gretton2012kernel |
| Signature features and signature MMD | lyons1998differential, chevyrev2016primer, kiraly2019kernels, chevyrev2022signature |
| Signature scores used to train financial generators | ni2021sig, liao2024sig, issa2023nonadversarial |
| Permutation tests exact under exchangeability; random permutations; +1 rule | lehmann2022testing, hemerik2018exact, phipson2010permutation |
| Studentised permutation test of a parameter (independent samples, unequal laws) | chung2013exact, janssen1997studentized |
| Studentised permutation under time-series dependence (autocorrelation tests; related only) | romano2022permutation |
| HAC variance for time-ordered windows (Bartlett kernel; cross-sectional sums by date) | newey1987simple, driscoll1998consistent |
| HAC lag: Andrews AR(1) plug-in (critic.py default) or Newey-West rule of thumb (`hac_lags="nw"`, Vuong) | andrews1991heteroskedasticity, newey1994automatic |
| Normal approximation for `z_hac`: CLT under mixing; GARCH/SV are beta-mixing; HAC consistency | herrndorf1984functional, carrasco2002mixing, andrews1991heteroskedasticity |
| Centred bootstrap p-value (relative test) | hall1991two |
| Kernel tests under temporal dependence | chwialkowski2014wild |
| Relative similarity / model comparison tests | bounliphone2016test, jitkrittum2018informative, gerber2024likelihood |
| Comparing several possibly misspecified simulated models (econometrics) | corradi2011predictive |
| Relative test as a loss-differential comparison | diebold1995comparing, giacomini2006tests, vuong1989likelihood |
| BH correction for per-ticker tests (and dependence caveat) | benjamini1995controlling, benjamini2001control |
| Neyman-Pearson LR oracle | lehmann2022testing |
| GBM, Merton, GARCH-t, Heston, CCC (DCC) models | black1973pricing, merton1976option, engle1982autoregressive, bollerslev1986generalized, bollerslev1987conditionally, heston1993closed, bollerslev1990modelling, engle2002dynamic |
| Heston full-truncation Euler scheme | lord2010comparison |
| CCC-GARCH-t two-stage (IFM) estimation | joe2005asymptotic |
| Multivariate t fitted by ECME | liu1995ml |
| GARCH(1,1) is a strong benchmark | hansen2005forecast |
| Heston likelihood not available in closed form (approximations exist) | aitsahalia2007maximum, fermanian2004nonparametric |
| Hill tail index | hill1975simple |
| Heston calibrated by simulated moments | mcfadden1989method, duffie1993simulated, gourieroux1997simulation |
| Indirect inference / EMM / simulation-based estimation | gourieroux1993indirect, smith1993estimating, gallant1996which, gourieroux1997simulation |
| Discriminator-based estimation and inference in econometrics | kaji2023adversarial, kaji2023metropolis, athey2024using |
| Simulation-based specification tests of financial models (dependence, PEE, block bootstrap) | corradi2005bootstrap, bhardwaj2008simulation, aitsahalia1996testing, hong2005nonparametric |
| Consistent specification tests over a class of test functions (neural included) | bierens1990consistent, stinchcombe1998consistent, lee1993testing |
| Likelihood-free / simulation-based inference with classifiers | gutmann2018likelihood, cranmer2020frontier, cranmer2015approximating |
| Minimum-distance (Wasserstein, MMD) estimation under misspecification | bernton2019approximate, cherief2022finite |
| PIT and density-forecast tests | diebold1998evaluating, berkowitz2001testing |
| Predictive log-likelihood as a proper score; weighted LR comparisons | gneiting2007strictly, amisano2007comparing |
| Non-nested model comparison | vuong1989likelihood |
| Stationary bootstrap generator; block-length rule | politis1994stationary, politis2004automatic, patton2009correction, kunsch1989jackknife, lahiri2003resampling |
| Mean-variance portfolios | markowitz1952portfolio |
| Estimation error dominates portfolio performance | michaud1989markowitz, chopra1993effect, kan2007optimal |
| Ledoit-Wolf shrinkage | ledoit2004well |
| 1/N benchmark | demiguel2009optimal |
| Long-only constraints as regularisation | jagannathan2003risk |
| Sharpe-ratio difference test | ledoit2008robust |
| TSTR | esteban2017real, yoon2019time |
| Synthetic data for portfolio construction; limits set by the initial sample size | cetingoz2026synthetic |
| Synthetic data judged by a downstream decision (econometrics) | athey2024using |
| Survivorship bias of current-constituent samples | brown1992survivorship, shumway1997delisting |

---

## Verified, held in reserve (not in `references.bib`)

Each was verified at the same standard but left out to keep the file focused. Add them only if
needed.

| Suggested key | Reference | Record |
|---|---|---|
| kubler2022automl | Kübler, Stimper, Buchholz, Muandet, Schölkopf, "AutoML Two-Sample Test", NeurIPS 35 (2022) | proceedings.neurips.cc; arXiv 2206.08843 |
| kubler2020learning | Kübler, Jitkrittum, Schölkopf, Muandet, "Learning Kernel Tests Without Data Splitting", NeurIPS 33 (2020), 6245-6255 | proceedings.neurips.cc |
| gerber2023kernel | Gerber, Jiang, Polyanskiy, Sun, "Kernel-Based Tests for Likelihood-Free Hypothesis Testing", NeurIPS 36 (2023), 15680-15715 | doi:10.52202/075280-0690 |
| sutherland2017generative | Sutherland et al., "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy", ICLR 2017 | iclr.cc 2017 list; arXiv 1611.04488 |
| schrab2023mmd | Schrab, Kim, Albert, Laurent, Guedj, Gretton, "MMD Aggregated Two-Sample Test", JMLR 24(194):1-81 (2023) | jmlr.org |
| jitkrittum2016interpretable | Jitkrittum, Szabó, Chwialkowski, Gretton, "Interpretable Distribution Features with Maximum Testing Power", NeurIPS 29 (2016) | proceedings.neurips.cc |
| nowozin2016fgan | Nowozin, Cseke, Tomioka, "f-GAN", NeurIPS 29 (2016) | proceedings.neurips.cc |
| heusel2017gans | Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID), NeurIPS 30 (2017) | proceedings.neurips.cc |
| sajjadi2018assessing | Sajjadi et al., "Assessing Generative Models via Precision and Recall", NeurIPS 31 (2018) | proceedings.neurips.cc |
| stanczuk2021wasserstein | Stanczuk, Etmann, Kreusser, Schönlieb, "Wasserstein GANs Work Because They Fail (to Approximate the Wasserstein Distance)" | arXiv 2103.01678 |
| mohamed2016learning | Mohamed, Lakshminarayanan, "Learning in Implicit Generative Models" | arXiv 1610.03483 |
| kidger2021neural | Kidger, Foster, Li, Lyons, "Neural SDEs as Infinite-Dimensional GANs", ICML, PMLR 139 (2021), 5453-5463 | proceedings.mlr.press |
| hambly2010uniqueness | Hambly, Lyons, "Uniqueness for the signature of a path of bounded variation and the reduced path group", Ann. Math. 171(1):109-167 (2010) | doi:10.4007/annals.2010.171.109 |
| koshiyama2021generative | Koshiyama, Firoozye, Treleaven, "Generative adversarial networks for financial trading strategies fine-tuning and combination", Quant. Finance 21(5):797-813 (2021) | doi:10.1080/14697688.2020.1790635 |
| lezmi2020improving | Lezmi, Roche, Roncalli, Xu, "Improving the Robustness of Trading Strategy Backtesting with Boltzmann Machines and GANs" | arXiv 2007.04838 |
| jordon2022synthetic | Jordon et al., "Synthetic Data -- what, why and how?" | arXiv 2205.03257 |
| hansen1982large | Hansen, "Large Sample Properties of GMM Estimators", Econometrica 50(4):1029-1054 (1982) | doi:10.2307/1912775 |
| thomas2022likelihood | Thomas, Dutta, Corander, Kaski, Gutmann, "Likelihood-Free Inference by Ratio Estimation", Bayesian Anal. 17(1):1-31 (2022) | doi:10.1214/20-BA1238 |
| bernton2019parameter | Bernton, Jacob, Gerber, Robert, "On parameter estimation with the Wasserstein distance", Inf. Inference 8(4):657-676 (2019) | doi:10.1093/imaiai/iaz003 |
| kim1998stochastic | Kim, Shephard, Chib, "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models", Rev. Econ. Stud. 65(3):361-393 (1998) | doi:10.1111/1467-937X.00050 (Crossref misspells "Shephard") |
| white2000reality | White, "A Reality Check for Data Snooping", Econometrica 68(5):1097-1126 (2000) | doi:10.1111/1468-0262.00152 |
| wang2026zeroflow | Wang, Wang, Liu, Suzuki, "Zero-Flow Two-Sample Tests" | arXiv 2607.21542 |
| zhou2025anchor | Zhou, Peng, Tian, Liu, "Anchor-based Maximum Discrepancy for Relative Similarity Testing" | arXiv 2510.10477 |

## Unverified (excluded)

No candidate named in the task failed verification. These references came up as possibly useful
but were **not checked** (no authoritative record was looked up), so they are not in the `.bib`.
Verify before citing.

- Black (1976), leverage effect ("Studies of stock price volatility changes", ASA proceedings): no
  DOI record looked up.
- Friedman (2003), "On multivariate goodness-of-fit and two-sample testing" (technical report):
  not looked up.
- Lyons, Caruana and Lévy (2007), Saint-Flour lecture notes on rough paths: not looked up.
- Glosten, Jagannathan and Runkle (1993), GJR-GARCH: not looked up (no asymmetric model in the
  paper).
- Hansen, Lunde and Nason (2011), model confidence set: not looked up.
- López de Prado (2018), *Advances in Financial Machine Learning*: not looked up.

## Verification notes (corrections to the candidate list)

- Amisano and Giacomini (2007): DOI is `10.1198/073500106000000332`. The DOI first tried,
  `...0368`, does not resolve.
- Kaji, Manresa and Pouliot (2023): DOI is `10.3982/ECTA18707`, Econometrica 91(6):2041-2063.
- Cheng and Cloninger (2022): the published title is "Classification Logit Two-Sample Testing by
  Neural Networks for Differentiating Near Manifold Densities" (IEEE Trans. Inf. Theory
  68(10):6631-6662).
- Ni et al. (2021) is the ICAIF proceedings paper "Sig-Wasserstein GANs for time series
  generation". The conditional SigWGAN (arXiv 2006.05421) appeared as Liao et al. (2024),
  *Mathematical Finance*. Both are included.
- Hemerik and Goeman: online 2017, print 2018, TEST 27(4). Cited as 2018.
- DeMiguel, Garlappi and Uppal: online 2007, print 2009, RFS 22(5). Cited as 2009.
- Fournier and Guillin: online 2014, print 2015. Cited as 2015.
- Gutmann et al.: online 2017, print 2018. Cited as 2018.
- Gouriéroux and Monfort, *Simulation-based Econometric Methods*: Crossref print date 9 Jan 1997
  (often cited as 1996). Cited as 1997.
- Kondratyev and Schwarz (2019): Crossref lists the first author as "Oleksiy". SSRN and *Risk*
  use "Alexei", which is the form used here.
- Ledoit and Wolf (2008): Crossref misspells "Oliver". Corrected to Olivier.
- Lehmann and Romano: the 4th edition (2022, Springer, two volumes) is cited.
- Chevyrev and Kormilitzin: the arXiv 2016 primer is cited, with a note pointing to the revised
  Springer chapter (doi:10.1007/978-3-031-97239-3_1, pp. 3-64).
- Cont, Cucuringu, Xu and Zhang, Tail-GAN: Management Science 72(4):2917-2936, print April 2026.

### Added after the adversarial review (2026-11-14)

- Bai et al.: online 27 Oct 2025, print January 2026 (ACM TOMACS 36(1)). Cited as 2026, key
  **bai2026calibrating** (the reviewer suggested `bai2025calibrating`; the print-year convention
  wins). The arXiv v1 title is "Calibrating Over-Parametrized Simulation Models: A Framework via
  Eligibility Set"; the published title differs (recorded in the entry's `note`). Section and page
  references in this file are to arXiv v1: Section 4, pp. 23-28 (features, SKS/SSMD/ESMD,
  critical values, Theorem 7); Section 5, p. 40 (critic output "the most conservative").
- Franco-Pedroso et al.: key **francopedroso2018ets**, not `franco2018ets`, following the
  `lopezpaz2017revisiting` convention for hyphenated surnames. arXiv only, no journal version found.
- Storchan, Vyetrenko and Balch: arXiv only (no `journal_ref`), cited as a preprint.
- Vyetrenko et al., "Get real": Crossref stores the title as "Get real" with the rest in the
  `subtitle` field. Full title restored from the subtitle and from arXiv 1912.04941.
- Cetingoz and Lehalle: online 7 Jul 2026, print Aug 2026 (QF 26(8)). arXiv 2501.03993v6.
- Lord, Koekkoek and van Dijk: online 2009, print 2010. Crossref lists "Dijk, Dick Van",
  corrected to "van Dijk, Dick".
- Hong and Li: online 2004, print 2005 (moved from the reserve table into the `.bib`).
- Kaji and Ročková: online 2022, print 2023 (JASA 118(544)).
- Romano and Tirlea: online Jan 2022, print Sep 2022 (JTSA 43(5)).
- Hall and Wilson (1991) and Bierens (1990): Crossref gives only the first page. Last pages
  completed from Semantic Scholar (757-762) and RePEc/IDEAS (1443-1458).
- Newey and West (1994): Crossref gives initials only; full given names used.
- Liu and Rubin (1995): no DOI; checked on the *Statistica Sinica* index page (5(1), 19-39).
- Wang, Gao and Xie (2022): PMLR page `proceedings.mlr.press/v151/wang22f.html` (moved from the
  reserve table into the `.bib`). Companion ISIT 2021 paper verified on Crossref.
- Gulrajani et al. (2017): recommended Adam betas are (0, 0.9) (Algorithm 1, and Section 5 with
  "our recommended settings"); (0.5, 0.9) appears only in Appendix E (LSUN robustness).
- Bühler et al. (2020): the earlier description ("evaluated with signature-based distances") was
  wrong. They use the signature-MMD *two-sample test* and marginal KS tests (Section 4.1.1).
