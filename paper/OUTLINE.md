# Paper outline, claims and label registry

*Is Synthetic Financial Data Good Enough? A Validated Critic Two-Sample Test with Likelihood
Benchmarks and Markowitz Utility.* Audience: CREST (mathematical statistics, financial
econometrics). English, LaTeX, `paper/main.tex`.

Binding inputs: `docs/DESIGN.md` (§1 concept and decision rule, §3.7 inference protocol, §4
experiments, Change log), `docs/theory.md` (statements and proofs), `docs/literature.md`
(positioning, citation map), `src/synthval` (what the code does), `experiments/*.py` docstrings
(designs and outputs). Notation: `paper/notation.md`.

## 0. Rules for all writers

- **Framing.** The \WC{} test is one well-calibrated, interpretable tool inside a validated
  protocol (exact or asymptotically valid inference, relative null, likelihood link,
  multivariate reach). It is **not** uniformly most powerful. Say plainly where simpler tests
  win (pooled KS against marginal deviations). Never write "first", "state of the art",
  "novel framework". \GOOD{} means "no detectable deviation of the pre-specified size", never
  "equal to the truth".
- **Numbers.** Only from generated tables (`paper/tables/*.tex`) or figures. Missing result:
  write the sentence around `\todo{...}` and reference the table/figure label that will hold it.
  Pilot numbers from theory.md §11 may be quoted only as "a pilot" and only where this outline
  says so.
- **Citations.** Only keys in `paper/references.bib` (check with `grep -n '{key,' references.bib`).
  Missing reference: cite the proposed key of §13 below (form `author_year_word`) and list it in
  your report; do not edit the `.bib`.
- **Statements.** Keep theory.md's status: proved / sketch / heuristic. Heuristics are remarks
  titled "(heuristic)". Do not prove what theory.md does not prove.
- **Tables and figures.** Experiments write bare `tabular`s to `paper/tables/<stem>.tex` and
  PDFs to `paper/figures/<stem>.pdf`. Wrap them in the section file with the templates of §11.
  Labels: `tab:`/`fig:` + file stem with `_` replaced by `-` (e.g. `e2_power_ci.tex` ->
  `tab:e2-power-ci`). Never edit a generated file.
- **Lengths** (11pt, A4, 2.5 cm margins, about 600 words per text page): main text about 35
  pages including tables and figures, appendix 15-25 pages.

## 1. Page budget

| # | File | Section | Label | Pages | Tables / figures in main text |
|---|---|---|---|---|---|
| - | `abstract.tex` | Abstract | - | 0.3 | - |
| 1 | `introduction.tex` | Introduction | `sec:intro` | 2.5 | - |
| 2 | `literature.tex` | Related literature | `sec:lit` | 2 | - |
| 3 | `methodology.tex` | Methodology | `sec:method` | 4 | (optional schematic) |
| 4 | `theory.tex` | Theory | `sec:theory` | 6.5 | - |
| 5 | `simulation.tex` | Simulation evidence | `sec:sim` | 5 | 3 tables, 2 figures |
| 6 | `data.tex` | Data and generators | `sec:data` | 2.5 | 2 tables, 1 figure |
| 7 | `empirical.tex` | Empirical results: S&P 500 | `sec:emp` | 5.5 | 6 tables, 3 figures |
| 8 | `markowitz.tex` | Markowitz utility of synthetic data | `sec:markowitz` | 4 | 4 tables, 1 figure |
| 9 | `robustness.tex` | Robustness | `sec:robust` | 1.5 | 2 tables, 1 figure |
| 10 | `conclusion.tex` | Conclusion | `sec:concl` | 1 | - |
| A-G | `appendix.tex` | Appendices | `app:*` | 15-25 | rest of the tables |

## 2. Abstract (`abstract.tex`, 180-220 words, written last)

Problem; method (critic trained alone, held-out statistic, exact permutation level for
independent windows, time-block + Driscoll-Kraay for dependent panels, anchored relative test,
pre-specified power check, three-valued verdict); theory in one sentence; simulation (valid size,
power on clustering, KS wins on marginal deviations but is invalid under within-window
dependence and blind to dependence-only deviations); S&P 500 verdicts and Markowitz link
(`\todo` until E3/E4 are final).

## 3. Section 1: Introduction (`sec:intro`, 2.5 pages)

Claims (each with its forward reference):
1. Synthetic return data is used for back-testing, data sharing, hedging and portfolio
   construction; its quality is mostly judged by stylized facts, scores without a null
   distribution, or formal tests with unchecked size and power (citation map rows 1-4).
2. A yes/no answer needs level control on dependent data, power against relevant deviations, a
   benchmark that respects non-stationarity, a likelihood link and downstream evidence.
3. Contributions C1-C6 (see the stub): inference protocol (`prop:exact-level`, `prop:hac`,
   `prop:ticker-bias`, `tab:e1-size-sim`); anchored relative test (`prop:anchored`,
   `prop:boundary-size`, `tab:e2-relative-boundary`); likelihood link (`prop:bayes-logit`,
   `prop:oracle-bound`, `prop:local-power`); honest power comparison (`fig:e2-power`,
   `fig:e2dep-power`); S&P 500 verdicts (`tab:e3-verdicts`, `tab:e3-agreement`); Markowitz
   (`sec:mk-theory`, `tab:e4-correlation`, `tab:e4-link`).
4. Limits of the claims (not most powerful; GOOD is conditional; survivorship).
Facts the intro may already state (source in brackets): within-window GARCH dependence makes
pooled KS oversized [`tab:e2-size`, note]; window tests detect clustering at matched variance
while KS dominates marginal deviations [`tab:e2-power`]; the real average correlation fell
between TRAIN and TEST [`tab:e4-correlation`]. The size of the naive protocol: `\todo` with
`tab:e1-size-sim` (theory.md pilot: about 45% at 5%, quote only as a pilot).

## 4. Section 2: Related literature (`sec:lit`, 2 pages)

Subsections `sec:lit-tests`, `sec:lit-dependence`, `sec:lit-synthetic`, `sec:lit-econometrics`,
`sec:lit-portfolio` (content in the stub). Claims: positioning only. State exactly what is not
ours: the held-out witness template (kubler2022witness, kirchler2020two, cheng2022classification),
split-then-permute Wasserstein testing (wang2022two), a WGAN critic to validate a simulator
(bai2026calibrating), formal tests of market generators (buehler2020data), simulation-based
specification tests (corradi2005bootstrap). Ours: the instance and its validation, the
dependent-data protocol, the anchored relative null, the likelihood benchmarks, the Markowitz
study. Present the critic as a learned, sample-split relative of Bierens and Stinchcombe-White
test functions (bierens1990consistent, stinchcombe1998consistent, lee1993testing). MMD-D
(liu2020learning) is not a baseline: say so.

## 5. Section 3: Methodology (`sec:method`, 4 pages)

Must match the code (file in brackets). Subsections and equations:

| Subsection | Label | Content (code) |
|---|---|---|
| Windows, samples, hypotheses | `sec:method-setup` | windows, $s_i$ scaling, equal per-ticker counts, roles of $X,Y,Z$, `def:abs-null` (stated in Section 4, used here) (`data.py`, `pipeline.panel_windows`) |
| The \WC{} statistic | `sec:method-critic` | split 50/50; $A$ from pooled training half; WGAN-GP `eq:wgan-gp` ($\lgp=10$, Adam lr $10^{-3}$, betas $(0.5,0.9)$, batch 128 per sample, 1000 steps, no early stopping); `eq:critic-stat`; MLP $128\times3$ LeakyReLU(0.2) for univariate windows, 1-D CNN (2 conv blocks, 64 channels, kernel 5, LayerNorm, flatten head) for $(N,L)$ windows (`critic.py`, `nets.py`). Architecture: `\todo` until the lead decides on the time-pooled critic pilot |
| Competitors and oracle | `sec:method-competitors` | \LRC{} = C2ST logit (balanced BCE, Adam lr $10^{-3}$, weight decay $10^{-4}$, early stopping); C2ST balanced accuracy, randomised ties; MMD `eq:mmd-u` (median heuristic, $\le3000$ windows); \SigMMD{} (time-augmented cumulative path, depth 4, standardised features); KS on pooled returns; oracle `eq:lr-oracle` (Monte Carlo critical value, 2000 null samples) (`c2st.py`, `mmd.py`, `signature.py`, `ks.py`, `likelihood.lr_oracle_test`) |
| Inference | `sec:method-inference` | `eq:perm-pvalue` ($B=1000$; kernel 500); dependent protocol: $K=10$ time blocks per sample, blocks assigned at random (interleaved, no gap), `eq:dk-var` (date sums, Bartlett, Andrews AR(1) plug-in), `eq:hac-z`; wild bootstrap (AR(1) multipliers shared by date, plug-in block length); KS invalid (`critic.time_block_labels`, `hac_var_mean`, `mmd.mmd2_wild_bootstrap`, `pipeline.protocol_kw`) |
| Relative tests | `sec:method-relative` | anchored `eq:anchored-stat` (R_new only in training; pool weights 1/2-1/2; permutation or HAC), two-critic `eq:two-critic-stat` (paired bootstrap, $B=500$), MMD version (Bounliphone); bootstrap generator INVALID (`relative.py`) |
| Decision rule | `sec:method-decision` | `eq:pmed`; power check D1/D2, vol $\times1.2$, $\bmin=0.8$, boundary size; `eq:verdict` (`pipeline.verdict`, `e3_sp500.py` verdict) |

## 6. Section 4: Theory (`sec:theory`, 6.5 pages + proofs in `app:proofs`)

All statements, labels and sources. "Main" = statement in Section 4; "App" = proof in Appendix A.

| Label | Statement | theory.md source | Status | Where |
|---|---|---|---|---|
| `def:abs-null` | absolute null $P=Q$ on window laws | Def 1.1 | - | main |
| `prop:window-ident` | windows identify dependence up to lag $L-1$; family identifies the process; counterexample at $L+1$ | Prop 1.2 | proved | main, proof App |
| `prop:markov-pairs` | pairs determine a stationary Markov chain (order $k$: $P_{k+1}$) | Prop 1.3 (+ Rem 1.4, 1.5) | proved | main, proof App |
| `rem:pooling` | equal per-ticker counts make the pooled null implied by per-ticker nulls | Rem 1.6 | - | main |
| `fact:kr-duality` | Kantorovich-Rubinstein duality | Fact 2.2 | fact | main |
| `rem:representation` | the critic sees standardised windows: $\W(P_A,Q_A)$ | Rem 2.3 | - | main |
| `prop:unbiased` | $\E[T\mid\Gsig]=\Dcrit(\fhat)\le\Lip(\ghat)\W(P_A,Q_A)$; conditional variance bound | Prop 2.5 | proved | main, proof App |
| `rem:gp-soft` | $T$ estimates a lower bound; GP is soft; empirical Lipschitz diagnostic | Rem 2.6-2.8 | - | main (short) |
| `rem:holdout` | why hold data out (plug-in $\W$ bias, curse of dimension) | Rem 2.9 | - | main |
| `ass:exch` | A1 (i.i.d., independent halves) and A1' (conditional invariance) | A1, A1' | - | main |
| `prop:exact-level` | exact level of the fixed-critic permutation test, for every critic | Prop 3.1 | proved | main, proof App |
| `lem:exchangeable` | random permutations give an exchangeable vector | Lemma 3.2 | proved | App |
| `lem:rank` | rank bound | Lemma 3.3 | proved | App |
| `rem:studentised` | weaker null, studentised permutation | Rem 3.5(a)-(c) | - | main |
| `rem:combine-seeds` | p-value lottery; $\min(1,2\,\mathrm{median})$ valid | Rem 3.5(d), 8.5 | - | main |
| `ass:consistency` | C1-C5 (C3 idealises the GP) | Assumption C | - | main |
| `lem:uniform-lip` | uniform convergence over $\Lip_c$ | Lemma 4.1 | proved | App |
| `prop:consistency` | consistency of the critic test | Prop 4.2 | proved | main, proof App |
| `cor:w1-limit` | $T\to\W(P_A^*,Q_A^*)$ under idealised training | Cor 4.3 | proved | main |
| `rem:c-realism` | how realistic C3-C5 are | Rem 4.4 | - | main |
| `prop:bayes-logit` | Bayes-optimal logit | Prop 5.1 | proved | main |
| `cor:jeffreys` | logit statistic targets the Jeffreys divergence; accuracy targets a TV lower bound | Cor 5.2, Rem 5.3 | proved | main |
| `prop:oracle-bound` | no two-sample test beats the NP oracle | Prop 5.5 | proved | main, proof App |
| `prop:local-power` | local power factor $r_f\sqrt{\rho\kappa/(1+\kappa)}$, at most 1/2 at defaults | Prop 5.6 | proved (fixed $f$), sketch (learned) | main, proof App |
| `ex:vol-level` | volatility level: $\norm{x}$ critic nearly LR-efficient | Ex 5.7 | proved | main |
| `rem:lr-vs-w1` | (heuristic) LR critic vs $W_1$ critic | Heur 5.8 | heuristic | main |
| `rem:kl-vs-w1` | likelihood ranks by KL, critic by $W_1$; numerical example | §5.5 | computed | main |
| `def:rel-null` | anchored null $\QG\fhat\ge\Ppast\fhat$; two-critic null $\delta^{\mathrm{2c}}\le0$; idealised $W_1$ null | Def 6.1, Rem 6.9(4) | - | main |
| `prop:anchored` | **new**: (a) exact level at $\QG=\Ppast$ for every critic and every $\Pnew$ (permutation of $G$/$\Rpast$ eval labels, $\fhat$ fixed); (b) studentised permutation asymptotically valid for the weaker null; (c) HAC version valid under D1-D4 when eval data are independent of $\Gsig$. Caveats: $G$ calibrated on $\Rpast$ (R1 idealisation), bootstrap generator excluded, non-stationary $\Rpast$ on real data | Rem 6.9(4) + Prop 3.1 + Rem 3.5(b) + Prop 7.2; code `relative_test_anchored` | to write (proof is short) | main, proof App |
| `prop:paired-boot` | paired bootstrap valid for the conditional two-critic null; percentile CI | Prop 6.2 | sketch | main (short) or App |
| `prop:boundary-size` | two-critic test: unconditional size $>\alpha$ at the boundary, $\to1/2$ for fixed critics | Prop 6.8 | proved given 6.2 | main, proof App |
| `rem:rel-guarantee` | what a rejection means; data-dependent null; bootstrap generator | Rem 6.3, 6.4, 6.6 | - | main |
| `rem:naive-perm` | (heuristic) size distortion of the naive permutation test | Heur 7.1 | heuristic | main |
| `ass:mixing` | D1-D4 | Assumption D | - | main |
| `prop:hac` | HAC z asymptotically $N(0,1)$ for a fixed critic | Prop 7.2 | proved (standard results) | main, proof App |
| `fact:mixing` | GARCH and SV are $\beta$-mixing; window coefficients | Fact 7.3 | fact | main |
| `lem:decoupling` | train/eval decoupling by $\beta$-mixing | Lemma 7.4 | proved | main, proof App |
| `prop:ticker-bias` | bias of $T$ under ticker splits on common dates (factor model) | Prop 7.5 | proved | main, proof App |
| `rem:small-T` | few eval dates; DK downward bias; fixed-b comparison (E1 rows) | §7.4 text | - | main |
| `rem:wild` | wild bootstrap for MMD under dependence (no proof; chwialkowski2014wild) | none | - | main |
| `def:power-check` | power check $(\Cdesign,\vartheta^*)$, boundary size | Def 8.1 | - | main |
| `def:verdict` | verdict | Def 8.2 | - | main |
| `prop:verdict` | what the verdict controls (BAD conditional on $\{\delta\le0\}$; GOOD under an unverifiable premise) | Prop 8.3 | proved | main |
| `rem:equivalence` | equivalence-test alternative | Rem 8.4 | - | main (short) |
| `rem:bh` | per-ticker BH / BY / Storey on calibrated p-values | §8.2 | - | main (short) |

Points the theory writer must handle (code differs from theory.md, see §15):
- **Time-block split.** The code assigns $K=10$ blocks at random to the two halves (interleaved,
  no gap). Lemma 7.4 then bounds the coupling by the $\beta$-mixing coefficient at lag 1 at each
  block boundary, which is not small; the theory must say that the split removes the
  common-date bias of `prop:ticker-bias` (training and eval dates are disjoint) but that residual
  train/eval dependence across block boundaries is controlled only empirically (E1, E2 part g).
- **HAC bandwidth.** Andrews (1991) AR(1) plug-in (code default), not the Newey-West rule.
- **Primary real-data p-value** = normal reference DK z; fixed-b values are E1 comparison rows.
- **Anchored test on real data.** $\Rpast$ is non-stationary, so the boundary is the pooled
  TRAIN law and single-seed p-values are not calibrated; the verdict uses $\pmed$ over seeds and
  the power/boundary size of that rule (D1, D2).

## 7. Section 5: Simulation evidence (`sec:sim`, 5 pages)

| Subsection | Label | Claims | Tables / figures |
|---|---|---|---|
| Design | `sec:sim-design` | independent windows, exact permutation p-values; base laws; families (a)-(d); E2-dep families mix, block; replications as run | - |
| Size | `sec:sim-size` | every window test holds its level with independent windows; KS oversized in family (c) | summary sentence; `tab:e2-size` in App C |
| Power | `sec:sim-power` | clustering at matched variance: window tests have power, \WC{} close to MMD / \SigMMD{}, above C2ST and \LRC{} at some sizes (n = 2000, persistence 0.95: in `tab:e2-power`); marginal deviations: pooled KS far more powerful than every window test; all tests below the oracle, as `prop:oracle-bound` and `prop:local-power` predict; dependence-only deviations: KS has no power by construction | `fig:e2-power`, `tab:e2-power` (+ note), `fig:e2dep-power` |
| Relative test at the boundary | `sec:sim-relative` | anchored test at nominal size at $\Law(G)=\Law(\Rpast)$; two-critic over-rejects (`prop:boundary-size`); resample generator: grouped split = size, naive split = leakage | `tab:e2-relative-boundary` (+ note) |
| Learned vs exact LLR | `sec:sim-llr` | how close the learned logit is to $\log p/q$ (Spearman, calibration slope), three training budgets | `tab:e2-llr` (+ note); `fig:e2-llr-scatter` in App C |
| Dependent windows | `sec:sim-dependent` | random-split permutation oversized on consecutive GARCH windows; time-block + HAC and wild bootstrap closer to nominal | `tab:e2-dependent-size` |

Status (2026-11-15): E2 main (families a-d) final: `e2_size.tex`, `e2_power.tex`,
`e2_power_ci.tex`, notes, `e2_power.pdf` exist. Parts e, f, g and E2-dep pending.

## 8. Section 6: Data and generators (`sec:data`, 2.5 pages)

Claims: data construction exactly as DESIGN §2 (479-ticker complete universe after the two
rule-based exclusions; 100-ticker panel; periods; windows); survivorship bias stated; generator
definitions and fitting (variance targeting and cap for GARCH-$t$; Heston QE with
$c\le\sqrt{1.5}$, which avoids zero-variance days but cannot match extreme kurtosis; stationary
bootstrap block rule); only GBM, Merton and GARCH-$t$ have likelihoods.
Tables/figures: `tab:e3-fit-diagnostics` (main), `tab:e3-stylized` + `fig:e3-stylized` (main),
`tab:e3-heston-moments` (App D). Status: `e3_fit_diagnostics.tex`, `e3_heston_moments.tex`
exist; stylized pending.

## 9. Section 7: Empirical results (`sec:emp`, 5.5 pages)

| Subsection | Label | Claims | Tables / figures |
|---|---|---|---|
| Size of the real-data pipeline | `sec:emp-placebo` | E1(a) simulated factor panel with E3 sizes: the naive protocol (ticker split + permutation) is severely oversized; the time-block + DK protocol is much closer to nominal (residual oversize reported, fixed-b rows); joint vs indep cells; E1(b) real halves secondary; E1(c) TRAIN vs TEST distinguishable (non-stationarity) | `tab:e1-size-sim`, `fig:e1-pvalue-qq`, `tab:e1-nonstat`; `tab:e1-size-real` in App E |
| Power check | `sec:emp-power` | pre-specified before real-data results; power and boundary size of the seed-combined rule under D1 and D2 | `tab:e3-power` |
| Tests by generator | `sec:emp-tests` | pooled absolute and relative results per model; "distinguishable from the future but not worse than history" where it occurs | `tab:e3-pooled-tests`, `fig:e3-pvalues` |
| What the critic detects | `sec:emp-witness` | witness values vs realised volatility and max $\abs{r}$ | `fig:e3-critic-witness` |
| Per-ticker tests | `sec:emp-perticker` | calibrated per-ticker p-values, BH; few per-ticker GOOD verdicts expected (power) | `tab:e3-perticker`; `tab:e3-perticker-null` in App E |
| Likelihood agreement | `sec:emp-likelihood` | predictive log-score, Berkowitz, PIT-KS, AG; Kendall tau and Cohen kappa; disagreement is informative (`rem:kl-vs-w1`) | `tab:e3-likelihood`, `tab:e3-agreement`; `tab:e3-likelihood-ag` in App E |
| Verdicts | `sec:emp-verdicts` | GOOD/BAD/INCONCLUSIVE per model, INVALID for the bootstrap, CP variant | `tab:e3-verdicts` |

As run (DESIGN Change log, compute budget): E3 with 5 synthetic seeds, power and realvol 100
replications; E1 real placebo 50 splits. Status: `e3_power.tex`, `e3_pooled_tests.tex`,
`e3_verdicts.tex` already written by running jobs; treat every number as provisional until the
lead declares E3 final. E1 pending.

## 10. Sections 8-10 and appendix

**Section 8, Markowitz (`sec:markowitz`, 4 pages).** `sec:mk-theory`: `lem:quad-loss` (Lemma
9.1), `prop:gmv-sensitivity` (Prop 9.2), `cor:gmv-drivers` (Cor 9.3; (ii)-(iii) heuristic),
`prop:no-free-lunch` (Prop 9.4), `ex:scale-blind` (Ex 9.5), `ex:spread` (Ex 9.6), `ex:corr-blind`
(Ex 9.7), `ex:drift` (Ex 9.8); proofs App A. `sec:mk-design`, `sec:mk-utility`, `sec:mk-link`.
Claims: utility of synthetic-trained portfolios relative to TRTR; the real average correlation
fell from TRAIN to TEST (`tab:e4-correlation`), so a generator that reproduces the past better is
not necessarily more useful; the link between test quantities and utility losses in four views,
conditional on five generators. Tables/figures: `tab:e4-utility`, `tab:e4-correlation`,
`tab:e4-tests` (+ note), `tab:e4-link` (+ note), `fig:e4-utility-vs-distance` (main);
`tab:e4-trtr`, `tab:e4-utility-agg`, `tab:e4-placebo` (+ note), `fig:e4-placebo-size`,
`tab:e4-link-pooled` (App F). As run: 2 test seeds, 60 placebo replications.

**Section 9, Robustness (`sec:robust`, 1.5 pages).** E5 parts run: grid, split (rule power 25
replications), simpower (50 replications); gridsize, gridpower, gridrealvol not run.
`tab:e5-sensitivity`, `fig:e5-sensitivity`, `tab:e5-split` (main); `tab:e5-simpower`,
`tab:e5-calibration` if produced (App G).

**Section 10, Conclusion (`sec:concl`, 1 page)** with `sec:concl-limitations`.

**Appendix.** `app:proofs` (A, 6-8 pages), `app:impl` (B, 3-4: architectures, training, split,
DK, Andrews lag, fixed-b, wild bootstrap, signatures, model fitting, seeds, compute cuts as run),
`app:sim` (C, 3-4), `app:data` (D, 1-2), `app:emp` (E, 2-3), `app:mk` (F, 2-3), `app:robust`
(G, 1-2).

## 11. Templates

Most generated tables are wider than the text at `\small` (e.g. `e2_size`, `e3_power`), so the
default is **template A**: `\fittable` shrinks the tabular to the text width when needed, and the
notes go in `\tablenote{}` (a text-width minipage). `\fittable` must not be used inside
`threeparttable` (threeparttable cannot measure a boxed tabular; compilation fails).

Template A (default; generated tabular, generated or hand-written note):

```latex
\begin{table}[tbp]
  \centering
  \caption{Size under $H_0$ with independent windows (E2).}
  \label{tab:e2-size}
  \small
  \fittable{e2_size}
  \tablenote{\inputnote{e2_size_note}}   % or hand-written: sample sizes, p-value type, brackets
\end{table}
```

Template B (only for a table known to fit the text width; notes as wide as the table):

```latex
\begin{table}[tbp]
  \centering
  \begin{threeparttable}
    \caption{...}\label{tab:e2-llr}
    \small
    \inputtable{e2_llr}
    \begin{tablenotes}[flushleft]\footnotesize
      \item[] \inputnote{e2_llr_note}
    \end{tablenotes}
  \end{threeparttable}
\end{table}
```

Figure:

```latex
\begin{figure}[tbp]
  \centering
  \inputfigure{e2_power}        % \inputfigure[0.8\linewidth]{...} for a narrower figure
  \caption{Power against the four E2 families. ...}
  \label{fig:e2-power}
\end{figure}
```

Captions above tables, below figures. `\inputtable`, `\fittable`, `\inputnote`, `\inputfigure`
print a red `\todo` when the file does not exist yet, so the paper always compiles. If a shrunk
table becomes unreadable (below about 7pt), move it to the appendix or ask for a narrower
generated table rather than editing the generated file.

## 12. Label registry

Sections: `sec:intro`, `sec:lit` (`-tests`, `-dependence`, `-synthetic`, `-econometrics`,
`-portfolio`), `sec:method` (`-setup`, `-critic`, `-competitors`, `-inference`, `-relative`,
`-decision`), `sec:theory` (`-windows`, `-critic`, `-level`, `-consistency`, `-likelihood`,
`-relative`, `-dependence`, `-verdict`), `sec:sim` (`-design`, `-size`, `-power`, `-relative`,
`-llr`, `-dependent`), `sec:data` (`-panel`, `-generators`, `-stylized`), `sec:emp` (`-placebo`,
`-power`, `-tests`, `-witness`, `-perticker`, `-likelihood`, `-verdicts`), `sec:markowitz`
(`sec:mk-theory`, `sec:mk-design`, `sec:mk-utility`, `sec:mk-link`), `sec:robust`, `sec:concl`,
`sec:concl-limitations`, `app:proofs`, `app:impl`, `app:sim`, `app:data`, `app:emp`, `app:mk`,
`app:robust`.

Equations (defined in Section 3 unless stated): `eq:wgan-gp`, `eq:critic-stat`,
`eq:perm-pvalue`, `eq:hac-z`, `eq:dk-var`, `eq:anchored-stat`, `eq:two-critic-stat`,
`eq:lr-oracle`, `eq:mmd-u`, `eq:pmed`, `eq:verdict`; Section 8: `eq:gmv-loss`, `eq:ce`.

Statements: §6 and §10 above (`def:`, `ass:`, `prop:`, `lem:`, `cor:`, `fact:`, `rem:`, `ex:`).

Tables and figures (file stem -> label -> place):

| File | Label | Place | Status |
|---|---|---|---|
| `tables/e1_size_sim.tex` | `tab:e1-size-sim` | §7.1 | pending |
| `figures/e1_pvalue_qq.pdf` | `fig:e1-pvalue-qq` | §7.1 | pending |
| `tables/e1_nonstat.tex` | `tab:e1-nonstat` | §7.1 or App E | pending |
| `tables/e1_size_real.tex` | `tab:e1-size-real` | App E | pending |
| `tables/e2_size.tex` (+ `e2_size_note`) | `tab:e2-size` | App C | final |
| `tables/e2_power.tex` (+ `e2_power_note`) | `tab:e2-power` | §5.3 | final |
| `tables/e2_power_ci.tex` | `tab:e2-power-ci` | App C | final |
| `figures/e2_power.pdf` | `fig:e2-power` | §5.3 | final |
| `tables/e2_relative_boundary.tex` (+ note) | `tab:e2-relative-boundary` | §5.4 | pending |
| `tables/e2_llr.tex` (+ `e2_llr_note`) | `tab:e2-llr` | §5.5 | pending |
| `figures/e2_llr_scatter.pdf` | `fig:e2-llr-scatter` | App C | pending |
| `tables/e2_dependent_size.tex` (+ note if failures) | `tab:e2-dependent-size` | §5.6 | pending |
| `tables/e2dep_size.tex` (+ note) | `tab:e2dep-size` | App C | pending |
| `tables/e2dep_power.tex` (+ note) | `tab:e2dep-power` | App C | pending |
| `figures/e2dep_power.pdf` | `fig:e2dep-power` | §5.3 | pending |
| `tables/e3_fit_diagnostics.tex` | `tab:e3-fit-diagnostics` | §6.2 | exists |
| `tables/e3_heston_moments.tex` | `tab:e3-heston-moments` | App D | exists |
| `tables/e3_stylized.tex` | `tab:e3-stylized` | §6.3 | pending |
| `figures/e3_stylized.pdf` | `fig:e3-stylized` | §6.3 | pending |
| `tables/e3_power.tex` | `tab:e3-power` | §7.2 | exists (provisional) |
| `tables/e3_pooled_tests.tex` | `tab:e3-pooled-tests` | §7.3 | exists (provisional) |
| `figures/e3_pvalues.pdf` | `fig:e3-pvalues` | §7.3 | pending |
| `figures/e3_critic_witness.pdf` | `fig:e3-critic-witness` | §7.4 | pending |
| `tables/e3_perticker.tex` | `tab:e3-perticker` | §7.5 | pending |
| `tables/e3_perticker_null.tex` | `tab:e3-perticker-null` | App E | pending |
| `tables/e3_likelihood.tex` | `tab:e3-likelihood` | §7.6 | pending |
| `tables/e3_likelihood_ag.tex` | `tab:e3-likelihood-ag` | App E | pending |
| `tables/e3_agreement.tex` | `tab:e3-agreement` | §7.6 | pending |
| `tables/e3_verdicts.tex` | `tab:e3-verdicts` | §7.7 | exists (provisional) |
| `tables/e4_utility.tex` | `tab:e4-utility` | §8.3 | exists (provisional) |
| `tables/e4_correlation.tex` | `tab:e4-correlation` | §8.3 | exists |
| `tables/e4_trtr.tex` | `tab:e4-trtr` | App F | exists |
| `tables/e4_utility_agg.tex` | `tab:e4-utility-agg` | App F | exists |
| `tables/e4_tests.tex` (+ `e4_tests_note`) | `tab:e4-tests` | §8.4 | exists (provisional) |
| `tables/e4_placebo.tex` (+ `e4_placebo_note`) | `tab:e4-placebo` | App F | pending |
| `figures/e4_placebo_size.pdf` | `fig:e4-placebo-size` | App F | pending |
| `tables/e4_link.tex` (+ `e4_link_note`) | `tab:e4-link` | §8.4 | pending |
| `tables/e4_link_pooled.tex` | `tab:e4-link-pooled` | App F | pending |
| `figures/e4_utility_vs_distance.pdf` | `fig:e4-utility-vs-distance` | §8.4 | pending |
| `tables/e5_sensitivity.tex` | `tab:e5-sensitivity` | §9 | pending |
| `figures/e5_sensitivity.pdf` | `fig:e5-sensitivity` | §9 | pending |
| `tables/e5_split.tex` | `tab:e5-split` | §9 | pending |
| `tables/e5_simpower.tex` | `tab:e5-simpower` | App G | pending |
| `tables/e5_calibration.tex` | `tab:e5-calibration` | App G | pending (may be partial: gridsize/gridpower not run) |

## 13. Missing references (proposed keys; not in `references.bib`)

Cite these keys where needed and list them in your report; the lead adds verified entries.
"Essential" = a planned statement needs it; "optional" = an existing key can stand in.

| Key | Reference | Needed for | Priority |
|---|---|---|---|
| `kiefer_2005_new` | Kiefer, N.M., Vogelsang, T.J. (2005). A new asymptotic theory for heteroskedasticity-autocorrelation robust tests. *Econometric Theory* 21(6), 1130-1164. doi:10.1017/S0266466605050565 | fixed-b rows (E1), `rem:small-T` | essential |
| `andersen_2008_simple` | Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance* 11(3), 1-42. doi:10.21314/JCF.2008.189 | Heston QE scheme | essential |
| `engle_1996_garch` | Engle, R.F., Mezrich, J. (1996). GARCH for groups. *Risk* 9(8), 36-40 | variance targeting | essential |
| `meinshausen_2009_pvalues` | Meinshausen, N., Meier, L., Buehlmann, P. (2009). p-values for high-dimensional regression. *JASA* 104(488), 1671-1681. doi:10.1198/jasa.2009.tm08647 | $\pmed$ (twice the median) | essential |
| `wilson_1927_probable` | Wilson, E.B. (1927). Probable inference, the law of succession, and statistical inference. *JASA* 22(158), 209-212. doi:10.1080/01621459.1927.10502953 | Wilson intervals in every table | essential |
| `clopper_1934_use` | Clopper, C.J., Pearson, E.S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26(4), 404-413. doi:10.1093/biomet/26.4.404 | $\betaL$, boundary-size intervals | essential |
| `storey_2002_direct` | Storey, J.D. (2002). A direct approach to false discovery rates. *JRSS B* 64(3), 479-498. doi:10.1111/1467-9868.00346 | per-ticker $\hat\pi_0$ | essential |
| `dufour_2006_monte` | Dufour, J.-M. (2006). Monte Carlo tests with nuisance parameters. *J. Econometrics* 133(2), 443-477. doi:10.1016/j.jeconom.2005.06.007 | exactness of the Monte Carlo oracle | essential |
| `vandervaart_1998_asymptotic` | van der Vaart, A.W. (1998). *Asymptotic Statistics*. Cambridge University Press. doi:10.1017/CBO9780511802256 | `prop:local-power`, bootstrap CLT | essential |
| `hoeffding_1952_large` | Hoeffding, W. (1952). The large-sample power of tests based on permutations of observations. *Ann. Math. Statist.* 23(2), 169-192. doi:10.1214/aoms/1177729436 | permutation critical value limit | essential |
| `romano_1990_behavior` | Romano, J.P. (1990). On the behavior of randomization tests without a group invariance assumption. *JASA* 85(411), 686-692. doi:10.1080/01621459.1990.10474928 | `rem:studentised` | essential |
| `mcshane_1934_extension` | McShane, E.J. (1934). Extension of range of functions. *Bull. AMS* 40(12), 837-842. doi:10.1090/S0002-9904-1934-05978-0 | proof of `prop:unbiased` | essential |
| `francq_2019_garch` | Francq, C., Zakoian, J.-M. (2019). *GARCH Models*, 2nd ed. Wiley. doi:10.1002/9781119313472 | GARCH ACF decay, moments | essential |
| `kantorovich_1948_functional` | Kantorovich, L.V. (1948). Functional analysis and applied mathematics. *Uspekhi Mat. Nauk* 3(6), 89-185 | Kantorovich inequality, `prop:gmv-sensitivity` | essential (or `horn_2013_matrix`) |
| `horn_2013_matrix` | Horn, R.A., Johnson, C.R. (2013). *Matrix Analysis*, 2nd ed. Cambridge University Press | same | essential (one of the two) |
| `dowson_1982_frechet` | Dowson, D.C., Landau, B.V. (1982). The Frechet distance between multivariate normal distributions. *J. Multivariate Anal.* 12(3), 450-455. doi:10.1016/0047-259X(82)90077-X | `ex:spread` (Gaussian $W_2$) | essential |
| `olkin_1982_distance` | Olkin, I., Pukelsheim, F. (1982). The distance between two random vectors with given dispersion matrices. *Linear Algebra Appl.* 48, 257-263. doi:10.1016/0024-3795(82)90112-4 | same | optional |
| `mcnemar_1947_note` | McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2), 153-157. doi:10.1007/BF02295996 | paired comparisons (E1 joint/indep, E5) | essential |
| `kendall_1938_new` | Kendall, M.G. (1938). A new measure of rank correlation. *Biometrika* 30(1-2), 81-93. doi:10.1093/biomet/30.1-2.81 | `tab:e3-agreement` | optional |
| `cohen_1960_coefficient` | Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educ. Psychol. Meas.* 20(1), 37-46. doi:10.1177/001316446002000104 | `tab:e3-agreement` | optional |
| `zhou_2025_anchor` | Zhou, Peng, Tian, Liu (2025). Anchor-based maximum discrepancy for relative similarity testing. arXiv:2510.10477 (verified, reserve list of literature.md) | relative-test positioning | recommended |
| `stanczuk_2021_wasserstein` | Stanczuk, J., Etmann, C., Kreusser, L.M., Schoenlieb, C.-B. (2021). Wasserstein GANs work because they fail (to approximate the Wasserstein distance). arXiv:2103.01678 (verified, reserve list) | `rem:gp-soft` | optional (`mallasto2019well`) |
| `petzka_2018_regularization` | Petzka, H., Fischer, A., Lukovnikov, D. (2018). On the regularization of Wasserstein GANs. ICLR | one-sided penalty remark | optional |
| `anil_2019_sorting` | Anil, C., Lucas, J., Grosse, R. (2019). Sorting out Lipschitz function approximation. ICML, PMLR 97, 291-301 | `rem:c-realism` | optional |
| `bester_2011_inference` | Bester, C.A., Conley, T.G., Hansen, C.B. (2011). Inference with dependent data using cluster covariance estimators. *J. Econometrics* 165(2), 137-151. doi:10.1016/j.jeconom.2011.01.007 | `rem:small-T` | optional |
| `dejong_2000_consistency` | de Jong, R.M., Davidson, J. (2000). Consistency of kernel estimators of heteroscedastic and autocorrelated covariance matrices. *Econometrica* 68(2), 407-423. doi:10.1111/1468-0262.00115 | `ass:mixing` D3 | optional |
| `bradley_2005_basic` | Bradley, R.C. (2005). Basic properties of strong mixing conditions. *Probab. Surveys* 2, 107-144. doi:10.1214/154957805100000104 | $\beta$-mixing coefficient, `lem:decoupling` | optional (`herrndorf1984functional`) |
| `kallenberg_2002_foundations` | Kallenberg, O. (2002). *Foundations of Modern Probability*, 2nd ed. Springer. doi:10.1007/978-1-4757-4015-8 | proofs of `prop:window-ident`, `prop:markov-pairs` | optional |
| `dudley_2002_real` | Dudley, R.M. (2002). *Real Analysis and Probability*. Cambridge University Press. doi:10.1017/CBO9780511755347 | `rem:c-realism`, Lemma 4.1 | optional |
| `varadarajan_1958_convergence` | Varadarajan, V.S. (1958). On the convergence of sample probability distributions. *Sankhya* 19, 23-26 | `lem:uniform-lip` | optional |
| `bickel_1981_some` | Bickel, P.J., Freedman, D.A. (1981). Some asymptotic theory for the bootstrap. *Ann. Statist.* 9(6), 1196-1217. doi:10.1214/aos/1176345637 | `prop:paired-boot` | optional |
| `hall_1992_bootstrap` | Hall, P. (1992). *The Bootstrap and Edgeworth Expansion*. Springer. doi:10.1007/978-1-4612-4384-7 | centred bootstrap p-value | optional (`hall1991two`) |
| `vovk_2020_combining` | Vovk, V., Wang, R. (2020). Combining p-values via averaging. *Biometrika* 107(4), 791-808. doi:10.1093/biomet/asaa027 | `rem:combine-seeds` | optional |
| `blackwell_1953_equivalent` | Blackwell, D. (1953). Equivalent comparisons of experiments. *Ann. Math. Statist.* 24(2), 265-272. doi:10.1214/aoms/1177729032 | `prop:no-free-lunch` remark | optional |
| `merton_1980_estimating` | Merton, R.C. (1980). On estimating the expected return on the market. *J. Financ. Econ.* 8(4), 323-361. doi:10.1016/0304-405X(80)90007-0 | `ex:drift` | optional (`chopra1993effect`) |
| `best_1991_sensitivity` | Best, M.J., Grauer, R.R. (1991). On the sensitivity of mean-variance-efficient portfolios to changes in asset means. *Rev. Financ. Stud.* 4(2), 315-342. doi:10.1093/rfs/4.2.315 | `ex:drift` | optional |
| `schuirmann_1987_comparison` | Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *J. Pharmacokinet. Biopharm.* 15(6), 657-680. doi:10.1007/BF01068419 | `rem:equivalence` | optional |
| `jeffreys_1946_invariant` | Jeffreys, H. (1946). An invariant form for the prior probability in estimation problems. *Proc. R. Soc. A* 186, 453-461. doi:10.1098/rspa.1946.0056 | `cor:jeffreys` | optional |
| `riskmetrics_1996_technical` | J.P. Morgan / Reuters (1996). *RiskMetrics Technical Document*, 4th ed. | Heston start state (EWMA 0.94), App B | optional |

## 14. Open decisions for the lead

1. Critic architecture (time-pooled pilot): affects `sec:method-critic`, `app:impl`, E1, E2 part
   e, E3 (one sentence in Section 3 and the appendix; the rest is architecture-agnostic).
2. Whether `prop:anchored` is stated as a Proposition (recommended; it is the paper's primary
   relative test and its proof is three lines from `prop:exact-level`).
3. Whether `tab:e1-nonstat` and `tab:e2-llr` go in the main text or the appendix (page budget).
4. Title wording (current: as requested).

## 15. Known inconsistencies between the docs and the code (describe the code)

1. **Time-block split.** theory.md §7.3-7.4 and note 11.5 recommend one training block and one
   evaluation block of the TEST dates separated by a gap (about 14 / 1 / 15 dates). The code
   (`critic.time_block_labels` + `train_eval_split`, used with `time_blocks=10`) cuts each
   sample's own date range into 10 equal blocks and assigns whole blocks at random to the two
   halves: blocks interleave and there is no gap. The theory.md pilot rates of note 11.5 refer to
   the single-split design.
2. **HAC lag.** theory.md §7.2 states the Newey-West (1994) rule as default; the code default is
   the Andrews (1991) AR(1) plug-in (DESIGN Change log). Newey-West is an E5 option.
3. **Primary real-data p-value.** theory.md §7.4 and note 11.5 recommend fixed-b critical values
   as primary; the pipeline reports the normal-reference DK p-value; fixed-b p-values exist only as
   post-hoc comparison rows in E1.
4. **Heston scheme.** theory.md Remark 1.5 says "Heston (Euler)"; the code uses Andersen's QE
   scheme with $c\le\sqrt{1.5}$ (DESIGN Change log).
5. **Anchored test.** theory.md has no proposition for it (only Remark 6.9(4)); `prop:anchored`
   must be written. theory.md §6 calls the two-critic statistic $\Delta$ (notation.md §4 item 2).
6. **Learning rate.** docs/literature.md (gulrajani2017improved entry) says "our critic uses ...
   lr = 1e-4"; the code default is lr = 1e-3 (DESIGN Change log). Write "the gradient-penalty
   weight as recommended by Gulrajani et al.; learning rate 1e-3 (E5 also runs 1e-4)".
7. **Replication counts.** Script docstrings still give the pre-budget settings: e3_sp500.py and
   e5_robustness.py say 10 synthetic seeds and 200 power replications (FULL configs: 5 seeds, 100
   replications; E5 split power 25, simpower 50); e1_placebo.py says 100 real splits (FULL: 50);
   e2_ground_truth.py says part e at n in {500, 2000} (FULL: 500 only); e2_dependence.py says
   n in {500, 2000} with 200/100 replications (FULL: n = 2000 only, 100/50); e4 runs with 2 test
   seeds and 60 placebo replications from the command line (`run_full.sh`). Report what was run
   (the CSV metadata `results/<exp>/meta*.json` is authoritative).
