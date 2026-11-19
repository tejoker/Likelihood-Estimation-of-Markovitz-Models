# Roadmap: what remains beyond the running computations (2026-11-19)

The current results are in [`STATUS_v2.md`](STATUS_v2.md). This page lists the work that adds
information beyond them: new data, new experiments, theory, and the paper rewrite. Runs that
are already queued are not repeated here.

**Publication path:**
1. CREST working paper and arXiv, from the v2 results. This secures priority: close work is
   appearing (Zhou et al. 2025; Bai et al. 2026).
2. Journal version, in this order of targets:
   - JASA (Theory & Methods) or the Journal of Econometrics;
   - then the Journal of Business & Economic Statistics;
   - fallback: the Journal of Financial Econometrics or Quantitative Finance.

**Labels:**
- **[WP]** needed for the working paper.
- **[J]** needed for the journal version.
- **Owner:** *author* means it needs a decision, data access or writing by the author; *agent*
  means it can be done from the repository.

## A. Data (highest value: the only way to sharpen the verdicts)

The INCONCLUSIVE at ±20% comes from having only 2.5 years of TEST data in one market regime.
More seeds or replications cannot fix that. More real data can.

| # | Item | Why | Tag | Owner |
|---|---|---|---|---|
| A1 | **Extended export, minimum:** same 503 stocks and fields (`PX_LAST`, `PX_VOLUME`), 2018-01-01 to the latest close (~16 MB) | About 17 months of unseen TEST data: a blind confirmation sample | WP | author |
| A2 | **Extended export, full:** every stock in the S&P 500 at any time since 2005, including delisted ones, 2005 to today (~150-200 MB) | Several train/test periods (2008, 2020, 2022), so verdicts repeat across regimes. Removes survivorship and look-ahead bias | J | author |
| A3 | Total return (`TOT_RETURN_INDEX_GROSS_DVDS`), same stocks and dates | Markowitz on true returns; referees will note that dividends are missing | J | author |
| A4 | Index membership history (month-end constituents) | Point-in-time universe | J | author |
| A5 | `SPX`, `SPXT`, `VIX` indices and the 3-month T-bill (`USGG3M`), daily | Regime labels and the risk-free rate | J | author |
| A6 | A second market or asset class (e.g. STOXX Europe 600, FX or futures) | External validity | J | author |
| A7 | **Before opening any new file:** write protocol amendment A3 in `PROTOCOL_v2.md` (split, rules, tolerance definition) | The new sample then counts as a pre-registered confirmation, not a fishing expedition | WP | agent |
| A8 | Loader for the new layout; new files stay out of git (`.gitignore`) | — | WP | agent |
| A9 | **Data licence.** The current price file is committed in this public repository. Check the vendor's redistribution terms; if needed, make the repository private or remove the file from history | Legal; journals also ask for a data-availability statement | WP | author |

CRSP via WRDS is the alternative for A2-A4 if CREST provides access.

## B. New experiments that add information

### For the working paper

| # | Experiment | What it adds | Tag | Compute |
|---|---|---|---|---|
| B1 | **Directional tolerance, pre-registered on the new sample.** Fix the A1 mis-specification: in D2, drop the power requirement for "calmer than the past" errors. Then apply the certified-tolerance rule to the A1 data | Turns the post-hoc ±40% into a pre-registered, confirmed result | WP | hours (GPU) |
| B2 | **Fixed realised-volatility statistic inside the GOOD/BAD rule**, as a baseline | Answers the referee question "is the learned critic needed?" (in v1 the fixed statistic reached 69-94% of the critic's strength) | WP | hours |
| B3 | **Significant ranking of generators.** Paired anchored tests between generators (GARCH-t vs Merton vs QuantGAN ...), alongside the Amisano-Giacomini likelihood comparisons | A ranking with error control, beyond the per-generator yes/no | WP | hours |
| B4 | **Witness diagnostics for QuantGAN.** Where the critic separates QuantGAN from real data (tails, volatility clustering) | Makes the neural-generator section a demonstration of the method, not engineering | WP | small |
| B5 | Holm correction across generators, reported next to the per-generator verdicts | Pre-empts a multiple-testing objection | WP | none |

### For the journal version

| # | Experiment | What it adds | Tag | Compute |
|---|---|---|---|---|
| B6 | **Size control under non-stationarity.** Calibrate the cut-offs from real past-vs-past comparisons over rolling sub-periods (needs A2) | Fixes the main methodological gap: boundary size 0.28 in D2 on the alternative split | J | 1-2 nights (GPU) |
| B7 | **Rolling-origin evaluation** over 2005-2025: one verdict per train/test period, then combined | Verdicts that repeat across regimes; the strongest answer to "inconclusive" | J | 1-2 nights |
| B8 | **Portfolio-aware critic.** Restrict the critic to portfolio returns (GMV, mean-variance weights) | Turns the negative Markowitz result into a method that tests what allocation actually uses | J | small |
| B9 | **Markowitz over several periods**, with inference across dates and total returns (A3) | The current evaluation rests on one period with falling correlations (0.42 to 0.27) | J | hours |
| B10 | **Over-rejection of the Markowitz placebo for GARCH-type generators** (0.17-0.45): find the cause, then test remedies (an embargo between TRAIN and evaluation blocks, longer blocks) | Closes an open validity gap | J | hours |
| B11 | **A second neural generator:** signature-WGAN or diffusion | Referees from machine learning will ask for more than one | J | ~1 h of training + tests |
| B12 | **Heston fit:** lift the vol-of-vol cap that limits its tails, or state it as a limitation | Fair comparison | J | small |

## C. Theory [J]

| # | Result | Status |
|---|---|---|
| C1 | Asymptotics of the time-block HAC critic test, including fixed-b | Heuristic today (`docs/theory.md`) |
| C2 | Power of the time-pooled critic: averaging over the window multiplies the effective sample size by about L | Explains the main empirical gain |
| C3 | The anchored relative test under non-stationarity: what its null region is, and why a calmer generator falls inside it (the A1 lesson) | Needed to justify B1 and B6 |
| C4 | Validity of the calibration in B6 | Depends on B6 |

C1, C2 and C3 together decide between JASA / the Journal of Econometrics and JBES.

## D. Paper

| # | Task | Tag | Owner |
|---|---|---|---|
| D1 | **Rewrite from the v2 results.** Pooled critic as the main test, 10- and 50-seed verdicts, both tolerance readings, the QuantGAN section, GPU reproducibility | WP | agent, author review |
| D2 | **Lead with the certified tolerance**, not with INCONCLUSIVE: "GBM is BAD; the others are GOOD within ±X% volatility" | WP | agent |
| D3 | **Condense.** The main text is 84 pages, against the usual 40-50; move the detail to the appendix | WP | agent |
| D4 | **Reposition for the journal.** The general method comes first (testing generative models against a non-stationary reference, under dependence); finance is the showcase | J | agent, author |
| D5 | Literature update: Zhou et al. 2025 (anchor-based discrepancy), Bai et al. 2026, arXiv 2512.21791 | WP | agent |
| D6 | **Authors and affiliations** (still a placeholder) | WP | author |
| D7 | Data-availability statement and a reproducibility package (scripts, GPU environment, how to obtain the data) | WP | agent, after A9 |
| D8 | Independent referee pass on the v2 draft: statistics, finance, ML, and a number audit | WP | agent |
| D9 | CREST seminar; SoFiE and ICAIF talks before submitting | J | author |

## Decisions needed from the author

1. Authors and affiliations (D6).
2. The data licence for the committed price file (A9).
3. Which export is possible: the minimum A1 or the full A2-A5. Is WRDS/CRSP available at CREST?
4. The target: the JASA / Journal of Econometrics route (needs C1-C3 and B6) or JBES.
5. The second neural generator for B11, if a specific one is preferred.
