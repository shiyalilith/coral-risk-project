Coral Heat Stress Early Warning — Research Log

1. Project Goal

The goal of this project is to explore whether short-term coral heat stress risk can be predicted using satellite-derived sea surface temperature (SST) data.

More specifically:

- Can we detect early warning signals of bleaching conditions?
- How reliable are predictions under limited data?
- How does the model behave under changing environmental conditions?

⸻

2. Initial Approach

I built a pipeline that:

1. Loads NOAA Coral Reef Watch SST data
2. Computes:
   - temperature anomaly
   - heat stress (thresholded anomaly)
   - rolling stress (proxy for accumulated thermal stress)
3. Defines a future risk label:
   - based on rolling stress shifted forward in time
4. Trains an XGBoost classifier
5. Evaluates using ROC-AUC and confusion matrix

⸻

3. Early Results (Misleading Performance)

Initial results showed:

- ROC-AUC ≈ 0.65

However, this was achieved under a messy and inconsistent notebook setup.

Possible issues:

- duplicated model training
- inconsistent feature definitions
- potential leakage or misalignment

Conclusion:
These results were likely overly optimistic.

⸻

4. Clean Pipeline (True Baseline)

After restructuring into a clean, single pipeline:

- ROC-AUC dropped to ≈ 0.56

Confusion matrix:

[[23  0]
[27  4]]

Interpretation:

- Model predicts “no risk” most of the time
- Very few true positives
- Almost no false positives

Conclusion:

The model is conservative and under-detects risk events.

⸻

5. Key Insight

The limitation is not the model, but the data and signal:

- Only ~1 year of data
- Very few positive (risk) events
- Weak variability in features
- Proxy label (rolling stress threshold) is noisy

The model is behaving rationally given the data:

Predicting “no risk” minimizes error under class imbalance.

⸻

6. Adjustments Tried

- Added lag features (SST_lag1, lag7, lag14)
- Added anomaly and rolling stress
- Added temporal derivatives:
  - temperature change
  - acceleration
- Adjusted classification threshold (0.5 → 0.3)
- Used class weighting (scale_pos_weight)

Result:

- Slight improvements in behavior
- But overall performance still limited

⸻

7. Reframing the Problem

Original framing:

“Predict bleaching risk”

Better framing:

“Detect early warning signals of thermal stress under uncertainty”

This shifts the goal from accuracy → robust detection of rare events.

⸻

8. Planned Next Step: Distribution Shift

To test robustness, I will simulate environmental change:

- Increase SST by +1°C
- Recompute features
- Evaluate model performance under shifted conditions

Key question:

Does the model remain reliable under climate change?

This tests:

- generalization
- robustness
- sensitivity to distribution shift

⸻

9. Open Questions

- Is SST alone sufficient to predict stress?
- How much does performance improve with multi-year data?
- Is the rolling stress proxy an adequate label?
- Can temporal models (LSTM, etc.) capture dynamics better?
- How early can risk realistically be detected?

⸻

10. Reflection

The project revealed an important pattern:

- Early results can be misleading without clean pipelines
- Real performance is often lower but more informative
- Weak signal, not model complexity, is the main bottleneck

The model achieves moderate performance in-distribution (ROC ~0.65), but collapses under a +1°C shift, predicting risk universally (ROC ~0.5). This indicates reliance on absolute temperature rather than invariant stress dynamics.

⸻

11. Data Expansion (Critical Improvement)

The dataset was extended from ~1 year to ~6 years of SST observations.

Impact:

- Increased number of positive (risk) events
- Improved seasonal coverage
- Reduced noise and instability
- Enabled learning of consistent temporal patterns

Result:

Model performance improved significantly once more data was introduced.

⸻

12. Correcting Label Leakage

Earlier versions used rolling stress as both feature and target proxy, introducing implicit leakage.

Fix:

- Switched to predicting future anomaly instead of rolling stress
- Ensured features do not directly encode the target

Result:

- Removed artificial performance inflation
- Established a more realistic prediction task

⸻

13. Improved Model Performance

With cleaner labels and multi-year data:

- ROC-AUC ≈ 0.91 (in-distribution)
- ROC-AUC ≈ 0.84 under +1°C shift

Interpretation:

- Model captures meaningful temporal patterns
- Performance remains strong under moderate distribution shift
- Indicates real predictive signal (not memorization)

⸻

14. Distribution Shift Experiments (Manual)

Instead of automated loops (which caused instability), shifts were applied manually.

Results:

- 0°C → 0.918
- +0.5°C → 0.887
- +1°C → ~0.84
- +2°C → ~0.60

⸻

15. Key Finding: Non-Linear Degradation

Performance degrades gradually for small shifts but collapses at +2°C.

Interpretation:

- Model is robust to mild climate variation
- Fails under extreme distribution shift
- Indicates reliance on absolute SST thresholds

This suggests the model does not learn invariant stress dynamics, but rather temperature-dependent heuristics.

⸻

16. Behavioral Insight

The model appears to learn:

- seasonal patterns
- temperature buildup
- short-term temporal structure

However, it struggles when:

- baseline temperature shifts significantly
- “normal” conditions resemble previously “extreme” ones

⸻

17. Visual Analysis Insight

Observed behavior:

- Ground truth labels are binary → appear flat and block-like
- Predictions are probabilistic → appear noisy and continuous

Long flat periods (e.g., May–Nov 2023):

- Correspond to low-stress seasons
- Reflect stable ocean temperatures
- Confirm model alignment with real-world seasonality

⸻

18. Updated Conclusion

The model:

- Performs strongly under current conditions
- Maintains robustness under moderate warming
- Breaks down under large climate shifts

Core insight:

Machine learning models trained on historical environmental data may fail under future climate conditions due to distribution shift.

⸻

19. Research Direction Upgrade

The project has evolved from:

“Can we predict coral bleaching risk?”

to:

“How do predictive models behave under climate-induced distribution shift?”

This reframing aligns the project with:

- robustness analysis
- generalization under uncertainty
- climate-aware machine learning

⸻

20. Next Steps

- Evaluate performance across multiple reef locations
- Test temporal generalization (train on past → predict future years)
- Explore invariant features (relative anomalies vs absolute SST)
- Investigate calibration and uncertainty
- Compare retraining vs static model under shift
