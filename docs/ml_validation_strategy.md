# ML Validation Strategy for Non-Stationary Energy Markets

## Why Static Splits Fail

Energy market data is non-stationary. Structural breaks (for example market
coupling, policy updates, fuel shocks, outage regimes, and weather extremes)
shift feature-target relationships over time. A static random split mixes
historical regimes and future regimes, which inflates validation scores and
underestimates deployment risk.

For this reason, validation must follow chronological order and preserve
realistic decision-time information sets.

## Purged Walk-Forward Cross-Validation

We use an expanding-window time-series split:

1. Train on earliest history.
2. Skip a purge/embargo block.
3. Validate on the next contiguous future block.
4. Expand training window and repeat.

### Why the 72h Purge Is Mandatory

Our sequence targets include horizons up to +72h. Therefore, the last training
row contains labels that overlap the first validation rows if no embargo is
applied.

Formally, with target horizon `h_max = 72`, a training sample at time `t`
contains `y(t+1) ... y(t+72)`. If validation starts at `t+1`, label overlap is
total. To remove this leakage channel, we enforce:

`val_start >= train_end + 72h`

This is implemented as `gap_hours=72` in `PurgedTimeSeriesSplit`.

## Train on Clipped, Evaluate on True

Two target views are maintained:

- `y_train_*`: clipped targets for optimization stability
- `y_true_*`: unclipped market truth for economic evaluation

Training uses clipped labels to stabilize gradients and reduce domination by
rare spikes. Evaluation always uses unclipped truth to preserve correct PnL and
market realism.

Typical workflow:

```python
for train_idx, val_idx in splitter.split(X, timestamps=ts):
    model.fit(X.iloc[train_idx], y_train.iloc[train_idx])
    y_pred = model.predict(X.iloc[val_idx])
    score = metric(y_pred, y_true.iloc[val_idx])
```

## Visual Audit

The script `scripts/visualize_cv_folds.py` plots each fold with:

- Blue: training window
- Red: 72h purge block (excluded)
- Green: validation window

This provides visual proof that causality and embargo constraints are enforced.

