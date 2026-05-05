"""
IT2032 - Output Analysis for Non-Terminating Simulations
=========================================================
Demonstrates all three methods:
  1. Welch Method
  2. Replication-Deletion Approach
  3. Batch Means Method

Scenario: Simulating average queue length in a call center
         (a non-terminating system — it runs indefinitely)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

np.random.seed(42)
plt.rcParams.update({'figure.facecolor': '#FAFAFA', 'axes.facecolor': '#F5F5F5',
                     'axes.grid': True, 'grid.alpha': 0.4, 'grid.color': '#CCCCCC',
                     'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})

# ─────────────────────────────────────────────
# SHARED: Simulate a queueing system
# ─────────────────────────────────────────────
def simulate_queue(n_obs: int, seed: int, true_steady: float = 12.0) -> np.ndarray:
    """
    Simulates queue-length observations.
    Starts far from steady state to create a visible transient phase.
    Returns n_obs data points.
    """
    rng = np.random.default_rng(seed)
    obs = np.zeros(n_obs)
    level = 30.0  # start high (transient)
    for t in range(n_obs):
        # Decay toward true steady state, with random noise
        noise = rng.normal(0, 2.5)
        level += 0.15 * (true_steady - level) + noise
        level = max(0, level)
        obs[t] = level
    return obs


# ═══════════════════════════════════════════════════════════
# METHOD 1: WELCH METHOD
# ═══════════════════════════════════════════════════════════
def welch_method(n_reps: int = 10, n_obs: int = 200, window: int = 10):
    """
    Steps:
    1. Run R independent replications
    2. Compute ensemble average Y̅(t) across replications at each time t
    3. Smooth Y̅(t) using a moving average window m → W(t)
    4. Identify warm-up period l* = first t where W(t) stabilizes
    5. Use data from l* onwards for steady-state estimation
    """
    print("=" * 60)
    print("METHOD 1: WELCH METHOD")
    print("=" * 60)

    # Step 1: Run replications
    replications = np.array([
        simulate_queue(n_obs, seed=100 + r) for r in range(n_reps)
    ])

    # Step 2: Ensemble average Y̅(t) across replications
    Y_bar = replications.mean(axis=0)          # shape: (n_obs,)

    # Step 3: Smooth with moving average of window m → W(t)
    def moving_avg(arr, m):
        W = np.zeros(len(arr))
        for t in range(len(arr)):
            lo = max(0, t - m)
            hi = min(len(arr) - 1, t + m)
            W[t] = arr[lo:hi+1].mean()
        return W

    W = moving_avg(Y_bar, window)

    # Step 4: Find warm-up cutoff l*
    # Heuristic: find where W(t) stays within ±5% of the tail mean
    tail_mean = W[n_obs // 2:].mean()
    tolerance = 0.05 * tail_mean
    l_star = n_obs // 3  # default fallback
    for t in range(10, n_obs - 20):
        if np.all(np.abs(W[t:] - tail_mean) < tolerance):
            l_star = t
            break

    # Step 5: Steady-state estimate from l* onward
    steady_data = Y_bar[l_star:]
    steady_mean = steady_data.mean()

    print(f"  Replications (R)   : {n_reps}")
    print(f"  Observations/rep   : {n_obs}")
    print(f"  Smoothing window m : {window}")
    print(f"  Warm-up cutoff l*  : {l_star}  ({l_star/n_obs*100:.0f}% of run)")
    print(f"  Steady-state mean  : {steady_mean:.4f}")
    print(f"  Data retained      : {n_obs - l_star} obs ({(n_obs-l_star)/n_obs*100:.0f}%)\n")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle("METHOD 1: Welch Method — Warm-up Detection", fontweight='bold', fontsize=13)
    t_axis = np.arange(1, n_obs + 1)

    # Top: individual replications + ensemble average
    ax = axes[0]
    for i, rep in enumerate(replications):
        ax.plot(t_axis, rep, alpha=0.2, linewidth=0.8, color='steelblue')
    ax.plot(t_axis, Y_bar, color='steelblue', linewidth=2, label='Ensemble avg Ȳ(t)')
    ax.axvline(l_star, color='red', linestyle='--', linewidth=1.5, label=f'Warm-up cutoff l*={l_star}')
    ax.axvspan(0, l_star, alpha=0.08, color='red')
    ax.set_ylabel("Queue Length")
    ax.set_title("Individual Replications + Ensemble Average")
    ax.legend(loc='upper right')

    # Bottom: smoothed W(t) showing where steady state begins
    ax2 = axes[1]
    ax2.plot(t_axis, Y_bar, color='steelblue', linewidth=1.5, alpha=0.5, label='Ȳ(t) raw')
    ax2.plot(t_axis, W, color='green', linewidth=2.5, label=f'W(t) smoothed (m={window})')
    ax2.axhline(tail_mean, color='gray', linestyle=':', linewidth=1.5, label=f'Steady mean ≈ {tail_mean:.2f}')
    ax2.axhline(tail_mean + tolerance, color='gray', linestyle=':', alpha=0.4)
    ax2.axhline(tail_mean - tolerance, color='gray', linestyle=':', alpha=0.4)
    ax2.axvline(l_star, color='red', linestyle='--', linewidth=1.5, label=f'l* = {l_star}')
    ax2.axvspan(0, l_star, alpha=0.08, color='red')
    ax2.fill_between(t_axis, tail_mean - tolerance, tail_mean + tolerance, alpha=0.08, color='green', label='±5% tolerance band')
    ax2.set_xlabel("Simulation Time (t)")
    ax2.set_ylabel("Queue Length")
    ax2.set_title("Smoothed Average W(t) — Stable region = steady state")
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/welch_method.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("  [Saved] welch_method.png\n")
    return l_star, steady_mean


# ═══════════════════════════════════════════════════════════
# METHOD 2: REPLICATION-DELETION APPROACH
# ═══════════════════════════════════════════════════════════
def replication_deletion(n_reps: int = 15, n_obs: int = 200, delete_frac: float = 0.20,
                          confidence: float = 0.95):
    """
    Steps:
    1. Run R independent replications
    2. Delete the first (delete_frac × n_obs) observations from EACH replication
    3. Compute per-replication mean Y̅ᵢ from remaining data
    4. Compute grand mean Ȳ and confidence interval using t-distribution:
           Ȳ ± t(α/2, R-1) × S/√R
    """
    print("=" * 60)
    print("METHOD 2: REPLICATION-DELETION APPROACH")
    print("=" * 60)

    delete_n = int(n_obs * delete_frac)    # observations to discard per replication
    alpha = 1 - confidence

    # Steps 1–3: replicate, delete, compute means
    rep_means = []
    for r in range(n_reps):
        data = simulate_queue(n_obs, seed=200 + r)
        steady_data = data[delete_n:]      # discard warm-up
        rep_means.append(steady_data.mean())

    rep_means = np.array(rep_means)

    # Step 4: Grand mean and CI
    grand_mean = rep_means.mean()
    S = rep_means.std(ddof=1)              # sample std dev
    t_crit = stats.t.ppf(1 - alpha / 2, df=n_reps - 1)
    half_width = t_crit * S / np.sqrt(n_reps)
    ci_lower = grand_mean - half_width
    ci_upper = grand_mean + half_width

    print(f"  Replications (R)    : {n_reps}")
    print(f"  Observations/rep    : {n_obs}")
    print(f"  Deleted (warm-up)   : {delete_n} obs ({delete_frac*100:.0f}%)")
    print(f"  Kept per rep        : {n_obs - delete_n} obs")
    print(f"  t-critical (df={n_reps-1})  : {t_crit:.4f}")
    print(f"  Grand mean Ȳ        : {grand_mean:.4f}")
    print(f"  Std deviation S     : {S:.4f}")
    print(f"  Half-width          : {half_width:.4f}")
    print(f"  {confidence*100:.0f}% CI             : [{ci_lower:.4f}, {ci_upper:.4f}]\n")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("METHOD 2: Replication-Deletion Approach", fontweight='bold', fontsize=13)

    # Left: show one replication with warm-up highlighted
    ax = axes[0]
    example = simulate_queue(n_obs, seed=200)
    t_axis = np.arange(1, n_obs + 1)
    ax.plot(t_axis[:delete_n], example[:delete_n], color='tomato', linewidth=1.5, label='Warm-up (deleted)')
    ax.plot(t_axis[delete_n:], example[delete_n:], color='steelblue', linewidth=1.5, label='Steady-state (kept)')
    ax.axvline(delete_n, color='red', linestyle='--', linewidth=1.5, label=f'Delete point = {delete_n}')
    ax.axhline(example[delete_n:].mean(), color='green', linestyle=':', linewidth=1.5,
               label=f'Rep mean = {example[delete_n:].mean():.2f}')
    ax.set_xlabel("Simulation Time (t)")
    ax.set_ylabel("Queue Length")
    ax.set_title("Single Replication: Warm-up vs Steady State")
    ax.legend(fontsize=9)

    # Right: per-replication means with CI
    ax2 = axes[1]
    x = np.arange(1, n_reps + 1)
    colors = ['tomato' if m < ci_lower or m > ci_upper else 'steelblue' for m in rep_means]
    bars = ax2.bar(x, rep_means, color=colors, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax2.axhline(grand_mean, color='navy', linewidth=2.5, label=f'Grand mean Ȳ = {grand_mean:.2f}')
    ax2.axhline(ci_upper, color='green', linestyle='--', linewidth=1.5, label=f'{confidence*100:.0f}% CI [{ci_lower:.2f}, {ci_upper:.2f}]')
    ax2.axhline(ci_lower, color='green', linestyle='--', linewidth=1.5)
    ax2.fill_between([0, n_reps + 1], ci_lower, ci_upper, alpha=0.07, color='green')
    ax2.set_xlabel("Replication (i)")
    ax2.set_ylabel("Per-replication mean Ȳᵢ")
    ax2.set_title(f"Per-Replication Means + {confidence*100:.0f}% Confidence Interval")
    ax2.set_xticks(x)
    ax2.legend(fontsize=9)

    # Annotate formula
    formula = f"Ȳ ± t(α/2, R-1) × S/√R\n= {grand_mean:.2f} ± {t_crit:.2f} × {S:.2f}/√{n_reps}\n= {grand_mean:.2f} ± {half_width:.2f}"
    ax2.text(0.97, 0.05, formula, transform=ax2.transAxes, fontsize=8.5,
             ha='right', va='bottom', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/replication_deletion.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("  [Saved] replication_deletion.png\n")
    return grand_mean, ci_lower, ci_upper


# ═══════════════════════════════════════════════════════════
# METHOD 3: BATCH MEANS METHOD
# ═══════════════════════════════════════════════════════════
def batch_means(n_obs: int = 1000, warmup_frac: float = 0.15,
                n_batches: int = 20, confidence: float = 0.95):
    """
    Steps:
    1. Run ONE long simulation
    2. Discard the first (warmup_frac × n_obs) observations as warm-up
    3. Divide remaining data into k equal-sized batches
    4. Compute batch mean Ȳⱼ for each batch j = 1..k
    5. Treat batch means as ~independent observations
    6. Compute grand mean + CI:
           Ȳ ± t(α/2, k-1) × S_batch/√k
    """
    print("=" * 60)
    print("METHOD 3: BATCH MEANS METHOD")
    print("=" * 60)

    alpha = 1 - confidence
    warmup_n = int(n_obs * warmup_frac)

    # Step 1: Single long run
    data = simulate_queue(n_obs, seed=999)

    # Step 2: Discard warm-up
    steady_data = data[warmup_n:]
    usable_n = len(steady_data)

    # Step 3: Divide into k batches (trim remainder)
    batch_size = usable_n // n_batches
    usable_trimmed = steady_data[:batch_size * n_batches]
    batches = usable_trimmed.reshape(n_batches, batch_size)

    # Step 4: Batch means
    batch_means_arr = batches.mean(axis=1)

    # Step 5–6: Grand mean + CI
    grand_mean = batch_means_arr.mean()
    S = batch_means_arr.std(ddof=1)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n_batches - 1)
    half_width = t_crit * S / np.sqrt(n_batches)
    ci_lower = grand_mean - half_width
    ci_upper = grand_mean + half_width

    # Autocorrelation check between consecutive batch means
    if n_batches > 3:
        r1 = np.corrcoef(batch_means_arr[:-1], batch_means_arr[1:])[0, 1]
    else:
        r1 = 0

    print(f"  Total observations  : {n_obs}")
    print(f"  Warm-up discarded   : {warmup_n} obs ({warmup_frac*100:.0f}%)")
    print(f"  Usable observations : {usable_n}")
    print(f"  Batches (k)         : {n_batches}")
    print(f"  Batch size (m)      : {batch_size} obs/batch")
    print(f"  Lag-1 autocorr r₁   : {r1:.4f}  ({'OK ✓' if abs(r1) < 0.2 else 'High — try larger batch size'})")
    print(f"  t-critical (df={n_batches-1})  : {t_crit:.4f}")
    print(f"  Grand mean Ȳ        : {grand_mean:.4f}")
    print(f"  Std deviation S     : {S:.4f}")
    print(f"  Half-width          : {half_width:.4f}")
    print(f"  {confidence*100:.0f}% CI             : [{ci_lower:.4f}, {ci_upper:.4f}]\n")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("METHOD 3: Batch Means Method", fontweight='bold', fontsize=13)
    t_axis = np.arange(1, n_obs + 1)
    t_steady = np.arange(warmup_n + 1, n_obs + 1)

    # Top: full time series with warm-up and batch divisions
    ax = axes[0]
    ax.plot(t_axis[:warmup_n], data[:warmup_n], color='tomato', linewidth=1.2, label='Warm-up (deleted)')
    ax.plot(t_steady, data[warmup_n:], color='steelblue', linewidth=1.0, alpha=0.7, label='Steady-state data')

    # Draw batch boundaries
    for b in range(n_batches + 1):
        x_line = warmup_n + b * batch_size
        ax.axvline(x_line, color='darkorange', linewidth=0.7, alpha=0.5)

    # Draw batch mean lines
    for b in range(n_batches):
        x_lo = warmup_n + b * batch_size + 1
        x_hi = warmup_n + (b + 1) * batch_size
        bm = batch_means_arr[b]
        ax.hlines(bm, x_lo, x_hi, colors='green', linewidth=2.5, zorder=5)

    ax.axvline(warmup_n, color='red', linestyle='--', linewidth=2, label=f'Warm-up end (t={warmup_n})')
    ax.axhline(grand_mean, color='navy', linewidth=1.5, linestyle=':', label=f'Grand mean = {grand_mean:.2f}')

    warm_patch = mpatches.Patch(color='tomato', label='Warm-up (deleted)')
    batch_patch = mpatches.Patch(color='steelblue', alpha=0.7, label='Steady-state data')
    bm_line = plt.Line2D([0], [0], color='green', linewidth=2.5, label='Batch mean Ȳⱼ')
    ax.legend(handles=[warm_patch, batch_patch, bm_line], loc='upper right', fontsize=9)
    ax.set_ylabel("Queue Length")
    ax.set_title(f"Single Long Run → Warm-up Deleted → Divided into k={n_batches} Batches")

    # Bottom: batch means + CI
    ax2 = axes[1]
    x = np.arange(1, n_batches + 1)
    colors = ['steelblue' if ci_lower <= m <= ci_upper else 'tomato' for m in batch_means_arr]
    ax2.bar(x, batch_means_arr, color=colors, alpha=0.75, edgecolor='white', label='Batch mean Ȳⱼ')
    ax2.axhline(grand_mean, color='navy', linewidth=2.5, label=f'Grand mean Ȳ = {grand_mean:.2f}')
    ax2.axhline(ci_upper, color='green', linestyle='--', linewidth=1.5,
                label=f'{confidence*100:.0f}% CI [{ci_lower:.2f}, {ci_upper:.2f}]')
    ax2.axhline(ci_lower, color='green', linestyle='--', linewidth=1.5)
    ax2.fill_between([0, n_batches + 1], ci_lower, ci_upper, alpha=0.08, color='green')
    ax2.set_xlabel("Batch number (j)")
    ax2.set_ylabel("Batch mean Ȳⱼ")
    ax2.set_title(f"Batch Means + {confidence*100:.0f}% CI  |  Lag-1 autocorrelation r₁ = {r1:.3f}")
    ax2.set_xticks(x)
    ax2.legend(fontsize=9)

    formula = f"Ȳ ± t(α/2, k-1) × S/√k\n= {grand_mean:.2f} ± {t_crit:.2f} × {S:.2f}/√{n_batches}\n= {grand_mean:.2f} ± {half_width:.2f}"
    ax2.text(0.97, 0.05, formula, transform=ax2.transAxes, fontsize=8.5,
             ha='right', va='bottom', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/batch_means.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("  [Saved] batch_means.png\n")
    return grand_mean, ci_lower, ci_upper


# ═══════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════
def comparison_table():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis('off')
    fig.suptitle("Comparison: Non-Terminating Simulation Output Analysis Methods", fontweight='bold', fontsize=13)

    headers = ["", "Welch Method", "Replication-Deletion", "Batch Means"]
    rows = [
        ["Runs required",     "Multiple (R ≥ 5)",   "Multiple (R ≥ 5)",    "Single long run"],
        ["How warm-up found", "Smooth ensemble avg", "Externally decided",  "Externally decided"],
        ["Key parameter",     "Window size m",       "Delete fraction d",   "Batch count k"],
        ["Output",            "Estimate of l*",      "CI via t-test on Ȳᵢ", "CI via t-test on Ȳⱼ"],
        ["Best for",          "Finding warm-up l*",  "Comparing designs",   "Single system, long run"],
        ["Main risk",         "Wrong window m",      "Deleting too little", "Batch autocorrelation"],
        ["Formula",           "Smooth W(t) = MA(Ȳ,m)", "Ȳ ± t·S/√R",       "Ȳ ± t·S_batch/√k"],
    ]

    col_widths = [0.2, 0.265, 0.265, 0.265]
    col_colors = [['#DDDDDD'] + ['#B3D1F5', '#B3EAD1', '#F5D9B3']] + \
                 [['#F0F0F0'] + ['#EAF3FC', '#EAF8F1', '#FEF5E7']] * len(rows)

    table = ax.table(
        cellText=[headers] + rows,
        loc='center',
        cellLoc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.1)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        if r == 0:
            cell.set_facecolor(['#CCCCCC', '#2E6DA4', '#1F7A4E', '#B05D0B'][c])
            cell.set_text_props(color='white' if c > 0 else 'black', fontweight='bold')
        elif c == 0:
            cell.set_facecolor('#EBEBEB')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor([None, '#EAF3FC', '#EAF8F1', '#FEF5E7'][c])

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/comparison_table.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("  [Saved] comparison_table.png\n")


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  IT2032 — Non-Terminating Simulation Output Analysis")
    print("  Scenario: Call-center queue length simulation")
    print("="*60 + "\n")

    l_star, w_mean   = welch_method(n_reps=10, n_obs=200, window=10)
    rd_mean, rd_lo, rd_hi = replication_deletion(n_reps=15, n_obs=200, delete_frac=0.20)
    bm_mean, bm_lo, bm_hi = batch_means(n_obs=1000, warmup_frac=0.15, n_batches=20)

    comparison_table()

    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  Welch Method         → Steady-state mean = {w_mean:.4f}  (warm-up l* = {l_star})")
    print(f"  Replication-Deletion → Mean = {rd_mean:.4f},  95% CI = [{rd_lo:.4f}, {rd_hi:.4f}]")
    print(f"  Batch Means          → Mean = {bm_mean:.4f},  95% CI = [{bm_lo:.4f}, {bm_hi:.4f}]")
    print("\n  All charts saved to /mnt/user-data/outputs/")
