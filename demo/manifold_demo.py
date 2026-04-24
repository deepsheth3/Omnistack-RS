"""
OmniStack-RS — Phase 0b: Grassmannian Manifold Persona Demo

Validates the core compression thesis: manifold pruning (128 → 8 dims)
doesn't destroy persona information. Five user taste archetypes remain
perfectly separable in the compressed Grassmannian subspace.

Run from the project root:
    python demo/manifold_demo.py

Expected output:
    - ARI > 0.90 (Adjusted Rand Index — cluster purity vs. ground-truth persona)
    - Explained variance breakdown per principal component
    - t-SNE visualization saved to demo/manifold_clusters.png
    - Compression statistics (VRAM reduction factor from manifold alone)
"""

from __future__ import annotations

import os
import sys
import warnings

# Allow running from project root or from demo/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apple Accelerate BLAS raises spurious IEEE 754 signals during DGEMM on
# denormal inputs; the results are always finite. Filter at import-time.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — saves to file without needing a display
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE

from data.synthetic.viewing_history import (
    PERSONAS,
    generate_user_history,
    history_to_embedding,
)
from omnistack_rs.manifold.grassmannian import GrassmannianProjector

# ── Hyperparameters ────────────────────────────────────────────────────────
USERS_PER_PERSONA = 100   # 100 × 5 = 500 total users
EMBEDDING_DIM = 128       # ambient dimension D
MANIFOLD_RANK = 8         # Grassmannian rank k (demo; production uses 32)
N_MOVIES = 500            # viewing events per user
NOISE_STD = 0.15          # genre preference noise (makes users non-identical)
HALF_LIFE_DAYS = 30.0     # temporal decay half-life for history_to_embedding
ARI_THRESHOLD = 0.90      # pass/fail threshold

# ── Visual identity ────────────────────────────────────────────────────────
PERSONA_STYLES = {
    "80s_horror_fan":     {"color": "#e74c3c", "marker": "o"},
    "korean_romcom_fan":  {"color": "#e91e8c", "marker": "s"},
    "prestige_drama":     {"color": "#3498db", "marker": "^"},
    "action_blockbuster": {"color": "#f39c12", "marker": "D"},
    "arthouse_cinephile": {"color": "#27ae60", "marker": "P"},
}
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"


# ── Step 1: Dataset generation ────────────────────────────────────────────

def generate_dataset() -> tuple[np.ndarray, list[str]]:
    persona_names = list(PERSONAS.keys())
    total = USERS_PER_PERSONA * len(persona_names)
    print(f"[1/4] Generating {total} synthetic users ({N_MOVIES} movies each, "
          f"{HALF_LIFE_DAYS}-day temporal decay)...")

    embeddings: list[np.ndarray] = []
    labels: list[str] = []

    for p_idx, persona in enumerate(persona_names):
        for u_idx in range(USERS_PER_PERSONA):
            seed = p_idx * 10_000 + u_idx
            history = generate_user_history(
                persona=persona,
                n_movies=N_MOVIES,
                noise_std=NOISE_STD,
                seed=seed,
            )
            emb = history_to_embedding(
                history,
                dim=EMBEDDING_DIM,
                half_life_days=HALF_LIFE_DAYS,
                projection_seed=0,
            )
            embeddings.append(emb)
            labels.append(persona)

    X = np.stack(embeddings)  # (500, 128)
    print(f"    ✓ Dataset shape: {X.shape}, dtype={X.dtype}")
    return X, labels


# ── Step 2: Grassmannian projection ───────────────────────────────────────

def run_manifold_analysis(X: np.ndarray, labels: list[str]) -> dict:
    print(f"\n[2/4] Fitting GrassmannianProjector(D={EMBEDDING_DIM}, k={MANIFOLD_RANK})...")

    projector = GrassmannianProjector(ambient_dim=EMBEDDING_DIM, rank=MANIFOLD_RANK)
    fit_result = projector.fit(X)

    # Print the explained variance breakdown
    fit_result.print_summary(label=f"(k={MANIFOLD_RANK})")

    # Project to k-dim Grassmannian coordinates
    Z = projector.project(X)   # (500, 8)
    recon_error = projector.reconstruction_error(X)
    print(f"\n    ✓ Reconstruction error: {recon_error:.4f} (relative Frobenius)")

    # Find rank needed for 95% variance
    rank_95 = projector.find_rank_for_variance(X, target_variance=0.95)
    rank_99 = projector.find_rank_for_variance(X, target_variance=0.99)
    print(f"    ✓ Rank for 95% variance: {rank_95}  →  {EMBEDDING_DIM}→{rank_95} "
          f"= {EMBEDDING_DIM / rank_95:.1f}× manifold compression")
    print(f"    ✓ Rank for 99% variance: {rank_99}  →  {EMBEDDING_DIM}→{rank_99} "
          f"= {EMBEDDING_DIM / rank_99:.1f}× manifold compression")

    return {
        "projector": projector,
        "Z": Z,
        "fit_result": fit_result,
        "recon_error": recon_error,
        "rank_95": rank_95,
        "rank_99": rank_99,
    }


# ── Step 3: Clustering and scoring ────────────────────────────────────────

def cluster_and_score(Z: np.ndarray, labels: list[str]) -> dict:
    n_clusters = len(PERSONAS)
    print(f"\n[3/4] KMeans(k={n_clusters}) in {MANIFOLD_RANK}-dim Grassmannian space...")

    # Run KMeans multiple times to avoid bad local optima
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    kmeans.fit(Z)

    ari = adjusted_rand_score(labels, kmeans.labels_)
    status = "✅ PASS" if ari >= ARI_THRESHOLD else f"❌ BELOW TARGET ({ARI_THRESHOLD:.2f})"
    print(f"    Adjusted Rand Index (ARI): {ari:.4f}  {status}")

    # Per-persona purity breakdown
    label_arr = np.array(labels)
    print("\n    Per-cluster composition:")
    for cluster_id in range(n_clusters):
        mask = kmeans.labels_ == cluster_id
        cluster_labels = label_arr[mask]
        dominant = max(set(cluster_labels), key=list(cluster_labels).count)
        purity = (cluster_labels == dominant).mean()
        print(f"      Cluster {cluster_id}: {dominant[:22]:<22}  purity={purity:.2f}  n={mask.sum()}")

    return {"kmeans": kmeans, "ari": ari}


# ── Step 4: Visualization ─────────────────────────────────────────────────

def save_visualization(
    Z: np.ndarray,
    labels: list[str],
    ari: float,
    fit_result,
    rank_95: int,
    output_path: str,
) -> None:
    print(f"\n[4/4] Generating t-SNE visualization → {output_path}")

    # t-SNE on 8-dim Grassmannian coordinates
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1200, learning_rate="auto")
    Z_2d = tsne.fit_transform(Z)

    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)

    # ── Left: t-SNE scatter ────────────────────────────────────────────────
    ax_scatter = fig.add_axes([0.05, 0.15, 0.55, 0.75], facecolor=PANEL_BG)

    label_arr = np.array(labels)
    for persona, style in PERSONA_STYLES.items():
        mask = label_arr == persona
        ax_scatter.scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            c=style["color"],
            marker=style["marker"],
            alpha=0.75,
            s=35,
            edgecolors="none",
            label=persona.replace("_", " ").title(),
            zorder=3,
        )

    ax_scatter.set_title(
        f"Grassmannian Persona Manifold  ·  ARI = {ari:.4f}",
        color="white", fontsize=13, fontweight="bold", pad=10,
    )
    ax_scatter.set_xlabel("t-SNE dimension 1", color="#8b949e", fontsize=10)
    ax_scatter.set_ylabel("t-SNE dimension 2", color="#8b949e", fontsize=10)
    ax_scatter.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor("#30363d")
    ax_scatter.grid(True, color="#21262d", linewidth=0.5, zorder=0)

    legend = ax_scatter.legend(
        framealpha=0.3, facecolor="#161b22", edgecolor="#30363d",
        labelcolor="white", fontsize=9, markerscale=1.5,
        loc="upper right",
    )

    # ── Right: Explained variance bar chart ────────────────────────────────
    ax_var = fig.add_axes([0.66, 0.55, 0.30, 0.35], facecolor=PANEL_BG)

    evr = fit_result.explained_variance_ratio
    cumevr = fit_result.cumulative_variance_ratio
    x_pos = np.arange(len(evr))

    bars = ax_var.bar(x_pos, evr, color="#58a6ff", alpha=0.8, width=0.7, zorder=3)
    ax_var.plot(x_pos, cumevr, "o-", color="#f78166", linewidth=1.5, markersize=4,
                zorder=4, label="Cumulative")
    ax_var.axhline(cumevr[-1], color="#f78166", linestyle="--", linewidth=0.8, alpha=0.5)

    ax_var.set_title(f"Variance by Component\n(k={MANIFOLD_RANK})", color="white",
                     fontsize=10, fontweight="bold")
    ax_var.set_xlabel("Principal Component", color="#8b949e", fontsize=8)
    ax_var.set_ylabel("Explained Variance", color="#8b949e", fontsize=8)
    ax_var.tick_params(colors="#8b949e", labelsize=7)
    ax_var.set_xticks(x_pos)
    ax_var.set_xticklabels([str(i + 1) for i in x_pos], fontsize=7)
    for spine in ax_var.spines.values():
        spine.set_edgecolor("#30363d")
    ax_var.grid(True, color="#21262d", linewidth=0.5, axis="y", zorder=0)
    ax_var.legend(framealpha=0.2, facecolor="#161b22", labelcolor="white", fontsize=7)

    # Annotate cumulative at k
    ax_var.annotate(
        f"{cumevr[-1]:.1%} total",
        xy=(x_pos[-1], cumevr[-1]),
        xytext=(x_pos[-1] - 1.5, cumevr[-1] - 0.08),
        color="#f78166", fontsize=7,
        arrowprops=dict(arrowstyle="->", color="#f78166", lw=0.8),
    )

    # ── Bottom: Stats panel ────────────────────────────────────────────────
    ax_stats = fig.add_axes([0.66, 0.10, 0.30, 0.38], facecolor=PANEL_BG)
    ax_stats.axis("off")

    stats = [
        ("Architecture", "OmniStack-RS"),
        ("Stage", "4 / 6 — Manifold Pruning"),
        ("", ""),
        ("Users", f"{len(Z):,}  (5 personas × {USERS_PER_PERSONA})"),
        ("Movies/user", f"{N_MOVIES:,}"),
        ("Temporal decay", f"{HALF_LIFE_DAYS:.0f}-day half-life"),
        ("", ""),
        ("Ambient dim D", f"{EMBEDDING_DIM}"),
        ("Manifold rank k", f"{MANIFOLD_RANK}"),
        ("Variance captured", f"{cumevr[-1]:.1%}"),
        ("Rank for 95% var", f"{rank_95}  →  {EMBEDDING_DIM/rank_95:.1f}× compression"),
        ("", ""),
        ("ARI", f"{ari:.4f}  {'[PASS]' if ari >= ARI_THRESHOLD else '[FAIL]'}"),
        ("VRAM reduction", f"{EMBEDDING_DIM // MANIFOLD_RANK}× (manifold alone)"),
        ("Total VRAM target", "12.8× (manifold + INT4+QJL)"),
    ]

    y = 0.95
    for key, val in stats:
        if key == "":
            y -= 0.035
            continue
        color_key = "#8b949e" if key != "ARI" else "#f78166"
        color_val = "white" if key not in ("VRAM reduction", "Total VRAM target") else "#58a6ff"
        ax_stats.text(0.0, y, f"{key}:", color=color_key, fontsize=8,
                      transform=ax_stats.transAxes, va="top")
        ax_stats.text(0.50, y, val, color=color_val, fontsize=8,
                      transform=ax_stats.transAxes, va="top", fontweight="bold")
        y -= 0.065

    # ── Overall title ─────────────────────────────────────────────────────
    fig.text(
        0.5, 0.96,
        "OmniStack-RS  ·  The Master & The Shadow  ·  Stage 4: Manifold Pruning",
        ha="center", color="#c9d1d9", fontsize=11, fontweight="bold",
    )
    fig.text(
        0.5, 0.92,
        f"{EMBEDDING_DIM}-dim embeddings → {MANIFOLD_RANK}-dim Grassmannian  ·  "
        f"500 users × 5 personas  ·  ARI = {ari:.4f}",
        ha="center", color="#8b949e", fontsize=9,
    )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    ✓ Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("OmniStack-RS — Phase 0b: Grassmannian Manifold Demo")
    print("=" * 65)

    X, labels = generate_dataset()
    analysis = run_manifold_analysis(X, labels)
    clustering = cluster_and_score(analysis["Z"], labels)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "manifold_clusters.png"
    )
    save_visualization(
        Z=analysis["Z"],
        labels=labels,
        ari=clustering["ari"],
        fit_result=analysis["fit_result"],
        rank_95=analysis["rank_95"],
        output_path=output_path,
    )

    print("\n" + "=" * 65)
    print("PHASE 0b RESULTS")
    print("=" * 65)
    evr_total = analysis["fit_result"].cumulative_variance_ratio[-1]
    print(f"  Embedding dim:         {EMBEDDING_DIM}")
    print(f"  Manifold rank:         {MANIFOLD_RANK}")
    print(f"  Variance captured:     {evr_total:.1%}")
    print(f"  Reconstruction error:  {analysis['recon_error']:.4f} (relative Frobenius)")
    print(f"  Rank for 95% var:      {analysis['rank_95']}  →  "
          f"{EMBEDDING_DIM / analysis['rank_95']:.1f}× manifold compression potential")
    print(f"  ARI:                   {clustering['ari']:.4f}  "
          f"(threshold: {ARI_THRESHOLD})")
    print(f"  VRAM reduction (manifold only): {EMBEDDING_DIM // MANIFOLD_RANK}×")
    print(f"  VRAM reduction (target, manifold + INT4+QJL): 12.8×")
    print("=" * 65)

    if clustering["ari"] >= ARI_THRESHOLD:
        print("✅  Phase 0b PASSED")
        print("    The Grassmannian manifold correctly separates 5 user personas.")
        print("    Stage 4 compression thesis is validated.")
    else:
        print(f"❌  Phase 0b needs tuning — ARI {clustering['ari']:.3f} < {ARI_THRESHOLD}")
        print("    Try: increase N_MOVIES, reduce NOISE_STD, or increase MANIFOLD_RANK.")


if __name__ == "__main__":
    main()
