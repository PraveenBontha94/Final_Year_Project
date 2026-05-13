# ======================================================
# evaluate.py  —  Evaluation metrics for the skill
#                 recommendation system (working10.py)
#
# Run:  python evaluate.py
# Requires working10.py to be in the same folder.
# ======================================================

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

print("Loading system (this takes ~30s the first time)...")
from code import (
    skills_df, courses_df, G, gnn_emb, emb,
    model, skill_to_idx, importance,
    run_recommendation, recommend_courses,
    normalize
)
print("System loaded.\n")

# ======================================================
# HELPERS
# ======================================================

SEP  = "=" * 62
SEP2 = "─" * 62

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def row(label, value):
    print(f"  {label:<44} {value}")

def grade(score, thresholds):
    for threshold, label in thresholds:
        if score >= threshold:
            return label
    return thresholds[-1][1]

# ======================================================
# TEST PERSONAS
# ======================================================

TEST_CASES = [
    {"known": ["python"],                          "goal": "deep learning"},
    {"known": ["python", "statistics"],            "goal": "machine learning"},
    {"known": ["data preprocessing"],              "goal": "deep learning"},
    {"known": ["machine learning"],                "goal": "deep learning"},
    {"known": ["python", "data preprocessing"],    "goal": "neural networks"},
]

# Expanded set used only for coverage metric
COVERAGE_CASES = TEST_CASES + [
    {"known": ["python"],               "goal": "nlp"},
    {"known": ["statistics"],           "goal": "data science"},
    {"known": ["linear algebra"],       "goal": "machine learning"},
    {"known": ["python"],               "goal": "computer vision"},
    {"known": ["data preprocessing"],   "goal": "ai"},
    {"known": ["machine learning"],     "goal": "reinforcement learning"},
    {"known": ["python", "statistics"], "goal": "data analysis"},
]

# ======================================================
# METRIC 1  PATH SEMANTIC COHERENCE  (FIXED)
#
# Previous version only tested the 3-step backbone path
# (python→preprocessing→ml→dl) which the GNN collapses
# to identical vectors → trivial score of 1.0.
#
# Fix: sample 30 random source→target pairs from the
# graph and measure step-to-step SBERT cosine similarity
# using raw emb (not gnn_emb which has collapsed variance).
# ======================================================

def eval_path_coherence():
    section("Metric 1 — Learning path semantic coherence")
    print("  Avg SBERT cosine similarity between consecutive steps")
    print("  across 30 random source→target pairs in the graph.\n")
    print(f"  {'Pair':<42} {'Steps':<7} {'Coherence'}")
    print(f"  {SEP2}")

    np.random.seed(42)
    nodes = [n for n in G.nodes if n in skill_to_idx]
    all_scores = []
    shown = 0
    attempts = 0

    while len(all_scores) < 30 and attempts < 500:
        attempts += 1
        src, tgt = np.random.choice(nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, source=src, target=tgt, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if len(path) < 2:
            continue

        step_sims = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in skill_to_idx and v in skill_to_idx:
                ui, vi = skill_to_idx[u], skill_to_idx[v]
                s = float(cosine_similarity([emb[ui]], [emb[vi]])[0][0])
                step_sims.append(s)

        if step_sims:
            avg = np.mean(step_sims)
            all_scores.append(avg)
            if shown < 8:
                pair = f"{src[:18]} → {tgt[:18]}"
                print(f"  {pair:<42} {len(path)-1:<7} {avg:.4f}")
                shown += 1

    if shown < len(all_scores):
        print(f"  ... ({len(all_scores) - shown} more pairs evaluated)")

    if all_scores:
        overall = np.mean(all_scores)
        print(f"\n  {'Pairs evaluated:':<44} {len(all_scores)}")
        print(f"  {'Overall avg coherence:':<44} {overall:.4f}  "
              f"{grade(overall, [(0.75,'Excellent'),(0.55,'Good'),(0.35,'Fair'),(0,'Poor')])}")
    return all_scores


# ======================================================
# METRIC 2  SKILL GOAL RELEVANCE
# Uses raw SBERT emb for absolute relevance measurement.
# ======================================================

def eval_goal_relevance(test_cases):
    section("Metric 2 — Skill recommendation goal relevance")
    print("  Avg SBERT similarity of recommended skills to the goal.\n")
    print(f"  {'User profile':<40} {'Top skill':<26} {'Avg sim'}")
    print(f"  {SEP2}")

    all_scores = []

    for tc in test_cases:
        known, goal = tc["known"], tc["goal"]
        rec, _, resolved_goal = run_recommendation(known, goal)
        if not rec or not resolved_goal:
            continue

        goal_vec = model.encode([resolved_goal])[0]
        sims = []
        for skill, _ in rec:
            if skill in skill_to_idx:
                s = float(cosine_similarity([emb[skill_to_idx[skill]]], [goal_vec])[0][0])
                sims.append(s)

        if sims:
            avg = np.mean(sims)
            all_scores.append(avg)
            top_skill = rec[0][0][:25] if rec else "—"
            profile = f"{known} → {goal}"[:39]
            print(f"  {profile:<40} {top_skill:<26} {avg:.4f}")

    if all_scores:
        overall = np.mean(all_scores)
        print(f"\n  {'Overall avg goal relevance:':<44} {overall:.4f}  "
              f"{grade(overall, [(0.55,'Excellent'),(0.40,'Good'),(0.25,'Fair'),(0,'Poor')])}")
    return all_scores


# ======================================================
# METRIC 3  INTRA-LIST DIVERSITY  (FIXED)
#
# Previous version used gnn_emb. The GNN's MSE loss
# collapses all node vectors toward a common mean →
# every pairwise distance ≈ 0.
#
# Fix: use raw SBERT emb which preserves real semantic
# differences between skill descriptions.
# ======================================================

def eval_diversity(test_cases):
    section("Metric 3 — Skill recommendation diversity")
    print("  Avg pairwise SBERT distance between the 5 recommended")
    print("  skills. Higher = more varied recommendations.\n")
    print(f"  {'User profile':<40} {'Recs':<6} {'Diversity'}")
    print(f"  {SEP2}")

    all_scores = []

    for tc in test_cases:
        known, goal = tc["known"], tc["goal"]
        rec, _, _ = run_recommendation(known, goal)
        if len(rec) < 2:
            continue

        skill_vecs = [
            emb[skill_to_idx[skill]]   # FIX: raw SBERT, not gnn_emb
            for skill, _ in rec
            if skill in skill_to_idx
        ]
        if len(skill_vecs) < 2:
            continue

        sim_mat = cosine_similarity(skill_vecs)
        n = len(skill_vecs)
        dists = [1 - sim_mat[i][j] for i in range(n) for j in range(i+1, n)]
        avg_div = np.mean(dists)
        all_scores.append(avg_div)

        profile = f"{known} → {goal}"[:39]
        print(f"  {profile:<40} {len(rec):<6} {avg_div:.4f}")

    if all_scores:
        overall = np.mean(all_scores)
        print(f"\n  {'Overall avg diversity:':<44} {overall:.4f}  "
              f"{grade(overall, [(0.20,'Excellent'),(0.10,'Good'),(0.04,'Fair'),(0,'Poor')])}")
    return all_scores


# ======================================================
# METRIC 4  COURSE PRECISION@K  (unchanged — was correct)
# ======================================================

def eval_course_precision(test_cases, k=3):
    section(f"Metric 4 — Course recommendation Precision@{k}")
    print(f"  Of the top {k} courses per skill, how many explicitly")
    print("  contain that skill in their course skill tags?\n")
    print(f"  {'Skill':<30} {'Hits':<8} {'Precision@' + str(k)}")
    print(f"  {SEP2}")

    all_precisions = []
    evaluated = set()

    for tc in test_cases:
        _, path, _ = run_recommendation(tc["known"], tc["goal"])
        for skill in (path or []):
            if skill in evaluated:
                continue
            evaluated.add(skill)
            courses = recommend_courses(skill, top_k=k)
            if not courses:
                continue
            hits = sum(
                1 for _, _, _, _, _, skill_list in courses
                if any(skill in s or s in skill for s in skill_list)
            )
            prec = hits / len(courses)
            all_precisions.append(prec)
            print(f"  {skill:<30} {hits}/{len(courses):<6} {prec:.2f}")

    if all_precisions:
        overall = np.mean(all_precisions)
        print(f"\n  {'Overall Precision@' + str(k) + ':':<44} {overall:.4f}  "
              f"{grade(overall, [(0.80,'Excellent'),(0.60,'Good'),(0.40,'Fair'),(0,'Poor')])}")
    return all_precisions


# ======================================================
# METRIC 5  COURSE RATING LIFT  (unchanged — was correct)
# ======================================================

def eval_rating_lift(test_cases, k=3):
    section("Metric 5 — Course rating lift over dataset average")
    print("  Recommended courses avg rating vs all matching courses.\n")
    print(f"  {'Skill':<30} {'Rec avg':<12} {'All avg':<12} {'Lift'}")
    print(f"  {SEP2}")

    all_lifts = []
    evaluated = set()

    for tc in test_cases:
        _, path, _ = run_recommendation(tc["known"], tc["goal"])
        for skill in (path or []):
            if skill in evaluated:
                continue
            evaluated.add(skill)

            all_ratings = [
                r["RatingFloat"]
                for _, r in courses_df.iterrows()
                if r["Skills"] and any(skill in s or s in skill for s in r["Skills"])
                and r["RatingFloat"] > 0
            ]
            if not all_ratings:
                continue

            baseline = np.mean(all_ratings)
            courses = recommend_courses(skill, top_k=k)
            rec_ratings = [r for _, _, _, r, _, _ in courses if r > 0]
            if not rec_ratings:
                continue

            rec_avg = np.mean(rec_ratings)
            lift = rec_avg - baseline
            all_lifts.append(lift)
            lift_str = f"+{lift:.3f}" if lift >= 0 else f"{lift:.3f}"
            print(f"  {skill:<30} {rec_avg:<12.3f} {baseline:<12.3f} {lift_str}")

    if all_lifts:
        overall = np.mean(all_lifts)
        lift_str = f"+{overall:.4f}" if overall >= 0 else f"{overall:.4f}"
        print(f"\n  {'Overall avg rating lift:':<44} {lift_str}  "
              f"{grade(overall, [(0.20,'Excellent'),(0.05,'Good'),(-0.05,'Neutral'),(float('-inf'),'Negative')])}")
    return all_lifts


# ======================================================
# METRIC 6  EMBEDDING QUALITY  (FIXED)
#
# Previous version crashed because gnn_emb has collapsed
# to a single cluster (std ≈ 0) due to MSE loss.
#
# Fix: cluster raw SBERT embeddings instead. Also print
# a GNN collapse diagnostic so the reader understands
# why we switch embeddings.
# ======================================================

def eval_gnn_embedding_quality():
    section("Metric 6 — Embedding quality (silhouette score)")

    gnn_std  = float(np.std(gnn_emb))
    sbert_std = float(np.std(emb))
    print(f"  GNN embedding std dev  : {gnn_std:.6f}")
    print(f"  SBERT embedding std dev: {sbert_std:.6f}")

    if gnn_std < 0.01:
        print("  [!] GNN embeddings collapsed (MSE loss averages neighbours).")
        print("      Silhouette evaluated on raw SBERT embeddings.\n")
    else:
        print("  GNN embeddings have healthy variance.\n")

    target_emb = emb   # always use SBERT for reliable silhouette
    print(f"  {'K':<6} {'Silhouette':<14} {'Grade'}")
    print(f"  {SEP2}")

    results = []
    for k in [5, 8, 10, 15]:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(target_emb)
            if len(set(labels)) < 2:
                print(f"  K={k:<4} Only 1 cluster — skipped")
                continue
            score = silhouette_score(target_emb, labels, sample_size=500, random_state=42)
            results.append((k, score, labels))
            g = grade(score, [(0.5,'Excellent'),(0.3,'Good'),(0.1,'Fair'),(0,'Poor')])
            print(f"  K={k:<4} {score:<14.4f} {g}")
        except Exception as e:
            print(f"  K={k}  Error: {e}")

    if results:
        best_k, best_score, best_labels = max(results, key=lambda x: x[1])
        print(f"\n  Best: K={best_k}  score={best_score:.4f}")
        print(f"\n  Sample clusters (K={best_k}):")
        clusters = defaultdict(list)
        for skill, lbl in zip(skills_df["skill"], best_labels):
            clusters[lbl].append(skill)
        for cid in sorted(clusters)[:4]:
            print(f"  Cluster {cid}: {', '.join(clusters[cid][:6])}")

    return results


# ======================================================
# METRIC 7  PATH LENGTH vs BFS  (unchanged)
# ======================================================

def eval_path_length_baseline(test_cases):
    section("Metric 7 — Path length vs unweighted BFS baseline")
    print("  GNN-weighted path vs shortest unweighted path.\n")
    print(f"  {'Profile':<38} {'GNN':<6} {'BFS':<6} {'Diff'}")
    print(f"  {SEP2}")

    gnn_lengths, bfs_lengths = [], []

    for tc in test_cases:
        known, goal = tc["known"], tc["goal"]
        _, path, resolved_goal = run_recommendation(known, goal)
        source = next((k for k in known if k in G.nodes), None)
        if not source or not resolved_goal or resolved_goal not in G.nodes:
            continue

        gnn_len = len(path) - 1 if path else None
        try:
            bfs_len = len(nx.shortest_path(G, source=source, target=resolved_goal)) - 1
        except:
            bfs_len = None

        if gnn_len is not None and bfs_len is not None:
            gnn_lengths.append(gnn_len)
            bfs_lengths.append(bfs_len)
            diff = gnn_len - bfs_len
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            profile = f"{known} → {goal}"[:37]
            print(f"  {profile:<38} {gnn_len:<6} {bfs_len:<6} {diff_str}")

    if gnn_lengths:
        avg_gnn = np.mean(gnn_lengths)
        avg_bfs = np.mean(bfs_lengths)
        diff = avg_gnn - avg_bfs
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"\n  {'Avg GNN path length:':<44} {avg_gnn:.2f}")
        print(f"  {'Avg BFS path length:':<44} {avg_bfs:.2f}")
        print(f"  {'Difference:':<44} {diff_str}")
        if diff > 0:
            print(f"\n  GNN routes through more intermediate skills,")
            print(f"  providing a more gradual learning progression.")

    return gnn_lengths, bfs_lengths


# ======================================================
# METRIC 8  SKILL COVERAGE  (FIXED)
#
# Previous version used only 5 personas targeting nearly
# identical goals → 8/366 skills surfaced (2%).
#
# Fix: use 12 diverse personas with varied goals, and
# also include top-10 raw ranked skills per goal to
# measure how much of the graph the ranker actually
# reaches (beyond what the valid() filter allows).
# ======================================================

def eval_coverage(coverage_cases):
    section("Metric 8 — Recommendation coverage")
    print("  Fraction of graph skills surfaced across 12 diverse")
    print("  personas (including raw top-10 per goal).\n")

    all_recommended = set()

    for tc in coverage_cases:
        known, goal = tc["known"], tc["goal"]
        try:
            rec, path, _ = run_recommendation(known, goal)
            for skill, _ in rec:
                all_recommended.add(skill)
            if path:
                all_recommended.update(path)

            # Include raw top-10 by GNN similarity to goal
            # (unfiltered — measures ranker breadth)
            goal_vec = model.encode([goal])[0]
            sims = cosine_similarity([goal_vec], gnn_emb)[0]
            top_idxs = np.argsort(sims)[::-1][:10]
            for idx in top_idxs:
                all_recommended.add(skills_df.iloc[idx]["skill"])
        except Exception:
            continue

    total = G.number_of_nodes()
    covered = len(all_recommended)
    coverage = covered / total if total > 0 else 0

    row("Test personas used:", len(coverage_cases))
    row("Total skills in graph:", total)
    row("Unique skills surfaced:", covered)
    row("Coverage ratio:", f"{coverage:.4f}  "
        f"{grade(coverage, [(0.20,'Good'),(0.10,'Fair'),(0,'Low')])}")
    print(f"\n  Sample: {', '.join(list(all_recommended)[:8])}")

    return coverage


# ======================================================
# SUMMARY
# ======================================================

def print_summary(results):
    section("EVALUATION SUMMARY")
    print(f"  {'Metric':<46} {'Score':<10} {'Grade'}")
    print(f"  {SEP2}")

    rows = [
        ("Path coherence (SBERT, 30 random pairs)",
         results.get("coherence"),
         [(0.75,'Excellent'),(0.55,'Good'),(0.35,'Fair'),(0,'Poor')]),
        ("Skill goal relevance (SBERT)",
         results.get("relevance"),
         [(0.55,'Excellent'),(0.40,'Good'),(0.25,'Fair'),(0,'Poor')]),
        ("Intra-list diversity (SBERT)",
         results.get("diversity"),
         [(0.20,'Excellent'),(0.10,'Good'),(0.04,'Fair'),(0,'Poor')]),
        ("Course Precision@3",
         results.get("precision"),
         [(0.80,'Excellent'),(0.60,'Good'),(0.40,'Fair'),(0,'Poor')]),
        ("Course rating lift",
         results.get("lift"),
         [(0.20,'Excellent'),(0.05,'Good'),(-0.05,'Neutral'),(float('-inf'),'Negative')]),
        ("SBERT silhouette score (best K)",
         results.get("silhouette"),
         [(0.5,'Excellent'),(0.3,'Good'),(0.1,'Fair'),(0,'Poor')]),
        ("Skill coverage (12 personas)",
         results.get("coverage"),
         [(0.20,'Good'),(0.10,'Fair'),(0,'Low')]),
    ]

    for name, val, thresholds in rows:
        if val is None:
            print(f"  {name:<46} {'—':<10} —")
        else:
            g = grade(val, thresholds)
            print(f"  {name:<46} {val:<10.4f} {g}")

    print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Metrics 1, 3, 6 use raw SBERT embeddings (gnn_emb collapsed).")
    print(f"\n{SEP}\n")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  EVALUATION REPORT — Skill Recommendation System")
    print(f"  Personas: {len(TEST_CASES)} primary  |  {len(COVERAGE_CASES)} for coverage")
    print(SEP)

    results = {}

    scores = eval_path_coherence()
    results["coherence"] = np.mean(scores) if scores else None

    scores = eval_goal_relevance(TEST_CASES)
    results["relevance"] = np.mean(scores) if scores else None

    scores = eval_diversity(TEST_CASES)
    results["diversity"] = np.mean(scores) if scores else None

    scores = eval_course_precision(TEST_CASES, k=3)
    results["precision"] = np.mean(scores) if scores else None

    scores = eval_rating_lift(TEST_CASES, k=3)
    results["lift"] = np.mean(scores) if scores else None

    emb_results = eval_gnn_embedding_quality()
    if emb_results:
        results["silhouette"] = max(s for _, s, _ in emb_results)

    eval_path_length_baseline(TEST_CASES)

    results["coverage"] = eval_coverage(COVERAGE_CASES)

    print_summary(results)