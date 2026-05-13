# ==============================
# FINAL SYSTEM (PHASE 1 → 5 + EXPLAINABILITY)
# ==============================

import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ==============================
# LOAD DATA
# ==============================

skills_df = pd.read_csv("dataset/skills.csv")
courses_df = pd.read_csv("dataset/Online_Courses.csv")

def normalize(x):
    return str(x).lower().strip()

skills_df["skill"] = skills_df["skill"].apply(normalize)
skills_df = skills_df.drop_duplicates(subset=["skill"]).reset_index(drop=True)

skills_df["description"] = skills_df["skill"].apply(
    lambda s: f"{s} is related to machine learning, artificial intelligence, and data science"
)

courses_df["Skills"] = courses_df["Skills"].fillna("").apply(normalize)
courses_df["Skills"] = courses_df["Skills"].apply(
    lambda x: [s.strip() for s in x.split(",")] if x else []
)

def parse_rating(r):
    try:
        return float(str(r).replace("stars", "").strip())
    except:
        return 0.0

courses_df["RatingFloat"] = courses_df["Rating"].apply(parse_rating)

# ==============================
# SKILL IMPORTANCE
# ==============================

all_skills = []
for s in courses_df["Skills"]:
    all_skills.extend(s)

freq = Counter(all_skills)
max_freq = max(freq.values()) if freq else 1

importance = {s: freq.get(s, 1) / max_freq for s in skills_df["skill"]}

# ==============================
# SBERT EMBEDDINGS
# ==============================

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding skills...")
emb = model.encode(skills_df["description"].tolist())

# ==============================
# GRAPH BUILDING
# ==============================

print("Building graph...")
G = nx.DiGraph()
sim_matrix = cosine_similarity(emb)

for i in range(len(skills_df)):
    for j in range(i + 1, len(skills_df)):
        if sim_matrix[i][j] > 0.55:
            s1 = skills_df.iloc[i]["skill"]
            s2 = skills_df.iloc[j]["skill"]
            w = (1 - sim_matrix[i][j]) * (2 - (importance[s1] + importance[s2]))
            if len(s1) <= len(s2):
                G.add_edge(s1, s2, weight=w)
            else:
                G.add_edge(s2, s1, weight=w)

core = [
    ("python", "data preprocessing"),
    ("data preprocessing", "machine learning"),
    ("machine learning", "deep learning")
]
for u, v in core:
    G.add_edge(u, v, weight=0.01)

print("Graph Stats:", G.number_of_nodes(), G.number_of_edges())

# ==============================
# GNN PREP
# ==============================

skill_to_idx = {s: i for i, s in enumerate(skills_df["skill"])}

edges = [
    [skill_to_idx[u], skill_to_idx[v]]
    for u, v in G.edges()
    if u in skill_to_idx and v in skill_to_idx
]

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
x = torch.tensor(emb, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# ==============================
# GNN MODEL
# ==============================

class GNN(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = SAGEConv(dim, dim)
        self.conv2 = SAGEConv(dim, dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

gnn = GNN(x.shape[1])
opt = torch.optim.Adam(gnn.parameters(), lr=0.01)

# ==============================
# TRAIN GNN
# ==============================

print("Training GNN...")
for epoch in range(50):
    opt.zero_grad()
    out = gnn(data.x, data.edge_index)
    loss = ((out - data.x) ** 2).mean()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}  Loss: {loss.item():.6f}")

gnn_emb = gnn(data.x, data.edge_index).detach().numpy()

# ==============================
# EXPLAINABILITY HELPERS
# ==============================

def explain_skill(skill, goal, sim_goal, imp, cent, path_dist, known):
    """
    Returns a list of human-readable reason strings for why this skill was recommended.
    Uses thresholds on each signal to pick the strongest reason(s).
    """
    reasons = []

    # Reason 1: proximity to goal
    if path_dist <= 1:
        reasons.append(f"Only 1 step away from '{goal}' on the skill graph")
    elif path_dist <= 2:
        reasons.append(f"Just {path_dist} steps away from your goal '{goal}'")

    # Reason 2: semantic similarity to goal
    if sim_goal >= 0.80:
        reasons.append(f"Very closely related to '{goal}' (semantic similarity: {sim_goal:.2f})")
    elif sim_goal >= 0.65:
        reasons.append(f"Semantically related to '{goal}'")

    # Reason 3: importance / frequency in courses
    if imp >= 0.6:
        reasons.append("Appears in a large number of courses — a widely required skill")
    elif imp >= 0.3:
        reasons.append("Commonly taught alongside your goal topic")

    # Reason 4: centrality — hub skill
    if cent >= 0.15:
        reasons.append("A hub skill — connects many other skills in the graph")
    elif cent >= 0.08:
        reasons.append("Well-connected skill — bridges multiple topic areas")

    # Reason 5: builds on what they know
    for k in known:
        # skip any known skills or candidate skill not present in the graph
        if not G.has_node(k) or not G.has_node(skill):
            continue
        # check directed relations safely
        if k in G.predecessors(skill) or skill in G.successors(k):
            reasons.append(f"Directly builds on your existing skill: '{k}'")
            break

    # Reason 6: neighbours toward goal
    if goal in G.nodes:
        goal_neighbors = set(G.predecessors(goal))
        if skill in goal_neighbors:
            reasons.append(f"Listed as a direct prerequisite for '{goal}'")

    # Fallback if no strong signals fired
    if not reasons:
        reasons.append(f"Ranked highly by the GNN model for your profile and goal")

    return reasons


def explain_path_step(u, v, known):
    """
    Returns a human-readable reason why edge u → v appears in the learning path.
    """
    edge_data = G.get_edge_data(u, v, default={})
    w = edge_data.get("weight", 1.0)

    reasons = []

    # Is this a core backbone edge?
    core_edges = {
        ("python", "data preprocessing"),
        ("data preprocessing", "machine learning"),
        ("machine learning", "deep learning")
    }
    if (u, v) in core_edges:
        reasons.append(f"Core prerequisite — '{u}' is a standard foundation for '{v}'")

    # Frequency-based co-occurrence
    u_courses = set(
        i for i, row in courses_df.iterrows()
        if any(u in s or s in u for s in row["Skills"])
    )
    v_courses = set(
        i for i, row in courses_df.iterrows()
        if any(v in s or s in v for s in row["Skills"])
    )
    overlap = len(u_courses & v_courses)
    if overlap >= 5:
        reasons.append(f"Co-occur in {overlap} courses — frequently taught together")
    elif overlap >= 2:
        reasons.append(f"Appear together in {overlap} courses")

    # Edge weight (lower = stronger / more direct link)
    if w <= 0.05:
        reasons.append("Very strong direct connection in the skill graph")
    elif w <= 0.3:
        reasons.append("Strong connection in the skill graph")

    # Semantic proximity
    if u in skill_to_idx and v in skill_to_idx:
        ui, vi = skill_to_idx[u], skill_to_idx[v]
        sem_sim = float(cosine_similarity([gnn_emb[ui]], [gnn_emb[vi]])[0][0])
        if sem_sim >= 0.85:
            reasons.append(f"Semantically very close (GNN similarity: {sem_sim:.2f})")

    if not reasons:
        reasons.append(f"Optimal next step on the shortest weighted path")

    return reasons


def explain_course(skill, title, cosine_s, rating, site, matched_skills):
    """
    Returns a human-readable reason why this course was recommended for this skill.
    """
    reasons = []

    # Direct skill match
    direct = [s for s in matched_skills if skill in s or s in skill]
    if direct:
        reasons.append(f"Directly teaches: {', '.join(direct[:3])}")

    # Content similarity
    if cosine_s >= 0.70:
        reasons.append(f"Strong content match for '{skill}' (similarity: {cosine_s:.2f})")
    elif cosine_s >= 0.50:
        reasons.append(f"Good content match for '{skill}'")
    else:
        reasons.append(f"Relevant content for '{skill}'")

    # Rating
    if rating >= 4.8:
        reasons.append(f"Exceptional learner rating: {rating:.1f}/5.0")
    elif rating >= 4.5:
        reasons.append(f"Highly rated: {rating:.1f}/5.0")
    elif rating > 0:
        reasons.append(f"Rated {rating:.1f}/5.0")

    # Platform
    if site and site != "nan":
        reasons.append(f"Available on {site}")

    return reasons

# ==============================
# COURSE RECOMMENDATION
# ==============================

def recommend_courses(skill, top_k=3):
    skill_vec = model.encode([skill])[0]
    results = []

    for _, row in courses_df.iterrows():
        if not row["Skills"]:
            continue
        if not any(skill in s or s in skill for s in row["Skills"]):
            continue

        skill_str = " ".join(row["Skills"])
        if any(x in skill_str for x in ["fmri", "biology", "neuroscience"]):
            continue

        title = str(row.get("Title", ""))
        text = title + " " + skill_str
        vec = model.encode([text])[0]

        cosine_score = float(cosine_similarity([skill_vec], [vec])[0][0])
        if cosine_score < 0.3:
            continue

        rating = row["RatingFloat"]
        blended_score = 0.7 * cosine_score + 0.3 * (rating / 5.0)
        site = str(row.get("Site", "")).strip()

        results.append((title, blended_score, cosine_score, rating, site, list(row["Skills"])))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    deduped = []
    seen_titles = set()
    for entry in results:
        title = entry[0]
        if title in seen_titles:
            continue
        if any(title in prev or prev in title for prev in seen_titles):
            continue
        deduped.append(entry)
        seen_titles.add(title)
        if len(deduped) == top_k:
            break

    return deduped

# ==============================
# CORE RECOMMENDATION ENGINE
# ==============================

def run_recommendation(known, goal):
    if goal not in G.nodes:
        candidates = [s for s in G.nodes if goal in s or s in goal]
        if candidates:
            goal = candidates[0]
            print(f"[Info] Goal matched to: '{goal}'")
        else:
            print(f"[Warning] Goal '{goal}' not found in graph.")
            goal = None

    user_vec = model.encode([" ".join(known)])[0]
    goal_vec = model.encode([goal if goal else "machine learning"])[0]
    scores = cosine_similarity([user_vec], gnn_emb)[0]

    centrality = nx.degree_centrality(G)

    path_distance = {}
    for skill in skills_df["skill"]:
        if goal:
            try:
                path_distance[skill] = nx.shortest_path_length(G, source=skill, target=goal)
            except:
                path_distance[skill] = 100
        else:
            path_distance[skill] = 50

    # Build scored list WITH raw signal values for explainability
    combined = []
    for i, skill in enumerate(skills_df["skill"]):
        sim_user = scores[i]
        sim_goal = float(cosine_similarity([gnn_emb[i]], [goal_vec])[0][0])
        imp = importance.get(skill, 0)
        cent = centrality.get(skill, 0)
        pdist = path_distance.get(skill, 100)

        final_score = (
            0.3 * sim_user +
            0.4 * sim_goal +
            0.1 * imp +
            0.1 * cent +
            0.1 * (1 / (1 + pdist))
        )
        combined.append((skill, final_score, sim_goal, imp, cent, pdist))

    combined = sorted(combined, key=lambda x: x[1], reverse=True)

    goal_keywords = ["machine learning", "deep learning", "neural", "data", "model", "ai"]
    hard_bad = ["javascript", "sql", "oracle", "azure", "html", "css", "aws", "cloud"]
    soft_bad = ["basic", "syntax", "beginner", "tools", "functions", "ides", "metadata"]

    def valid(skill):
        if any(b in skill for b in hard_bad): return False
        if any(b in skill for b in soft_bad): return False
        if not any(g in skill for g in goal_keywords): return False
        if goal and skill == goal: return False
        if skill in known: return False
        return True

    rec = []         # list of (skill, reasons)
    seen = set()
    for skill, _, sim_goal, imp, cent, pdist in combined:
        if skill not in seen and valid(skill):
            reasons = explain_skill(skill, goal, sim_goal, imp, cent, pdist, known)
            rec.append((skill, reasons))
            seen.add(skill)
        if len(rec) == 5:
            break

    path = None
    if goal:
        source = next((k for k in known if k in G.nodes), None)
        if source and goal in G.nodes:
            try:
                path = nx.shortest_path(G, source=source, target=goal, weight="weight")
            except nx.NetworkXNoPath:
                print(f"[Warning] No path from '{source}' to '{goal}'.")

    return rec, path, goal

# ==============================
# DISPLAY
# ==============================

def print_sep(char="=", width=58):
    print("\n" + char * width)

def display_results(known, goal, rec, path):
    print_sep()
    print("  RESULTS")
    print_sep()

    print(f"\n  Your skills : {', '.join(known)}")
    print(f"  Your goal   : {goal}")

    # ── Recommended skills with reasons ──
    print("\n" + "─" * 58)
    print("  Recommended next skills")
    print("─" * 58)

    if rec:
        for i, (skill, reasons) in enumerate(rec, 1):
            print(f"\n  {i}. {skill.upper()}")
            for r in reasons:
                print(f"     → {r}")
    else:
        print("  No recommendations found.")

    # ── Learning path with step reasons ──
    print("\n" + "─" * 58)
    print("  Learning path")
    print("─" * 58)

    if path:
        print(f"\n  {' → '.join(path)}\n")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            step_reasons = explain_path_step(u, v, known)
            print(f"  Step {i+1}: {u}  →  {v}")
            for r in step_reasons:
                print(f"     → {r}")
            print()
    else:
        print("  Path could not be computed.")

    # ── Courses with reasons ──
    print("─" * 58)
    print("  Recommended courses per step")
    print("─" * 58)

    steps = path if path else [s for s, _ in rec[:4]]
    for skill in steps:
        print(f"\n  [ {skill.upper()} ]")
        courses = recommend_courses(skill)
        if not courses:
            print("    No courses found.")
        else:
            for title, blended, cosine_s, rating, site, matched_skills in courses:
                course_reasons = explain_course(skill, title, cosine_s, rating, site, matched_skills)
                print(f"\n    • {title}")
                for r in course_reasons:
                    print(f"      → {r}")

    print_sep()

# ==============================
# INTERACTIVE USER LOOP
# ==============================

def get_user_input():
    print_sep()
    print("  SKILL RECOMMENDATION SYSTEM  (with explainability)")
    print_sep()

    raw = input("\nEnter your current skills (comma-separated):\n> ").strip()
    known = [normalize(s) for s in raw.split(",") if s.strip()]
    if not known:
        known = ["python"]
        print(f"[Default] Using: {known}")

    goal = normalize(input("\nEnter your learning goal (e.g. deep learning):\n> ").strip())
    if not goal:
        goal = "deep learning"
        print(f"[Default] Using: '{goal}'")

    return known, goal

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    while True:
        try:
            known, goal = get_user_input()
            rec, path, resolved_goal = run_recommendation(known, goal)
            display_results(known, resolved_goal, rec, path)
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break

        print_sep()
        again = input("Run again for a different skill/goal? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!")
            break