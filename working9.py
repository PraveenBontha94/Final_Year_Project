# ==============================
# FINAL SYSTEM (PHASE 1 → 5)
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

# Normalize rating: extract float from strings like "4.9stars"
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
sim = cosine_similarity(emb)

for i in range(len(skills_df)):
    for j in range(i + 1, len(skills_df)):
        if sim[i][j] > 0.55:
            s1 = skills_df.iloc[i]["skill"]
            s2 = skills_df.iloc[j]["skill"]
            w = (1 - sim[i][j]) * (2 - (importance[s1] + importance[s2]))
            if len(s1) <= len(s2):
                G.add_edge(s1, s2, weight=w)
            else:
                G.add_edge(s2, s1, weight=w)

# Core backbone path
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
# COURSE RECOMMENDATION (IMPROVED)
# ==============================

def recommend_courses(skill, top_k=3):
    """
    Returns top_k courses for a given skill.
    Improvements over v8:
      1. Removed domain_keywords filter (was blocking foundation skills).
      2. Encodes title + skills together for richer similarity signal.
      3. Lowered threshold to 0.3.
      4. Blends cosine similarity (70%) with course rating (30%).
      5. Deduplicates near-duplicate titles (specialization + sub-course).
    """
    skill_vec = model.encode([skill])[0]
    results = []

    for _, row in courses_df.iterrows():
        if not row["Skills"]:
            continue

        # Must contain the skill as a substring match (at least one direction)
        if not any(skill in s or s in skill for s in row["Skills"]):
            continue

        # Filter clearly unrelated domains
        skill_str = " ".join(row["Skills"])
        if any(x in skill_str for x in ["fmri", "biology", "neuroscience"]):
            continue

        # FIX 2: encode title + skills together
        title = str(row.get("Title", ""))
        text = title + " " + skill_str
        vec = model.encode([text])[0]

        cosine_score = float(cosine_similarity([skill_vec], [vec])[0][0])

        # FIX 3: lowered threshold
        if cosine_score < 0.3:
            continue

        # FIX 1: blend with rating
        rating = row["RatingFloat"]
        blended_score = 0.7 * cosine_score + 0.3 * (rating / 5.0)

        site = str(row.get("Site", "")).strip()
        results.append((title, blended_score, cosine_score, rating, site))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    # FIX: deduplicate near-duplicate titles
    deduped = []
    seen_titles = set()
    for title, blended, cosine_s, rating, site in results:
        if title in seen_titles:
            continue
        # Skip if current title is a substring of an already-added title or vice versa
        if any(title in prev or prev in title for prev in seen_titles):
            continue
        deduped.append((title, blended, cosine_s, rating, site))
        seen_titles.add(title)
        if len(deduped) == top_k:
            break

    return deduped

# ==============================
# CORE RECOMMENDATION ENGINE
# ==============================

def run_recommendation(known, goal):
    """
    Given a list of known skills and a goal skill string,
    returns recommended skills, learning path, and courses.
    """

    # Validate goal exists in graph (fuzzy match if needed)
    all_skill_names = list(skills_df["skill"])
    if goal not in G.nodes:
        # Try fuzzy: find closest skill name containing goal keyword
        candidates = [s for s in G.nodes if goal in s or s in goal]
        if candidates:
            goal = candidates[0]
            print(f"[Info] Goal matched to: '{goal}'")
        else:
            print(f"[Warning] Goal '{goal}' not found in graph. Showing global recommendations.")
            goal = None

    # Encode user profile
    user_vec = model.encode([" ".join(known)])[0]
    goal_vec = model.encode([goal if goal else "machine learning"])[0]

    scores = cosine_similarity([user_vec], gnn_emb)[0]

    # Feature: centrality
    centrality = nx.degree_centrality(G)

    # Feature: path distance to goal
    path_distance = {}
    for skill in skills_df["skill"]:
        if goal:
            try:
                path_distance[skill] = nx.shortest_path_length(G, source=skill, target=goal)
            except:
                path_distance[skill] = 100
        else:
            path_distance[skill] = 50

    # Ranking
    combined = []
    for i, skill in enumerate(skills_df["skill"]):
        sim_user = scores[i]
        sim_goal = float(cosine_similarity([gnn_emb[i]], [goal_vec])[0][0])
        final_score = (
            0.3 * sim_user +
            0.4 * sim_goal +
            0.1 * importance.get(skill, 0) +
            0.1 * centrality.get(skill, 0) +
            0.1 * (1 / (1 + path_distance.get(skill, 100)))
        )
        combined.append((skill, final_score))

    combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Filtering
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

    rec = []
    seen = set()
    for s, _ in combined:
        if s not in seen and valid(s):
            rec.append(s)
            seen.add(s)
        if len(rec) == 5:
            break

    # Learning path
    path = None
    if goal:
        # Try to find a valid source node from known skills
        source = None
        for k in known:
            if k in G.nodes:
                source = k
                break

        if source and goal in G.nodes:
            try:
                path = nx.shortest_path(G, source=source, target=goal, weight="weight")
            except nx.NetworkXNoPath:
                print(f"[Warning] No path found from '{source}' to '{goal}'.")

    return rec, path, goal

# ==============================
# PHASE 5: INTERACTIVE USER LOOP
# ==============================

def print_separator():
    print("\n" + "=" * 55)

def get_user_input():
    print_separator()
    print("  SKILL RECOMMENDATION SYSTEM")
    print_separator()

    # Known skills
    raw = input("\nEnter your current skills (comma-separated):\n> ").strip()
    known = [normalize(s) for s in raw.split(",") if s.strip()]
    if not known:
        known = ["python"]
        print(f"[Default] Using: {known}")

    # Goal
    goal = normalize(input("\nEnter your learning goal (e.g. deep learning):\n> ").strip())
    if not goal:
        goal = "deep learning"
        print(f"[Default] Using: '{goal}'")

    return known, goal

def display_results(known, goal, rec, path):
    print_separator()
    print("  RESULTS")
    print_separator()

    print(f"\nYour Skills : {', '.join(known)}")
    print(f"Your Goal   : {goal}")

    print("\n── Recommended Next Skills ──")
    if rec:
        for i, s in enumerate(rec, 1):
            print(f"  {i}. {s}")
    else:
        print("  No recommendations found.")

    print("\n── Learning Path ──")
    if path:
        print("  " + " → ".join(path))
        print("\n  Step-by-step:")
        for i in range(len(path) - 1):
            print(f"  {i+1}. {path[i]}  →  {path[i+1]}")
    else:
        print("  Path could not be computed.")

    print("\n── Recommended Courses per Step ──")
    steps = path if path else rec[:4]
    for skill in steps:
        print(f"\n  [{skill.upper()}]")
        courses = recommend_courses(skill)
        if not courses:
            print("    No courses found.")
        else:
            for title, blended, cosine_s, rating, site in courses:
                stars = f"{rating:.1f}★" if rating > 0 else "N/A"
                platform = f"  |  {site}" if site and site != "nan" else ""
                print(f"    • {title}")
                print(f"      Score: {blended:.2f}  (similarity: {cosine_s:.2f}, rating: {stars}){platform}")

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

        print_separator()
        again = input("\nRun again for a different skill/goal? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!")
            break