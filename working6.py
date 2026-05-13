# ==============================
# FINAL PHASE 3 (CLEAN + STABLE)
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

# ==============================
# SKILL IMPORTANCE
# ==============================

all_skills = []
for s in courses_df["Skills"]:
    all_skills.extend(s)

freq = Counter(all_skills)
max_freq = max(freq.values()) if freq else 1

importance = {s: freq.get(s, 1)/max_freq for s in skills_df["skill"]}

# ==============================
# SBERT EMBEDDINGS
# ==============================

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding skills...")
emb = model.encode(skills_df["description"].tolist())

# ==============================
# GRAPH BUILDING (WEIGHTED)
# ==============================

print("Building graph...")

G = nx.DiGraph()
sim = cosine_similarity(emb)

for i in range(len(skills_df)):
    for j in range(i+1, len(skills_df)):
        if sim[i][j] > 0.55:

            s1 = skills_df.iloc[i]["skill"]
            s2 = skills_df.iloc[j]["skill"]

            w = (1 - sim[i][j]) * (2 - (importance[s1] + importance[s2]))

            if len(s1) <= len(s2):
                G.add_edge(s1, s2, weight=w)
            else:
                G.add_edge(s2, s1, weight=w)

# core learning path
core = [
    ("python","data preprocessing"),
    ("data preprocessing","machine learning"),
    ("machine learning","deep learning")
]

for u,v in core:
    G.add_edge(u,v,weight=0.01)

print("Graph Stats:", G.number_of_nodes(), G.number_of_edges())

# ==============================
# GNN PREP
# ==============================

skill_to_idx = {s:i for i,s in enumerate(skills_df["skill"])}

edges = [[skill_to_idx[u], skill_to_idx[v]] for u,v in G.edges()
         if u in skill_to_idx and v in skill_to_idx]

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
x = torch.tensor(emb, dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# ==============================
# GNN MODEL (FIXED DIMENSIONS)
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
    loss = ((out - data.x)**2).mean()
    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        print("Epoch", epoch, "Loss:", loss.item())

gnn_emb = gnn(data.x, data.edge_index).detach().numpy()

# ==============================
# USER INPUT
# ==============================

known = ["python"]
goal = "deep learning"

# ==============================
# RECOMMENDATION (FINAL FIX)
# ==============================

user_vec = model.encode([" ".join(known)])[0]
goal_vec = model.encode([goal])[0]

scores = cosine_similarity([user_vec], gnn_emb)[0]

combined = []

for i, skill in enumerate(skills_df["skill"]):
    su = scores[i]
    sg = cosine_similarity([gnn_emb[i]], [goal_vec])[0][0]
    combined.append((skill, 0.3*su + 0.7*sg))

combined = sorted(combined, key=lambda x: x[1], reverse=True)

# 🔥 GOAL-AWARE FILTER
goal_keywords = [
    "machine learning","deep learning",
    "neural","data","model",
    "ai","analysis","statistics"
]

hard_bad = ["javascript","oracle","sql","azure","html","css"]
soft_bad = ["basic","syntax","introduction","beginner","tools","functions","ides"]

def is_relevant(skill):
    if any(b in skill for b in hard_bad):
        return False
    if any(b in skill for b in soft_bad):
        return False
    if not any(g in skill for g in goal_keywords):
        return False
    return True

def normalize_skill(s):
    return s.replace("learning","").strip()

rec = []
seen = set()
seen_norm = set()

for s,_ in combined:
    base = normalize_skill(s)

    if s not in known and s not in seen:
        if base not in seen_norm and is_relevant(s):
            rec.append(s)
            seen.add(s)
            seen_norm.add(base)

    if len(rec) == 5:
        break

# ==============================
# PATH
# ==============================

try:
    path = nx.shortest_path(G, source="python", target="deep learning", weight="weight")
except:
    path = ["No path"]

# ==============================
# COURSE RECOMMENDATION (FINAL)
# ==============================

def recommend_courses(skill, top_k=3):
    skill_vec = model.encode([skill])[0]
    results = []

    for _, row in courses_df.iterrows():
        if not row["Skills"]:
            continue

        # strict + flexible matching
        if not any(skill in s or s in skill for s in row["Skills"]):
            continue

        course_text = " ".join(row["Skills"])
        course_vec = model.encode([course_text])[0]

        score = cosine_similarity([skill_vec], [course_vec])[0][0]

        # filter weak matches
        if score > 0.5:
            results.append((row["Title"], score))

    # sort
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # remove duplicates
    final = []
    seen = set()

    for title, score in results:
        if title not in seen:
            final.append((title, score))
            seen.add(title)
        if len(final) == top_k:
            break

    return final


# ==============================
# OUTPUT
# ==============================


print("\nFINAL OUTPUT\n")

print("User Skills:", known)
print("Goal:", goal)

print("\nRecommended Skills:")
for s in rec:
    print("-", s)

print("\nLearning Path:")
print(" → ".join(path))

print("\nExplanation:")
for i in range(len(path)-1):
    print(f"{path[i]} → {path[i+1]} (GNN-enhanced)")

# ==============================
# PRINT COURSE RECOMMENDATIONS
# ==============================

print("\n📚 Recommended Courses:")

if "No" not in path[0]:
    for skill in path:
        print(f"\n🔹 Skill: {skill}")

        courses = recommend_courses(skill)

        if not courses:
            print("No courses found")
        else:
            for title, score in courses:
                print(f"- {title} ({round(score,2)})")