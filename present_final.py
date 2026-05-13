# ==============================
# FINAL SYSTEM (PHASE 1 → 4)
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
# SKILL IMPORTANCE (PHASE 2)
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
# GRAPH BUILDING (PHASE 2)
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
# GNN PREP (PHASE 3)
# ==============================

skill_to_idx = {s:i for i,s in enumerate(skills_df["skill"])}

edges = [[skill_to_idx[u], skill_to_idx[v]] for u,v in G.edges()
         if u in skill_to_idx and v in skill_to_idx]

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

user_vec = model.encode([" ".join(known)])[0]
goal_vec = model.encode([goal])[0]

scores = cosine_similarity([user_vec], gnn_emb)[0]

# ==============================
# FEATURE EXTRACTION (PHASE 4)
# ==============================

centrality = nx.degree_centrality(G)

path_distance = {}
for skill in skills_df["skill"]:
    try:
        dist = nx.shortest_path_length(G, source=skill, target=goal)
        path_distance[skill] = dist
    except:
        path_distance[skill] = 100

# ==============================
# RANKING MODEL (PHASE 4)
# ==============================

combined = []

for i, skill in enumerate(skills_df["skill"]):
    sim_user = scores[i]
    sim_goal = cosine_similarity([gnn_emb[i]], [goal_vec])[0][0]

    importance_score = importance.get(skill, 0)
    centrality_score = centrality.get(skill, 0)
    distance_score = 1 / (1 + path_distance.get(skill, 100))

    final_score = (
        0.3 * sim_user +
        0.4 * sim_goal +
        0.1 * importance_score +
        0.1 * centrality_score +
        0.1 * distance_score
    )

    combined.append((skill, final_score))

combined = sorted(combined, key=lambda x: x[1], reverse=True)

# ==============================
# FILTERING
# ==============================

goal_keywords = ["machine learning","deep learning","neural","data","model","ai"]

hard_bad = ["javascript","sql","oracle","azure","html","css"]
soft_bad = ["basic","syntax","beginner","tools","functions","ides"]

def valid(skill):
    if any(b in skill for b in hard_bad): return False
    if any(b in skill for b in soft_bad): return False
    if not any(g in skill for g in goal_keywords): return False
    return True

rec = []
seen = set()

for s,_ in combined:
    if s not in known and s not in seen and valid(s):
        rec.append(s)
        seen.add(s)
    if len(rec)==5:
        break

# ==============================
# PATH
# ==============================

path = nx.shortest_path(G, source="python", target="deep learning", weight="weight")

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

        text = " ".join(row["Skills"])
        vec = model.encode([text])[0]

        score = cosine_similarity([skill_vec], [vec])[0][0]

        if score > 0.5:
            results.append((row["Title"], score))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    seen_titles = set()
    final = []

    for t,s in results:
        if t not in seen_titles:
            final.append((t,s))
            seen_titles.add(t)
        if len(final)==top_k:
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
    print(f"{path[i]} → {path[i+1]} (GNN + Ranking)")

print("\n📚 Recommended Courses:")

for skill in path:
    print(f"\n🔹 {skill}")
    courses = recommend_courses(skill)

    if not courses:
        print("No courses found")
    else:
        for title, score in courses:
            print(f"- {title} ({round(score,2)})")
