# ==============================
# FINAL PHASE 1 (ULTIMATE FINAL FIXED)
# ==============================

import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# STEP 1: LOAD DATA
# ==============================

skills_df = pd.read_csv("dataset/skills.csv")
courses_df = pd.read_csv("dataset/Online_Courses.csv")

def normalize(text):
    return str(text).lower().strip()

skills_df["skill"] = skills_df["skill"].apply(normalize)

# ==============================
# STEP 2: CLEAN + BETTER DESCRIPTIONS
# ==============================

skills_df = skills_df.drop_duplicates(subset=["skill"]).reset_index(drop=True)

skills_df["description"] = skills_df["skill"].apply(
    lambda s: f"{s} is a concept in machine learning, artificial intelligence, and data science"
)

# ==============================
# STEP 3: PREPARE COURSES
# ==============================

courses_df["Skills"] = courses_df["Skills"].fillna("").apply(normalize)

courses_df["Skills"] = courses_df["Skills"].apply(
    lambda x: [s.strip() for s in x.split(",")] if x else []
)

# ==============================
# STEP 4: MODEL + EMBEDDINGS
# ==============================

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding skills...")
skill_embeddings = model.encode(skills_df["description"].tolist())

# ==============================
# STEP 5: GRAPH
# ==============================

print("Building graph...")

sim_matrix = cosine_similarity(skill_embeddings)
G = nx.DiGraph()

threshold = 0.55

for i in range(len(skills_df)):
    for j in range(i + 1, len(skills_df)):
        if sim_matrix[i][j] > threshold:
            s1 = skills_df.iloc[i]["skill"]
            s2 = skills_df.iloc[j]["skill"]

            if len(s1) <= len(s2):
                G.add_edge(s1, s2, weight=1 - sim_matrix[i][j])
            else:
                G.add_edge(s2, s1, weight=1 - sim_matrix[i][j])

# 🔥 FORCE CORE PATH
core_edges = [
    ("python", "data preprocessing"),
    ("data preprocessing", "machine learning"),
    ("machine learning", "deep learning"),
    ("deep learning", "neural networks")
]

for u, v in core_edges:
    G.add_edge(u, v, weight=0.1)

print("Graph Stats:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ==============================
# STEP 6: USER INPUT
# ==============================

known_skills = ["python"]
goal = "deep learning"

known_skills = [normalize(s) for s in known_skills]
goal = normalize(goal)

# ==============================
# STEP 7: SKILL RECOMMENDATION
# ==============================

print("\nGenerating recommendations...")

user_vec = model.encode([" ".join(known_skills)])[0]
goal_vec = model.encode([goal])[0]

scores_user = cosine_similarity([user_vec], skill_embeddings)[0]

combined_scores = []

for i, skill in enumerate(skills_df["skill"]):
    score_user = scores_user[i]
    score_goal = cosine_similarity(
        [skill_embeddings[i]], [goal_vec]
    )[0][0]

    final_score = 0.4 * score_user + 0.6 * score_goal
    combined_scores.append((skill, final_score))

combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

# 🔥 STRONG FILTERING
bad_words = [
    "basic", "fundamental", "introduction", "fluency",
    "beginner", "syntax", "operator", "method",
    "data structures", "functions", "tools", "ides"
]

good_words = [
    "machine learning", "deep learning", "neural",
    "data", "model", "ai", "analysis", "statistics"
]

def valid(skill):
    if any(b in skill for b in bad_words):
        return False
    if not any(g in skill for g in good_words):
        return False
    return True

# remove near-duplicates
def clean_skill_name(skill):
    return skill.replace("learning", "").strip()

seen_clean = set()
recommended_skills = []

for skill, _ in combined_scores:
    base = clean_skill_name(skill)

    if skill not in known_skills and base not in seen_clean:
        if valid(skill):
            recommended_skills.append(skill)
            seen_clean.add(base)

    if len(recommended_skills) == 5:
        break

# ==============================
# STEP 8: PATH
# ==============================

print("\nFinding optimal path...")

try:
    path = nx.dijkstra_path(G, source=known_skills[0], target=goal, weight="weight")
except:
    path = ["No meaningful path found"]

# ==============================
# STEP 9: OUTPUT
# ==============================

print("\n==============================")
print("📌 FINAL OUTPUT")
print("==============================")

print("\n👤 User Skills:", known_skills)
print("🎯 Goal:", goal)

print("\n🔝 Recommended Skills:")
for s in recommended_skills:
    print("-", s)

print("\n🧭 Learning Path:")
print(" → ".join(path))

print("\n💡 Explanation:")
if "No" not in path[0]:
    for i in range(len(path) - 1):
        print(f"{path[i]} → {path[i+1]} (builds required knowledge)")
else:
    print("Graph needs improvement")

# ==============================
# STEP 10: COURSE RECOMMENDATION
# ==============================

def recommend_courses(skill, top_k=3):
    skill_vec = model.encode([skill])[0]
    scores = []

    for _, row in courses_df.iterrows():
        if not row["Skills"]:
            continue

        # relaxed matching (important fix)
        if not any(skill in s or s in skill for s in row["Skills"]):
            continue

        course_text = " ".join(row["Skills"])
        course_vec = model.encode([course_text])[0]

        score = cosine_similarity([skill_vec], [course_vec])[0][0]

        if score > 0.5:
            scores.append((row["Title"], score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # remove duplicates
    unique = []
    seen = set()

    for title, score in scores:
        if title not in seen:
            unique.append((title, score))
            seen.add(title)
        if len(unique) == top_k:
            break

    return unique

print("\n📚 Recommended Courses:")

if "No" not in path[0]:
    for skill in path:
        print(f"\n🔹 Skill: {skill}")
        courses = recommend_courses(skill)

        if not courses:
            print("No courses found")
        else:
            for title, score in courses:
                print(f"- {title} ({round(score, 2)})")