# GNN-Based Personalized Skill and Course Recommendation System with Explainability

## Overview

This project presents an intelligent and explainable learning recommendation system that generates personalized skill learning paths and course recommendations using semantic embeddings, graph-based reasoning, and Graph Neural Networks (GNNs).

The system models skills and courses as a knowledge graph and recommends relevant learning paths based on a user’s existing skills and learning goals. Unlike traditional rule-based systems, the proposed approach uses embedding-based semantic understanding and graph learning to discover hidden relationships between skills.

The system provides:
- Personalized skill recommendations
- Explainable learning paths
- Course recommendations
- Graph-based reasoning
- Semantic similarity learning
- Cold-start recommendations

---

# Problem Statement

Existing online learning platforms often recommend courses without explaining:
- Why a skill is required
- How skills are connected
- What sequence should be followed

This project addresses the problem by generating explainable and personalized learning paths using embeddings, graph neural networks, and knowledge graph reasoning.

---

# System Architecture

```text
User Skills + Goal
        ↓
SBERT Embeddings
        ↓
Skill Similarity Graph
        ↓
Graph Neural Network (GraphSAGE)
        ↓
Skill Recommendation
        ↓
Learning Path Generation
        ↓
Explainability
        ↓
Course Recommendation
```

---

# Methodology

## 1. Dataset Preparation

The system uses:
- `skills.csv` → cleaned technical skill dataset
- `Online_Courses.csv` → courses and associated skills

The datasets contain:
- Skill names
- Course titles
- Course descriptions
- Skill mappings

---

## 2. Semantic Embeddings

Sentence-BERT (`all-MiniLM-L6-v2`) is used to generate dense semantic embeddings for:
- Skills
- Course descriptions

This enables the system to identify hidden semantic relationships between concepts such as:
- Deep Learning ↔ Neural Networks
- Transformers ↔ Large Language Models

---

## 3. Skill Graph Construction

A similarity-based knowledge graph is constructed using cosine similarity between skill embeddings.

### Nodes
- Skills
- Courses

### Edges
- Skill ↔ Skill similarity
- Course → Skill relationship

The graph is dynamically generated instead of manually handcrafted.

---

## 4. Graph Neural Network

The project uses GraphSAGE to learn refined node representations from graph neighborhoods.

The GNN helps the system:
- Learn hidden skill dependencies
- Capture neighborhood influence
- Improve recommendation quality

---

## 5. Personalized Recommendation

User skills are converted into embeddings and compared against candidate skill embeddings using cosine similarity.

The system recommends:
- Relevant missing skills
- Ordered learning paths
- Related courses

---

## 6. Explainability

The recommendation process is explainable using:
- Graph traversal paths
- Similarity reasoning
- Skill dependency chains

### Example

```text
Python → Machine Learning → Deep Learning
```

This explains why Deep Learning is recommended for a user who already knows Python.

---

# Features

- Personalized learning path generation
- Explainable AI recommendations
- Graph-based recommendation engine
- Skill dependency reasoning
- Cold-start recommendation support
- Semantic similarity learning
- Course recommendation system

---

# Project Structure

```text
dataset/
    skills.csv
    Online_Courses.csv

scripts/
    generate_skills.py

code.py
evaluate.py
README.md
```

---

# Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- SentenceTransformers
- Scikit-learn
- Pandas
- NumPy

---

# Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Recommendation System

```bash
python code.py
```

---

# Run Evaluation

```bash
python evaluate.py
```

---

# Example Output

```text
User Skills: Python
Goal: Deep Learning

Recommended Skills:
- Machine Learning
- Neural Networks
- Data Preprocessing

Learning Path:
Python → Machine Learning → Deep Learning

Recommended Courses:
- Machine Learning Specialization
- Deep Learning Specialization
```

---

# Future Enhancements

- Advanced GNN architectures (GCN, GAT)
- Learning-to-Rank models
- Real-time user interaction tracking
- Adaptive learning path optimization
- Reinforcement learning-based recommendation
- Web-based interactive interface

---

# Conclusion

This project demonstrates how semantic embeddings, graph-based reasoning, and Graph Neural Networks can be combined to build an explainable and personalized learning recommendation system.

The proposed system moves beyond static roadmap generation by dynamically learning relationships between skills and generating interpretable learning paths tailored to individual users.
