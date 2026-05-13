# GNN-Based Personalized Skill and Course Recommendation System with Explainability

## Overview
This project proposes a personalized and explainable learning recommendation system using:
- Sentence-BERT (SBERT)
- GraphSAGE Graph Neural Network
- Skill Knowledge Graph
- Dijkstra’s Algorithm

The system generates:
- Skill recommendations
- Learning paths
- Course recommendations
- Multi-level explainability

## Features
- Cold-start recommendation
- Skill-level learning path generation
- Graph-based recommendation engine
- Explainable AI recommendations

## Project Structure

```text
dataset/
    skills.csv
    Online_Courses.csv

scripts/
    generate_skills.py

code.py
evaluate.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Recommendation System

```bash
python code.py
```

## Run Evaluation

```bash
python evaluate.py
```

## Technologies Used
- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- SentenceTransformers
- Scikit-learn