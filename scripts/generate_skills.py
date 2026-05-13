import os
import re
import pandas as pd
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '..', 'dataset'))
input_path = os.path.join(data_dir, 'Online_Courses.csv')
output_path = os.path.join(data_dir, 'skills.csv')

print('Reading:', input_path)

df2 = pd.read_csv(input_path)
# normalize column name 'Skills' if needed
if 'Skills' not in df2.columns and 'skills' in df2.columns:
    df2.rename(columns={'skills': 'Skills'}, inplace=True)

skills_series = df2.get('Skills', pd.Series(dtype=str)).fillna('')

all_skills = []
for s in skills_series:
    if not isinstance(s, str):
        continue
    parts = [p.strip() for p in s.split(',') if p.strip()]
    all_skills.extend(parts)

# canonical technical keywords/phrases to match (lowercase)
technical_keywords = [
    'python','java','c++','c#','r','sql','nosql','mongodb','postgres','mysql','oracle',
    'tensorflow','keras','pytorch','torch','scikit','sklearn','numpy','pandas','matplotlib',
    'seaborn','opencv','nlp','natural language processing','computer vision','deep learning',
    'machine learning','ml','data engineering','hadoop','spark','scala',
    'hive','pig','aws','amazon web services','azure','gcp','google cloud','docker','kubernetes','helm',
    'terraform','ansible','ci/cd','jenkins','git','github','gitlab','html','css','javascript',
    'node','react','angular','vue','flask','django','rest api','api','apache','elasticsearch','kafka',
    'redis','rabbitmq','xgboost','lightgbm','tableau','power bi','powerbi','dax','sql server',
    'mlops','bash','linux','cassandra','hbase','s3','lambda','ecs','ec2','bigquery','dataproc','snowflake'
]

# compile regex patterns (word-boundary for alphanumeric phrases)
patterns = []
for kw in technical_keywords:
    esc = re.escape(kw)
    # if kw contains spaces or non-word chars, match as substring with boundaries around words
    if re.search(r"[^\w\s]", kw) or ' ' in kw:
        patterns.append(re.compile(rf"{esc}", flags=re.I))
    else:
        patterns.append(re.compile(rf"\b{esc}\b", flags=re.I))

def clean_token(tok):
    t = tok.strip()
    # remove common list bullets/numbering
    t = re.sub(r'^[\-\u2022\*\s\d\.)]+', '', t)
    t = re.sub(r'[\t\n]+', ' ', t)
    t = t.strip(' -:;,')
    return t

def is_technical(tok):
    s = tok.lower()
    # reject extremely long descriptions
    if len(s.split()) > 6:
        return False
    for p in patterns:
        if p.search(s):
            return True
    # common short tokens
    if s in {'js', 'cpp', 'c#', 'c++'}:
        return True
    return False

technical_set = set()
for raw in all_skills:
    tok = clean_token(raw)
    if not tok:
        continue
    if is_technical(tok):
        technical_set.add(tok.lower())

technical = sorted(technical_set)

# build descriptions: follow user's template "This skill involves {skill}"
skills_df = pd.DataFrame({'skill': technical})
skills_df['description'] = skills_df['skill'].apply(lambda x: f'This skill involves {x}')

# write CSV
skills_df.to_csv(output_path, index=False, encoding='utf-8')
print(f'Wrote {len(skills_df)} skills to', output_path)
