
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

target_labels = ["food", "water", "shelter", "medical_help",
                 "search_and_rescue", "death",
                 "earthquake", "fire", "storm"]

app = FastAPI()

high_urgency = ["dying","dead","death","killed","emergency",
                "critical","severe","immediate","now","urgent",
                "bleeding","unconscious","trapped"]

medium_urgency = ["help","need","require","please","sick",
                  "injured","hurt","pain","hungry","starving",
                  "thirsty","missing","lost","alone"]

low_urgency = ["cold","tired","worried","scared","unsafe",
               "problem","issue","concern","difficult"]

def predict_category(text):
    flood_words = ["flood","flooding","flooded"]
    pred = model.predict([text])[0]
    detected = [target_labels[i] for i, val in enumerate(pred) if val == 1]
    if any(word in text.lower() for word in flood_words):
        if "floods" not in detected:
            detected.append("floods")
    return detected if detected else ["other"]

def calculate_priority(text):
    text_lower = text.lower()
    score = 0
    for word in high_urgency:
        if word in text_lower:
            score += 3
    for word in medium_urgency:
        if word in text_lower:
            score += 2
    for word in low_urgency:
        if word in text_lower:
            score += 1
    score = min(score, 10)
    if score >= 7:
        level = "HIGH"
    elif score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level

def match_volunteers(case_text, volunteers):
    all_texts = [case_text] + [v["skills"] for v in volunteers]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    case_vector = tfidf_matrix[0]
    volunteer_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(case_vector, volunteer_vectors)[0]
    ranked = sorted(zip(volunteers, scores), key=lambda x: x[1], reverse=True)
    result = []
    for volunteer, score in ranked:
        result.append({
            "name": volunteer["name"],
            "skills": volunteer["skills"],
            "match_score": round(float(score), 2)
        })
    return result

class TextInput(BaseModel):
    text: str

class CaseInput(BaseModel):
    text: str
    category: str

class VolunteerMatchInput(BaseModel):
    case_text: str
    volunteers: list

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict-category")
def predict_category_api(data: TextInput):
    result = predict_category(data.text)
    return {"categories": result}

@app.post("/priority-score")
def priority_score_api(data: CaseInput):
    score, level = calculate_priority(data.text)
    return {"score": score, "level": level}

@app.post("/match-volunteers")
def match_volunteers_api(data: VolunteerMatchInput):
    result = match_volunteers(data.case_text, data.volunteers)
    return {"ranked_volunteers": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
