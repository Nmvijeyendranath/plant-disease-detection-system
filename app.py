import os
import sys
import shutil
import json
import sqlite3
from datetime import datetime
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

import tensorflow as tf
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils.model_loader import get_model

load_dotenv()

app = FastAPI(title="CropGuard AI")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
DB_PATH = os.path.join(BASE_DIR, "database", "history.db")

# Mount static files and templates
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Database Setup ─────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT NOT NULL,
            display_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            urgency TEXT NOT NULL,
            color TEXT,
            img_url TEXT,
            city TEXT DEFAULT '',
            field_area REAL DEFAULT 1.0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ─── Gemini AI Setup ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
chat_client = None

try:
    from google import genai
    if GEMINI_API_KEY:
        chat_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[INFO] Google Gemini AI loaded successfully.")
    else:
        print("[WARNING] GEMINI_API_KEY not set. Chatbot will run in offline mode.")
except Exception as e:
    print(f"[WARNING] Could not load google-genai: {e}. Chatbot disabled.")

# These must match the order your training directory sorted the class folders
CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

DISEASE_INFO = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display": "Corn Gray Leaf Spot",
        "scientific": "Cercospora zeae-maydis",
        "icon": "🌾",
        "color": "#a8a29e",
        "cause": "Fungal infection caused by Cercospora zeae-maydis",
        "symptoms": "Rectangular, gray to tan lesions that run parallel to leaf veins. Lesions may coalesce, causing large areas of dead tissue.",
        "treatment": (
            "1. Apply foliar fungicides (strobilurin or triazole-based) at early signs.\n"
            "2. Remove and destroy crop residue after harvest.\n"
            "3. Rotate with non-host crops (soybeans, small grains) for 1–2 years.\n"
            "4. Use resistant or tolerant corn hybrids.\n"
            "5. Ensure proper plant spacing for air circulation."
        ),
        "prevention": "Use resistant hybrids. Practice crop rotation. Reduce surface residue through tillage. Avoid continuous corn planting.",
        "urgency": "moderate",
        "yield_loss_pct": 30
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Corn Common Rust",
        "scientific": "Puccinia sorghi",
        "icon": "🟤",
        "color": "#d97706",
        "cause": "Fungal infection caused by Puccinia sorghi",
        "symptoms": "Small, elongated, cinnamon-brown to dark brown pustules on both leaf surfaces. Pustules may appear in bands across the leaf.",
        "treatment": (
            "1. Apply foliar fungicides if infection is detected before tasseling.\n"
            "2. Plant resistant hybrids in areas prone to rust.\n"
            "3. Monitor fields regularly during warm, humid weather.\n"
            "4. Remove heavily infected plant debris after harvest."
        ),
        "prevention": "Plant resistant hybrids. Early planting to avoid peak spore periods. Scout fields regularly during humid conditions.",
        "urgency": "moderate",
        "yield_loss_pct": 20
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Corn Northern Leaf Blight",
        "scientific": "Exserohilum turcicum",
        "icon": "🍃",
        "color": "#7c6f64",
        "cause": "Fungal infection caused by Exserohilum turcicum (Helminthosporium turcicum)",
        "symptoms": "Long, elliptical, grayish-green to tan lesions (1–6 inches) on leaves. Lesions may have wavy margins and can coalesce to blight entire leaves.",
        "treatment": (
            "1. Apply foliar fungicides (azoxystrobin, propiconazole) at first sign of disease.\n"
            "2. Remove crop residue to reduce inoculum.\n"
            "3. Rotate crops — avoid continuous corn.\n"
            "4. Use resistant or tolerant hybrids.\n"
            "5. Ensure adequate plant nutrition to improve plant vigor."
        ),
        "prevention": "Use resistant hybrids. Rotate crops. Manage crop residue. Scout fields during warm, humid weather.",
        "urgency": "moderate",
        "yield_loss_pct": 30
    },
    "Corn_(maize)___healthy": {
        "display": "Healthy Corn Plant",
        "scientific": "Zea mays (healthy)",
        "icon": "✅",
        "color": "#22c55e",
        "cause": "No pathogen detected",
        "symptoms": "Leaves appear vibrant green with no spots, lesions, or discolouration.",
        "treatment": (
            "1. Continue regular watering and fertilization schedule.\n"
            "2. Apply balanced NPK fertiliser during growing season.\n"
            "3. Inspect leaves weekly for early signs of disease.\n"
            "4. Maintain weed control to reduce pest habitats.\n"
            "5. Ensure adequate sunlight exposure for the crop."
        ),
        "prevention": "Continue standard crop management. Regular inspection. Maintain soil health with organic matter.",
        "urgency": "none",
        "yield_loss_pct": 0
    },
    "Potato___Early_blight": {
        "display": "Potato Early Blight",
        "scientific": "Alternaria solani",
        "icon": "🍂",
        "color": "#f0b429",
        "cause": "Fungal infection caused by Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings on lower/older leaves, surrounded by yellow halo.",
        "treatment": (
            "1. Apply chlorothalonil or copper-based fungicide every 7–10 days.\n"
            "2. Remove and destroy all infected leaves immediately.\n"
            "3. Water at the base — avoid wetting foliage.\n"
            "4. Rotate crops next season to break the disease cycle.\n"
            "5. Ensure adequate spacing for good air circulation."
        ),
        "prevention": "Rotate crops annually. Use certified disease-free seeds. Apply mulch to prevent soil splash. Maintain proper plant spacing.",
        "urgency": "moderate",
        "yield_loss_pct": 25
    },
    "Potato___Late_blight": {
        "display": "Potato Late Blight",
        "scientific": "Phytophthora infestans",
        "icon": "🚨",
        "color": "#ef4444",
        "cause": "Oomycete pathogen Phytophthora infestans",
        "symptoms": "Water-soaked, dark lesions on leaves and stems; white mold on leaf undersides in humid conditions.",
        "treatment": (
            "1. Apply copper-based or mancozeb fungicide IMMEDIATELY — late blight spreads fast.\n"
            "2. Quarantine or destroy heavily infected plants to protect neighbours.\n"
            "3. Avoid overhead irrigation; water only at soil level.\n"
            "4. Remove all plant debris from the field.\n"
            "5. Monitor surrounding plants daily — report outbreak to local agriculture office."
        ),
        "prevention": "Use certified blight-free seed potatoes. Plant resistant varieties. Avoid planting near previously infected fields.",
        "urgency": "critical",
        "yield_loss_pct": 50
    },
    "Potato___healthy": {
        "display": "Healthy Potato Plant",
        "scientific": "Solanum tuberosum (healthy)",
        "icon": "✅",
        "color": "#22c55e",
        "cause": "No pathogen detected",
        "symptoms": "Leaves appear vibrant with no spots, lesions, or discolouration.",
        "treatment": (
            "1. Continue regular watering schedule (avoid waterlogging).\n"
            "2. Apply balanced NPK fertiliser during growing season.\n"
            "3. Inspect leaves weekly for early signs of disease.\n"
            "4. Maintain weed control to reduce pest habitats.\n"
            "5. Ensure adequate sunlight exposure for the crop."
        ),
        "prevention": "Continue standard crop management. Regular inspection. Maintain soil health with organic matter.",
        "urgency": "none",
        "yield_loss_pct": 0
    },
    "Tomato___Bacterial_spot": {
        "display": "Tomato Bacterial Spot",
        "scientific": "Xanthomonas campestris pv. vesicatoria",
        "icon": "🦠",
        "color": "#eab308",
        "cause": "Bacterium Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small, water-soaked, dark spots on leaves and stems, eventually turning necrotic.",
        "treatment": (
            "1. Apply copper-based bactericides.\n"
            "2. Avoid overhead watering to reduce spread.\n"
            "3. Remove heavily infected plants.\n"
            "4. Practice crop rotation and weed control."
        ),
        "prevention": "Use disease-free seeds and transplants. Avoid handling plants when wet. Rotate crops every 2-3 years.",
        "urgency": "moderate",
        "yield_loss_pct": 20
    },
    "Tomato___Early_blight": {
        "display": "Tomato Early Blight",
        "scientific": "Alternaria solani",
        "icon": "🍂",
        "color": "#f59e0b",
        "cause": "Fungal infection caused by Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings on lower/older leaves, surrounded by yellow halo.",
        "treatment": (
            "1. Apply chlorothalonil or copper-based fungicide every 7–10 days.\n"
            "2. Remove and destroy all infected leaves immediately.\n"
            "3. Water at the base — avoid wetting foliage.\n"
            "4. Ensure adequate spacing for good air circulation."
        ),
        "prevention": "Rotate crops annually. Use certified disease-free seeds. Apply mulch to prevent soil splash.",
        "urgency": "moderate",
        "yield_loss_pct": 25
    },
    "Tomato___Late_blight": {
        "display": "Tomato Late Blight",
        "scientific": "Phytophthora infestans",
        "icon": "🚨",
        "color": "#ef4444",
        "cause": "Oomycete pathogen Phytophthora infestans",
        "symptoms": "Water-soaked, dark lesions on leaves and stems; white mold on leaf undersides in humid conditions.",
        "treatment": (
            "1. Apply copper-based or mancozeb fungicide IMMEDIATELY — late blight spreads fast.\n"
            "2. Quarantine or destroy heavily infected plants to protect neighbours.\n"
            "3. Avoid overhead irrigation; water only at soil level.\n"
            "4. Remove all plant debris from the field."
        ),
        "prevention": "Use certified blight-free seeds. Plant resistant varieties. Monitor weather for humid conditions.",
        "urgency": "critical",
        "yield_loss_pct": 50
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato Leaf Mold",
        "scientific": "Passalora fulva",
        "icon": "🍄",
        "color": "#a8a29e",
        "cause": "Fungus Passalora fulva",
        "symptoms": "Pale green or yellowish spots on upper leaf surfaces; olive-green to brown fuzzy mold on undersides.",
        "treatment": (
            "1. Improve air circulation by pruning and spacing.\n"
            "2. Reduce humidity around plants.\n"
            "3. Apply appropriate fungicides if the disease progresses."
        ),
        "prevention": "Ensure good ventilation in greenhouses. Avoid excessive humidity. Use resistant varieties.",
        "urgency": "low",
        "yield_loss_pct": 10
    },
    "Tomato___Septoria_leaf_spot": {
        "display": "Tomato Septoria Leaf Spot",
        "scientific": "Septoria lycopersici",
        "icon": "⚫",
        "color": "#78716c",
        "cause": "Fungus Septoria lycopersici",
        "symptoms": "Numerous small, circular spots with dark borders and lighter centers on older leaves.",
        "treatment": (
            "1. Remove affected leaves.\n"
            "2. Water at the base to keep foliage dry.\n"
            "3. Apply fungicidal sprays if necessary.\n"
            "4. Clean up plant debris at the end of the season."
        ),
        "prevention": "Remove crop debris. Avoid overhead watering. Rotate crops. Use pathogen-free transplants.",
        "urgency": "moderate",
        "yield_loss_pct": 20
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display": "Tomato Spider Mites",
        "scientific": "Tetranychus urticae",
        "icon": "🕷️",
        "color": "#b91c1c",
        "cause": "Two-spotted spider mite (Tetranychus urticae)",
        "symptoms": "Stippling or small yellow dots on leaves; fine webbing may be visible on undersides.",
        "treatment": (
            "1. Apply insecticidal soap or neem oil.\n"
            "2. Ensure plants are well-watered to reduce stress.\n"
            "3. Introduce natural predators like ladybugs or predatory mites."
        ),
        "prevention": "Maintain adequate watering. Avoid dusty conditions. Introduce biological controls early.",
        "urgency": "moderate",
        "yield_loss_pct": 15
    },
    "Tomato___Target_Spot": {
        "display": "Tomato Target Spot",
        "scientific": "Corynespora cassiicola",
        "icon": "🎯",
        "color": "#8b5cf6",
        "cause": "Fungus Corynespora cassiicola",
        "symptoms": "Dark brown, necrotic lesions with concentric rings, similar to early blight but often larger.",
        "treatment": (
            "1. Apply appropriate fungicides.\n"
            "2. Improve ventilation and avoid overhead watering.\n"
            "3. Remove infected plant material."
        ),
        "prevention": "Use resistant varieties. Maintain good field hygiene. Ensure proper ventilation.",
        "urgency": "moderate",
        "yield_loss_pct": 20
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato Yellow Leaf Curl Virus",
        "scientific": "Begomovirus (TYLCV)",
        "icon": "📉",
        "color": "#facc15",
        "cause": "Begomovirus transmitted by whiteflies",
        "symptoms": "Upward curling of leaf margins, yellowing (chlorosis) of leaf tissues, stunted plant growth.",
        "treatment": (
            "1. Control whitefly populations using sticky traps or insecticidal soaps.\n"
            "2. Remove and destroy infected plants to prevent spread.\n"
            "3. Use virus-resistant tomato varieties if possible."
        ),
        "prevention": "Use reflective mulch to deter whiteflies. Install insect nets. Plant resistant varieties.",
        "urgency": "critical",
        "yield_loss_pct": 60
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": "Tomato Mosaic Virus",
        "scientific": "Tobamovirus (ToMV)",
        "icon": "🧩",
        "color": "#4ade80",
        "cause": "Tobamovirus",
        "symptoms": "Mottled, light and dark green mosaic patterns on leaves; stunted growth and distorted leaves.",
        "treatment": (
            "1. No cure; remove and destroy infected plants.\n"
            "2. Disinfect tools and wash hands frequently to prevent mechanical transmission.\n"
            "3. Control weeds that may harbor the virus."
        ),
        "prevention": "Sterilize tools. Wash hands before handling plants. Use certified virus-free seeds.",
        "urgency": "critical",
        "yield_loss_pct": 40
    },
    "Tomato___healthy": {
        "display": "Healthy Tomato Plant",
        "scientific": "Solanum lycopersicum (healthy)",
        "icon": "✅",
        "color": "#22c55e",
        "cause": "No pathogen detected",
        "symptoms": "Leaves appear vibrant with no spots, lesions, or discolouration.",
        "treatment": (
            "1. Continue regular watering schedule (avoid waterlogging).\n"
            "2. Apply balanced NPK fertiliser during growing season.\n"
            "3. Inspect leaves weekly for early signs of disease.\n"
            "4. Ensure adequate sunlight exposure for the crop."
        ),
        "prevention": "Continue standard crop management. Regular inspection. Maintain soil health with organic matter.",
        "urgency": "none",
        "yield_loss_pct": 0
    }
}

# ─── Prediction ─────────────────────────────────────────────────────────────
def predict_disease(img_path: str):
    trained_model = get_model()

    if trained_model is None:
        return {
            "error": True,
            "message": "Model not found. Please train the model first by running train_cnn_final.py."
        }

    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        # EfficientNetV2 expects raw [0,255] pixels (no manual normalisation)

        preds = trained_model.predict(img_array, verbose=0)
        class_id = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # All-class probabilities
        class_probs = {CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 1)
                       for i in range(len(CLASS_NAMES))}

        disease = CLASS_NAMES[class_id]
        info = DISEASE_INFO[disease]

        # Severity
        if "healthy" in disease.lower():
            severity = "None"
            severity_level = 0
        elif confidence > 0.85:
            severity = "High"
            severity_level = 3
        elif confidence > 0.65:
            severity = "Medium"
            severity_level = 2
        else:
            severity = "Low"
            severity_level = 1

        return {
            "error": False,
            "disease": disease,
            "display_name": info["display"],
            "scientific_name": info.get("scientific", ""),
            "icon": info["icon"],
            "color": info["color"],
            "confidence": round(confidence * 100, 1),
            "severity": severity,
            "severity_level": severity_level,
            "cause": info["cause"],
            "symptoms": info["symptoms"],
            "treatment": info["treatment"],
            "prevention": info.get("prevention", ""),
            "urgency": info["urgency"],
            "yield_loss_pct": info.get("yield_loss_pct", 0),
            "class_probs": class_probs
        }
    except Exception as e:
        return {"error": True, "message": f"Prediction failed: {str(e)}"}


# ─── History Helpers ─────────────────────────────────────────────────────────
def save_to_history(result: dict, img_url: str, city: str, field_area: float):
    conn = get_db()
    conn.execute(
        """INSERT INTO scan_history (disease, display_name, confidence, severity, urgency, color, img_url, city, field_area, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (result["disease"], result["display_name"], result["confidence"],
         result["severity"], result["urgency"], result["color"],
         img_url, city, field_area, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_scan_history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM scan_history ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    history = []
    for r in rows:
        history.append({
            "disease": r["display_name"],
            "confidence": r["confidence"],
            "severity": r["severity"],
            "color": r["color"],
            "img_url": r["img_url"] or "",
            "date": r["created_at"][:16].replace("T", " "),
        })
    return history


def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
    healthy = conn.execute("SELECT COUNT(*) FROM scan_history WHERE disease LIKE '%healthy%'").fetchone()[0]
    diseased = total - healthy
    # Sum yield losses: each scan's field_area * yield_loss_pct * baseline_yield / 100
    # Simplified: just count total
    conn.close()
    return {"total": total, "healthy": healthy, "diseased": diseased, "yield_loss": diseased * 20}


def get_disease_distribution():
    conn = get_db()
    rows = conn.execute(
        "SELECT display_name, COUNT(*) as cnt FROM scan_history GROUP BY display_name ORDER BY cnt DESC"
    ).fetchall()
    total = sum(r["cnt"] for r in rows) if rows else 1
    conn.close()

    colors = ["#f0b429", "#ef4444", "#22c55e", "#eab308", "#8b5cf6", "#a8a29e", "#b91c1c", "#4ade80", "#facc15"]
    distribution = []
    for i, r in enumerate(rows):
        distribution.append({
            "name": r["display_name"],
            "count": r["cnt"],
            "pct": round(r["cnt"] / total * 100, 1),
            "color": colors[i % len(colors)]
        })
    return distribution


# Store last prediction result in memory for page reload
last_prediction = {}


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def index(request: Request):
    model_trained = os.path.exists(MODEL_PATH)
    show_result = request.query_params.get("show_result")

    # Build template context
    ctx = {
        "model_trained": model_trained,
        "result": None,
        "stats": get_stats(),
        "disease_distribution": get_disease_distribution(),
        "scan_history": get_scan_history(),
    }

    # Show last prediction result if redirected after predict
    if show_result and last_prediction:
        r = last_prediction
        ctx.update({
            "result": r.get("display_name"),
            "disease_raw": r.get("disease", ""),
            "scientific_name": r.get("scientific_name", ""),
            "confidence": r.get("confidence", 0),
            "severity": r.get("severity", "Unknown"),
            "urgency": r.get("urgency", "Unknown"),
            "color": r.get("color", "#f0b429"),
            "treatment": r.get("treatment", ""),
            "prevention": r.get("prevention", ""),
            "icon": r.get("icon", "🔬"),
            "class_probs": r.get("class_probs", {}),
            "result_time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "city": r.get("_city", ""),
            "field_area": r.get("_field_area", 1.0),
            # Yield estimates
            "expected_yield": round(float(r.get("_field_area", 1.0)) * 25, 1),
            "yield_loss": round(float(r.get("_field_area", 1.0)) * 25 * r.get("yield_loss_pct", 0) / 100, 1),
            "loss_pct": r.get("yield_loss_pct", 0),
        })

        # Individual class probabilities for the prob bars
        probs = r.get("class_probs", {})
        
        # Determine crop type from the detected disease
        detected_disease = r.get("disease", "")
        if "Corn" in detected_disease:
            crop_type = "Corn"
        elif "Tomato" in detected_disease:
            crop_type = "Tomato"
        else:
            crop_type = "Potato"
        
        filtered_probs = []
        for cls_name, prob_val in probs.items():
            if crop_type in cls_name:
                display_name = DISEASE_INFO[cls_name]["display"].replace(f"{crop_type} ", "").replace("Corn ", "")
                color = DISEASE_INFO[cls_name]["color"]
                filtered_probs.append({"name": display_name, "prob": prob_val, "color": color})
                
        # Sort descending by probability
        filtered_probs.sort(key=lambda x: x["prob"], reverse=True)
        ctx["filtered_probs"] = filtered_probs

    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.post("/predict")
async def predict_route(
    file: UploadFile = File(...),
    city: str = Form(""),
    field_area: float = Form(1.0),
):
    global last_prediction

    if not file.filename:
        return JSONResponse({"error": True, "message": "No file uploaded"}, status_code=400)

    uploads_dir = os.path.join(BASE_DIR, "static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Save with timestamp to avoid overwrites
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    save_name = f"scan_{ts}{ext}"
    save_path = os.path.join(uploads_dir, save_name)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return JSONResponse({"error": True, "message": f"Could not save file: {e}"}, status_code=500)

    result = predict_disease(save_path)

    if result.get("error"):
        return result

    img_url = f"/static/uploads/{save_name}"
    result["img_url"] = img_url

    # Store city/field_area for template context
    result["_city"] = city
    result["_field_area"] = field_area

    # Save to database
    save_to_history(result, img_url, city, field_area)

    # Store for page reload
    last_prediction = result

    return result


@app.post("/clear_history")
async def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM scan_history")
    conn.commit()
    conn.close()
    return {"status": "ok"}


class ChatRequest(BaseModel):
    message: str = ""
    language: str = "en-US"
    context: str = ""


@app.post("/chat")
async def chat(data: ChatRequest):
    user_message = data.message.strip()
    language = data.language
    context = data.context   # disease context from last prediction

    if not user_message:
        return JSONResponse({"response": "Please type a question."}, status_code=400)

    if not chat_client:
        return {
            "response": (
                "I'm currently in offline mode. "
                "To enable the AI assistant, add your GEMINI_API_KEY to the .env file and restart the app."
            )
        }

    try:
        scan_hint = ""
        if context:
            scan_hint = (
                f"\n\nNote: The farmer's most recent scan detected: {context}. "
                f"Only reference this if the user's question is clearly about it."
            )

        prompt = (
            f"You are CropGuard AI, an expert agricultural assistant helping farmers identify and manage "
            f"crop diseases for ANY plant — not just the last scanned one. "
            f"You are friendly, concise, and practical.{scan_hint}\n\n"
            f"LANGUAGE INSTRUCTIONS:\n"
            f"- The user's selected language code is: {language}\n"
            f"- The user may type in English, or in their native language, or in transliterated form "
            f"(e.g. Tamil/Hindi/Telugu written in English letters like 'yellai patti sollu' or 'isko kaise theek kare').\n"
            f"- Detect the language or transliteration from the message itself.\n"
            f"- Always reply in the SAME language the user wrote in. If they wrote transliterated Tamil, reply in transliterated Tamil. "
            f"If they wrote Hindi script, reply in Hindi script. If English, reply in English.\n"
            f"- Never switch languages mid-response.\n\n"
            f"Keep response to 3–5 sentences max. Be actionable.\n\n"
            f"User: {user_message}"
        )

        # Try multiple models as fallback for rate limits
        models_to_try = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
        last_error = None

        for model_name in models_to_try:
            try:
                response = chat_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return {"response": response.text}
            except Exception as model_err:
                last_error = model_err
                error_str = str(model_err)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    continue  # Try next model
                else:
                    raise  # Non-rate-limit error, re-raise

        # All models exhausted
        return {
            "response": (
                "⚠️ The AI service is temporarily rate-limited. "
                "Please wait a moment and try again. "
                "Your API key is valid but the free tier quota has been reached."
            )
        }
    except Exception as e:
        return JSONResponse({"response": f"AI service error: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
