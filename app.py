from utils.sustainability_module import calculate_sustainability
from utils.urban_intelligence import analyze_urban_density, assess_greenery_sufficiency
from utils.tree_placement import generate_tree_positions, draw_tree_positions
from utils.suitability_heatmap import generate_suitability_heatmap, overlay_heatmap

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import requests
from fpdf import FPDF
import tempfile
import os
import json
import base64
from datetime import datetime
from utils.area_and_plant_calculator import calculate_plantable_area
from utils.plant_category_estimator import estimate_plants_by_category

# ── GROQ API KEY ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Green Space Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fdf8; }
    .section-header {
        background: linear-gradient(90deg, #2d6a4f, #52b788);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        margin: 1rem 0 0.5rem 0;
    }
    .badge-high   { background:#006400; color:white; padding:10px 20px; border-radius:8px; text-align:center; font-size:18px; font-weight:bold; }
    .badge-medium { background:#FF8C00; color:white; padding:10px 20px; border-radius:8px; text-align:center; font-size:18px; font-weight:bold; }
    .badge-low    { background:#8B0000; color:white; padding:10px 20px; border-radius:8px; text-align:center; font-size:18px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── U-NET MODEL ───────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class PlantableUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 32);   self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128); self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out  = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x);          e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))

# ── LOAD MODEL ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PlantableUNet().to(device)
    model_path = Path("model/checkpoints/best_model.pth")
    if not model_path.exists():
        model_path = Path("model/plantable_trained_model.pth")
    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        st.sidebar.success(f"Model loaded ({model_path.name})")
    else:
        st.sidebar.warning("No trained model found - using untrained weights")
    model.eval()
    return model, device

# ── WEATHER & SOIL ────────────────────────────────────────────────
def get_weather(lat, lon):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,"
               f"precipitation,wind_speed_10m"
               f"&forecast_days=3&timezone=auto")
        r = requests.get(url, timeout=8)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_soil(lat, lon):
    try:
        url = (f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
               f"lon={lon}&lat={lat}&property=phh2o"
               f"&depth=0-5cm&value=mean")
        r = requests.get(url, timeout=8)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def suitability_from_weather(weather, soil):
    score   = 50
    reasons = []
    if weather:
        c        = weather.get("current", {})
        temp     = c.get("temperature_2m", 25)
        humidity = c.get("relative_humidity_2m", 50)
        precip   = c.get("precipitation", 0)
        if 15 <= temp <= 35:
            score += 15; reasons.append(f"Temperature {temp}C is ideal for planting")
        else:
            score -= 10; reasons.append(f"Temperature {temp}C is not ideal")
        if humidity >= 40:
            score += 10; reasons.append(f"Humidity {humidity}% supports plant growth")
        else:
            score -= 5;  reasons.append(f"Low humidity ({humidity}%) - irrigation needed")
        if precip > 0:
            score += 5;  reasons.append(f"Recent rainfall detected - good soil moisture")
    if soil:
        try:
            layers = soil["properties"]["layers"]
            for layer in layers:
                if layer["name"] == "phh2o":
                    ph = layer["depths"][0]["values"]["mean"]
                    if ph:
                        ph_val = ph / 10
                        if 5.5 <= ph_val <= 7.5:
                            score += 15
                            reasons.append(f"Soil pH {ph_val:.1f} - good for most plants")
                        else:
                            score -= 5
                            reasons.append(f"Soil pH {ph_val:.1f} - may need amendment")
        except:
            reasons.append("Soil pH data unavailable for this location")
    return max(0, min(100, score)), reasons

# ── PDF REPORT ────────────────────────────────────────────────────
def generate_pdf(data, overlay_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_fill_color(45, 106, 79)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 8)
    pdf.cell(0, 14, "Urban Green Space Optimizer - Analysis Report", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 35)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True)
    pdf.cell(0, 6, f"Location: {data.get('location', 'Not specified')}", ln=True)
    pdf.ln(4)
    pdf.set_fill_color(82, 183, 136)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "  Analysis Results", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    for label, value in [
        ("Plantable Area",        f"{data['area']:.2f} sq.m"),
        ("Estimated Trees",       str(data['trees'])),
        ("Shrubs",                str(data['shrubs'])),
        ("Small Plants",          str(data['small_plants'])),
        ("Green Coverage",        f"{data['green_pct']:.1f}%"),
        ("Greenery Status",       data.get('greenery_status', 'N/A')),
        ("Urban Density",         data['density_class']),
        ("Plantation Potential",  data['plantation_potential']),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(80, 7, label + ":", border="B")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, border="B", ln=True)
    pdf.ln(4)
    pdf.set_fill_color(82, 183, 136)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "  Greenery Assessment", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 6, data.get("greenery_message", ""))
    pdf.ln(2)
    pdf.cell(0, 6, f"Recommendation: {data.get('greenery_rec', '')}", ln=True)
    pdf.ln(4)
    pdf.set_fill_color(82, 183, 136)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "  Sustainability Impact", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    for label, value in [
        ("CO2 Absorption/year",  f"{data['co2_kg']:.0f} kg"),
        ("Cooling Effect",       f"{data['cooling']:.2f} C"),
        ("Sustainability Index", f"{data['sus_index']:.0f} / 100"),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(80, 7, label + ":", border="B")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, border="B", ln=True)
    pdf.ln(4)
    if data.get("weather_score"):
        pdf.set_fill_color(82, 183, 136)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "  Weather and Soil Suitability", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)
        pdf.cell(0, 7, f"Suitability Score: {data['weather_score']} / 100", ln=True)
        for r in data.get("weather_reasons", []):
            pdf.cell(0, 6, f"  - {r}", ln=True)
        pdf.ln(4)
    if overlay_img is not None:
        pdf.set_fill_color(82, 183, 136)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "  Plantable Area Overlay", ln=True, fill=True)
        pdf.ln(3)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        try:
            pdf.image(tmp_path, x=15, w=180)
        finally:
            os.unlink(tmp_path)
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6,
        "Urban Green Space Optimizer - SRM Institute of Science and Technology",
        align="C")
    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name

# ── PREDICT ───────────────────────────────────────────────────────
def predict_mask(model, device, image_rgb):
    img    = cv2.resize(image_rgb, (256, 256)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()
    mask   = (pred > 0.5).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ── IMAGE TO BASE64 ───────────────────────────────────────────────
def img_to_b64(img_rgb):
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()

# ── FLOATING CHATBOT WIDGET ───────────────────────────────────────
def inject_chatbot(report_data, api_key):
    import streamlit.components.v1 as components
    rd = report_data

    # Build system prompt as a plain Python string (no f-string JS conflicts)
    system_prompt = (
        "You are a helpful assistant for the Urban Green Space Optimizer app. "
        "The user has analysed a land image. Here are the results: "
        f"Plantable Area: {rd.get('area', 0):.1f} sq.m, "
        f"Trees: {rd.get('trees', 0)}, "
        f"Shrubs: {rd.get('shrubs', 0)}, "
        f"Small Plants: {rd.get('small_plants', 0)}, "
        f"Green Coverage: {rd.get('green_pct', 0)}%, "
        f"Greenery Status: {rd.get('greenery_status', 'N/A')}, "
        f"Recommendation: {rd.get('greenery_rec', 'N/A')}, "
        f"Urban Density: {rd.get('density_class', 'N/A')}, "
        f"Built-up Area: {rd.get('built_up_pct', 0)}%, "
        f"Plantation Potential: {rd.get('plantation_potential', 'N/A')}, "
        f"CO2 Absorption per year: {rd.get('co2_kg', 0):.0f} kg, "
        f"Cooling Effect: {rd.get('cooling', 0):.2f} C, "
        f"Sustainability Index: {rd.get('sus_index', 0):.0f} out of 100, "
        f"WHO Standard Met: {'Yes' if rd.get('who_standard') else 'No'}, "
        f"Urban Planning Standard Met: {'Yes' if rd.get('urban_standard') else 'No'}, "
        f"Location: {rd.get('location', 'Not specified')}, "
        f"Weather Score: {rd.get('weather_score', 'N/A')}. "
        "Answer questions based on these results AND your general knowledge about "
        "urban green spaces, plants, sustainability, CO2, ecology, and plantation planning. "
        "Keep answers short, clear, and friendly."
    )

    # Escape for safe JS string embedding
    safe_prompt = system_prompt.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    safe_key    = api_key.strip()

    html_code = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: sans-serif; }

  #chat-wrapper {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 99999;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 10px;
  }

  #chat-panel {
    width: 320px;
    height: 420px;
    background: white;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.22);
    display: none;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid #d0ebd8;
  }

  #chat-header {
    background: linear-gradient(90deg, #2d6a4f, #52b788);
    color: white;
    padding: 10px 14px;
    font-weight: bold;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }

  #chat-close {
    background: none;
    border: none;
    color: white;
    font-size: 17px;
    cursor: pointer;
    line-height: 1;
    padding: 0 2px;
  }

  #chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 7px;
    background: #f6fcf7;
  }

  .msg {
    max-width: 88%;
    padding: 7px 11px;
    border-radius: 12px;
    font-size: 12.5px;
    line-height: 1.45;
    word-wrap: break-word;
  }
  .msg.user {
    background: #2d6a4f;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 3px;
  }
  .msg.bot {
    background: white;
    color: #1a1a1a;
    align-self: flex-start;
    border: 1px solid #ddd;
    border-bottom-left-radius: 3px;
  }
  .msg.typing {
    background: #eee;
    color: #777;
    align-self: flex-start;
    font-style: italic;
  }

  #chat-suggestions {
    padding: 5px 8px;
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    background: white;
    border-top: 1px solid #eee;
    flex-shrink: 0;
  }
  .sug {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    color: #2d6a4f;
    padding: 3px 7px;
    border-radius: 10px;
    font-size: 11px;
    cursor: pointer;
  }
  .sug:hover { background: #c8e6c9; }

  #chat-input-row {
    display: flex;
    padding: 8px;
    gap: 6px;
    border-top: 1px solid #eee;
    background: white;
    flex-shrink: 0;
  }
  #chat-input {
    flex: 1;
    border: 1px solid #ccc;
    border-radius: 18px;
    padding: 6px 11px;
    font-size: 12.5px;
    outline: none;
  }
  #chat-input:focus { border-color: #52b788; }
  #chat-send {
    background: linear-gradient(135deg, #2d6a4f, #52b788);
    color: white;
    border: none;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    cursor: pointer;
    font-size: 13px;
    flex-shrink: 0;
  }

  #chat-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #2d6a4f, #52b788);
    color: white;
    border: none;
    border-radius: 28px;
    padding: 10px 18px;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(0,0,0,0.22);
    font-size: 14px;
    font-weight: bold;
    transition: transform 0.15s;
  }
  #chat-btn:hover { transform: scale(1.05); }
  #chat-btn-icon { font-size: 20px; }
  #chat-btn-label { font-size: 13px; letter-spacing: 0.3px; }
</style>
</head>
<body>

<div id="chat-wrapper">
  <div id="chat-panel">
    <div id="chat-header">
      <span>🤖 Green Space Assistant</span>
      <button id="chat-close" onclick="toggleChat()">✕</button>
    </div>
    <div id="chat-messages">
      <div class="msg bot">Hi! Ask me about your analysis results or anything about green spaces 🌱</div>
    </div>
    <div id="chat-suggestions">
      <button class="sug" onclick="suggest('How many trees can I plant?')">🌳 Trees?</button>
      <button class="sug" onclick="suggest('Is this land suitable?')">✅ Suitable?</button>
      <button class="sug" onclick="suggest('WHO standard met?')">🏥 WHO?</button>
      <button class="sug" onclick="suggest('What is my CO2 impact?')">🌿 CO2?</button>
    </div>
    <div id="chat-input-row">
      <input id="chat-input" type="text" placeholder="Ask something..." />
      <button id="chat-send" onclick="send()">➤</button>
    </div>
  </div>

  <button id="chat-btn" onclick="toggleChat()">
    <span id="chat-btn-icon">🌱</span>
    <span id="chat-btn-label">Chatbot</span>
  </button>
</div>

<script>
  const GROQ_KEY = \"""" + safe_key + """\";
  const SYS = `""" + safe_prompt + """`;
  let history = [];
  let open = false;

  function toggleChat() {
    open = !open;
    document.getElementById('chat-panel').style.display = open ? 'flex' : 'none';
    if (open) setTimeout(() => document.getElementById('chat-input').focus(), 100);
  }

  document.getElementById('chat-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') send();
  });

  function suggest(q) {
    document.getElementById('chat-input').value = q;
    send();
  }

  function addMsg(role, text) {
    const box = document.getElementById('chat-messages');
    const d = document.createElement('div');
    d.className = 'msg ' + role;
    d.textContent = text;
    box.appendChild(d);
    box.scrollTop = box.scrollHeight;
    return d;
  }

  async function send() {
    const inp = document.getElementById('chat-input');
    const msg = inp.value.trim();
    if (!msg) return;
    inp.value = '';
    addMsg('user', msg);
    history.push({role: 'user', content: msg});
    const typing = addMsg('typing', 'Thinking...');
    try {
      const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer ' + GROQ_KEY,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'llama-3.1-8b-instant',
          messages: [{role: 'system', content: SYS}, ...history],
          max_tokens: 300,
          temperature: 0.5
        })
      });
      const data = await res.json();
      typing.remove();
      const reply = (data.choices && data.choices[0])
        ? data.choices[0].message.content
        : 'Sorry, something went wrong.';
      addMsg('bot', reply);
      history.push({role: 'assistant', content: reply});
    } catch(e) {
      typing.remove();
      addMsg('bot', 'Connection error. Please try again.');
    }
  }
</script>
</body>
</html>
"""
    components.html(html_code, height=520, scrolling=False)


# ═══════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════
model, device = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=80)
    st.title("Green Space Optimizer")
    st.markdown("---")
    st.markdown("### Settings")
    tree_spacing = st.slider("Tree Spacing (px)", 20, 80, 40, 5)
    st.markdown("---")
    st.markdown("### GPS Location")
    use_gps = st.checkbox("Enable Location Features")
    lat = lon = location_name = None
    if use_gps:
        location_name = st.text_input("Site Name",
                                      placeholder="e.g. Anna Nagar, Chennai")
        col_a, col_b  = st.columns(2)
        lat = col_a.number_input("Latitude",  value=13.0827, format="%.4f")
        lon = col_b.number_input("Longitude", value=80.2707, format="%.4f")
        st.caption("Default: Chennai, Tamil Nadu")
    st.markdown("---")
    st.markdown("### About")
    st.caption("Deep Learning - U-Net\nRTX 3050 - Val IoU: 0.7640\n"
               "DeepGlobe + LoveDA dataset\nSRM Institute")

# ── HEADER ────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#1b4332,#2d6a4f);
            padding:1.5rem 2rem; border-radius:12px; margin-bottom:1rem'>
    <h1 style='color:white;margin:0'>🌱 Urban Green Space Optimizer</h1>
    <p style='color:#b7e4c7;margin:0.3rem 0 0 0'>
        AI-powered plantation planning using Deep Learning and Computer Vision
    </p>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload land image (JPG / PNG)",
                             type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.markdown("<div class='section-header'>📷 Uploaded Image</div>",
                unsafe_allow_html=True)
    st.image(image_rgb, width=700)

    st.info("""
**Image Guidelines for Accurate Results:**
- Aerial or drone photos (top-down view) work best
- Satellite images of open land
- Overhead ground photos of open plots
- Side-view or street-level photos will give inaccurate results
""")

    # ── WEATHER ───────────────────────────────────────────────────
    weather_score   = None
    weather_reasons = []

    if use_gps and lat and lon:
        st.markdown(
            "<div class='section-header'>🌦️ Weather and Soil Suitability</div>",
            unsafe_allow_html=True)
        with st.spinner("Fetching live weather and soil data..."):
            weather_data  = get_weather(lat, lon)
            soil_data     = get_soil(lat, lon)
            weather_score, weather_reasons = suitability_from_weather(
                weather_data, soil_data)
        if weather_data:
            c = weather_data.get("current", {})
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Temperature",   f"{c.get('temperature_2m','--')} C")
            w2.metric("Humidity",      f"{c.get('relative_humidity_2m','--')}%")
            w3.metric("Precipitation", f"{c.get('precipitation','--')} mm")
            w4.metric("Wind Speed",    f"{c.get('wind_speed_10m','--')} km/h")
        st.progress(weather_score / 100)
        st.markdown(f"**Site Suitability Score: {weather_score} / 100**")
        for r in weather_reasons:
            st.write(r)

    # ── ANALYSE BUTTON ────────────────────────────────────────────
    if st.button("🚀 Analyse Plantable Area", type="primary",
                 use_container_width=True):
        with st.spinner("Running AI analysis..."):
            mask         = predict_mask(model, device, image_rgb)
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

            heatmap         = generate_suitability_heatmap(mask_resized)
            heatmap_overlay = overlay_heatmap(image_rgb, heatmap)
            tree_positions  = generate_tree_positions(mask_resized, tree_spacing)
            tree_overlay    = draw_tree_positions(image_rgb.copy(), tree_positions)

            suitability_map = image_rgb.copy()
            suitability_map[mask_resized == 255] = [0, 200, 0]
            suitability_map[mask_resized == 0]   = [200, 0, 0]
            blended = cv2.addWeighted(image_rgb, 0.6, suitability_map, 0.4, 0)

            plantable_pixels, plantable_area = calculate_plantable_area(mask_resized)
            if plantable_area < 10:
                plantable_area = image.shape[0] * image.shape[1] * 0.02
            if plantable_area > 5000:
                plantable_area = plantable_area * 0.2

            plant_plan     = estimate_plants_by_category(plantable_area)
            sustainability = calculate_sustainability(
                plantable_area,
                plant_plan["small_plants"],
                plant_plan["shrubs"],
                plant_plan["trees"])
            urban_stats    = analyze_urban_density(mask_resized, image_rgb)
            sufficiency    = assess_greenery_sufficiency(
                urban_stats["green_percentage"], plantable_area)

            # ── SAVE ALL RESULTS TO SESSION STATE ─────────────────
            st.session_state.results = {
                "blended":          blended,
                "heatmap_overlay":  heatmap_overlay,
                "tree_overlay":     tree_overlay,
                "tree_positions":   tree_positions,
                "plantable_area":   plantable_area,
                "plant_plan":       plant_plan,
                "sustainability":   sustainability,
                "urban_stats":      urban_stats,
                "sufficiency":      sufficiency,
                "image_rgb":        image_rgb,
            }
            st.session_state.report_data = {
                "location":             location_name or "Not specified",
                "area":                 plantable_area,
                "trees":                plant_plan["trees"],
                "shrubs":               plant_plan["shrubs"],
                "small_plants":         plant_plan["small_plants"],
                "green_pct":            urban_stats["green_percentage"],
                "greenery_status":      sufficiency["status"],
                "greenery_message":     sufficiency["message"],
                "greenery_rec":         sufficiency["recommendation"],
                "density_class":        urban_stats["density_class"],
                "plantation_potential": urban_stats["plantation_potential"],
                "co2_kg":               sustainability["co2_kg"],
                "cooling":              sustainability["cooling_effect"],
                "sus_index":            sustainability["sustainability_index"],
                "weather_score":        weather_score,
                "weather_reasons":      weather_reasons,
                "who_standard":         sufficiency["who_standard"],
                "urban_standard":       sufficiency["urban_standard"],
                "built_up_pct":         urban_stats["built_up_percentage"],
            }

# ── RENDER RESULTS FROM SESSION STATE (persists across reruns) ────
if "results" in st.session_state:
    r       = st.session_state.results
    rd      = st.session_state.report_data
    blended          = r["blended"]
    heatmap_overlay  = r["heatmap_overlay"]
    tree_overlay     = r["tree_overlay"]
    tree_positions   = r["tree_positions"]
    plantable_area   = r["plantable_area"]
    plant_plan       = r["plant_plan"]
    sustainability   = r["sustainability"]
    urban_stats      = r["urban_stats"]
    sufficiency      = r["sufficiency"]
    image_rgb        = r["image_rgb"]

    # ── BEFORE / AFTER SLIDER ─────────────────────────────────
    st.markdown(
        "<div class='section-header'>🗺️ Before / After Comparison</div>",
        unsafe_allow_html=True)
    slider_val = st.slider("Original  |  Overlay", 0, 100, 50,
                           key="compare_slider")
    h, w    = image_rgb.shape[:2]
    split_x = int(w * slider_val / 100)
    comparison = image_rgb.copy()
    comparison[:, split_x:] = blended[:, split_x:]
    cv2.line(comparison, (split_x, 0), (split_x, h), (255, 255, 255), 2)
    st.image(comparison, width=700,
             caption="Left: Original  |  Right: Plantable overlay")

    # ── HEATMAP ───────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>🌡️ Suitability Heatmap</div>",
        unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    h1.image(heatmap_overlay, width=500, caption="AI Suitability Heatmap")
    h2.image(tree_overlay, width=500,
             caption=f"Tree Placement Map ({len(tree_positions)} locations)")
    st.markdown(
        "Green = Best planting zones  |  "
        "Yellow = Moderate  |  Red = Not suitable")

    # ── METRICS ───────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>📊 Analysis Results</div>",
        unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plantable Area",  f"{plantable_area:.0f} m2")
    c2.metric("Trees",           plant_plan["trees"])
    c3.metric("Shrubs",          plant_plan["shrubs"])
    c4.metric("Small Plants",    plant_plan["small_plants"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CO2/year",    f"{sustainability['co2_kg']:.0f} kg")
    c6.metric("Cooling",     f"{sustainability['cooling_effect']:.1f} C")
    c7.metric("Green Cover", f"{urban_stats['green_percentage']}%")
    c8.metric("Sus. Index",  f"{sustainability['sustainability_index']:.0f}/100")

    score = sustainability["sustainability_index"]
    badge_class = ("badge-high"   if score > 70 else
                   "badge-medium" if score > 40 else "badge-low")
    badge_text  = ("HIGH Sustainability Potential"   if score > 70 else
                   "MEDIUM Sustainability Potential" if score > 40 else
                   "LOW Sustainability Potential")
    st.markdown(f"<div class='{badge_class}'>{badge_text}</div>",
                unsafe_allow_html=True)

    # ── GREENERY SUFFICIENCY ──────────────────────────────────
    st.markdown(
        "<div class='section-header'>🌿 Greenery Assessment</div>",
        unsafe_allow_html=True)
    color_map = {
        "green":   "#1b5e20",
        "orange":  "#e65100",
        "red":     "#7f0000",
        "darkred": "#4a0000"
    }
    bg = color_map.get(sufficiency["color"], "#333333")
    st.markdown(f"""
<div style='background:{bg}; padding:16px; border-radius:10px;
            color:white; margin:8px 0'>
    <h3 style='margin:0 0 8px 0'>Greenery Status: {sufficiency["status"]}</h3>
    <p style='margin:0 0 6px 0'>{sufficiency["message"]}</p>
    <b>Recommendation:</b> {sufficiency["recommendation"]}
</div>
""", unsafe_allow_html=True)

    gs1, gs2 = st.columns(2)
    gs1.metric("WHO Standard (9 m2 per person)",
               "Met" if sufficiency["who_standard"]   else "Not Met")
    gs2.metric("Urban Planning Standard (20% green)",
               "Met" if sufficiency["urban_standard"] else "Not Met")

    # ── URBAN INTELLIGENCE ────────────────────────────────────
    st.markdown(
        "<div class='section-header'>🏙️ Urban Intelligence</div>",
        unsafe_allow_html=True)
    ui1, ui2, ui3 = st.columns(3)
    ui1.metric("Urban Density",        urban_stats["density_class"])
    ui2.metric("Built-up Area",        f"{urban_stats['built_up_percentage']}%")
    ui3.metric("Plantation Potential", urban_stats["plantation_potential"])

    # ── PDF DOWNLOAD ──────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>📄 Download Report</div>",
        unsafe_allow_html=True)
    with st.spinner("Generating PDF report..."):
        pdf_path = generate_pdf(rd, blended)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download Full PDF Report",
            data=f,
            file_name=f"green_space_report_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    os.unlink(pdf_path)
    st.success("✅ Analysis complete!")

    # ── FLOATING CHATBOT ──────────────────────────────────────
    inject_chatbot(rd, GROQ_API_KEY)