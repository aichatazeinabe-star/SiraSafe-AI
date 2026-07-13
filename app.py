"""
app.py — SiraSafe AI · Streamlit MVP
─────────────────────────────────────────────────────────────────────────────
Dual-Pipeline Human Trafficking Prevention Platform

Architecture:
  Pipeline 1 · Scan social-media screenshots → OCR → IBM Granite analysis
  Pipeline 2 · Centralized community blacklist query & management

IBM Call for Code 2026 · IBM Granite-7B · watsonx.ai · Mali / West Africa
─────────────────────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Internal components ───────────────────────────────────────────────────────
from components.ai_engine import analyze_with_ibm_granite
from components.database  import (
    add_to_blacklist,
    check_phone_number,
    get_all_records,
    get_total_count,
)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

APP_TITLE   = "SiraSafe AI"
APP_VERSION = "2.0.0-MVP"

LANGUAGES = ["Bambara", "Fulfulde", "French"]

# Audio file mapping: (language, risk_level) → filename in audio/ folder
AUDIO_FILES = {
    ("Bambara",  "danger"): "audio/danger_bambara.mp3.m4a",
    ("Bambara",  "safe"):   "audio/safe_bambara.mp3",
    ("Fulfulde", "danger"): "audio/danger_fulfulde.mp3",
    ("Fulfulde", "safe"):   "audio/safe_fulfulde.mp3",
    ("French",   "danger"): "audio/danger_french.mp3",
    ("French",   "safe"):   "audio/safe_french.mp3",
}

# Risk threshold above which we trigger the HIGH RISK alert
HIGH_RISK_THRESHOLD = 75


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title      = f"{APP_TITLE} — Human Trafficking Prevention",
    page_icon       = "🛡️",
    layout          = "wide",
    initial_sidebar_state = "expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Premium dark-mode design system
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp {
    background: radial-gradient(ellipse at top left, #0f0f1a 0%, #090910 60%, #0a0d12 100%) !important;
    color: #e8e8f0 !important;
}
.main .block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0c18 0%, #0f101e 100%) !important;
    border-right: 1px solid rgba(255,107,53,0.2) !important;
}

/* ── Hero banner ──────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, #1a0a2e 0%, #0d1535 50%, #0a1f20 100%);
    border: 1px solid rgba(255,107,53,0.35);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 0%, rgba(255,107,53,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #FF6B35, #FFD166, #06D6A0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.3rem !important;
}
.hero-sub { font-size: 0.88rem; color: #8888aa; margin: 0 !important; }
.hero-stats {
    display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px; padding: 0.2rem 0.8rem;
    font-size: 0.72rem; color: #aaaacc;
}

/* ── Cards ────────────────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.25s;
}
.card:hover { border-color: rgba(255,107,53,0.3); }
.card-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #FF6B35; margin-bottom: 0.6rem;
}

/* ── Risk banner ──────────────────────────────────────────── */
.risk-critical {
    background: linear-gradient(135deg, #3d0000, #1a0000);
    border: 2px solid #FF1744; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin: 1rem 0;
    animation: criticalPulse 2s ease-in-out infinite;
}
@keyframes criticalPulse {
    0%,100% { box-shadow: 0 0 20px rgba(255,23,68,0.4); }
    50%      { box-shadow: 0 0 45px rgba(255,23,68,0.75); }
}
.risk-title { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.4rem !important; font-weight: 800 !important; color: #FF5252 !important; margin: 0 0 0.5rem !important; }
.risk-sub   { font-size: 0.85rem; color: #ff8a80; }

/* ── Flag row ─────────────────────────────────────────────── */
.flag-item {
    display: flex; align-items: center; gap: 0.6rem;
    background: rgba(255,23,68,0.08);
    border: 1px solid rgba(255,23,68,0.25);
    border-radius: 8px; padding: 0.55rem 0.9rem;
    margin-bottom: 0.4rem; font-size: 0.83rem;
}

/* ── Blacklist table ──────────────────────────────────────── */
.blacklist-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; font-weight: 700; color: #FFD166;
    margin: 0.5rem 0;
}
.match-banner {
    background: linear-gradient(135deg, #3d1a00, #1a0d00);
    border: 2px solid #FF6D00; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin: 0.75rem 0;
}
.match-title { font-size: 1.1rem; font-weight: 700; color: #FF9100; margin: 0 0 0.3rem; }
.match-detail { font-size: 0.82rem; color: #ffcc80; }

/* ── Sidebar credits ──────────────────────────────────────── */
.sidebar-credit {
    background: rgba(255,107,53,0.08);
    border: 1px solid rgba(255,107,53,0.25);
    border-radius: 10px; padding: 0.75rem 1rem;
    margin-top: 1.5rem; font-size: 0.72rem;
    color: #FF6B35; text-align: center;
}

/* ── Streamlit widget tweaks ──────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #FF6B35, #FF4500) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(255,107,53,0.4) !important;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important; padding: 0.8rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ SiraSafe AI")
    st.markdown(f"<span style='font-size:0.72rem;color:#555577;'>v{APP_VERSION} · IBM Call for Code 2026</span>",
                unsafe_allow_html=True)
    st.divider()

    # ── Language selector ────────────────────────────────────────────────
    st.markdown("**🌍 Alert Language**")
    selected_language = st.selectbox(
        label     = "Language / Langue",
        options   = LANGUAGES,
        index     = 0,
        key       = "lang_selector",
        help      = "Safety alerts will be spoken in this language.",
        label_visibility = "collapsed",
    )
    lang_flags = {"Bambara": "🇲🇱", "Fulfulde": "🌍", "French": "🇫🇷"}
    st.markdown(
        f"<span style='font-size:0.8rem;color:#8888aa;'>"
        f"{lang_flags[selected_language]} Selected: <strong style='color:#e8e8f0'>"
        f"{selected_language}</strong></span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Offline / Edge AI toggle ─────────────────────────────────────────
    st.markdown("**📡 Deployment Mode**")
    offline_mode = st.toggle(
        "🔌 Offline Mode Simulation (Edge AI)",
        value = False,
        key   = "offline_toggle",
        help  = (
            "Simulates a compressed IBM Granite-3B model running locally "
            "on a smartphone's NPU — for rural Mali villages without internet."
        ),
    )
    if offline_mode:
        st.info(
            "**MVP Architecture Note:**  \n"
            "In production, this toggle activates local execution of a "
            "compressed **IBM Granite-3B** model on the smartphone's processor "
            "(Edge AI / NPU) for rural areas in Mali without internet connectivity.  \n"
            "For this cloud demo, API routing is used.",
            icon="⚡",
        )

    # ── watsonx.ai Connection / Simulation Selector ─────────────────────
    st.divider()
    st.markdown("**☁️ IBM watsonx.ai Connection**")
    
    import os
    
    # Dynamic check of the watsonx credentials (supports both with/without underscore for robust compatibility)
    raw_apikey = os.environ.get("WATSONX_APIKEY") or os.environ.get("WATSONX_API_KEY") or ""
    raw_project_id = os.environ.get("PROJECT_ID") or os.environ.get("WATSONX_PROJECT_ID") or ""
    
    watsonx_apikey = raw_apikey.strip()
    project_id = raw_project_id.strip()
    
    # Template placeholder values to ignore
    invalid_keywords = ["your_ibm_api_key_here", "your_project_id_here", "your-", "your_", "example", "placeholder"]
    
    def is_valid_cred(val):
        if not val:
            return False
        val_lower = val.lower()
        return not any(kw in val_lower for kw in invalid_keywords)
        
    has_watsonx_env = is_valid_cred(watsonx_apikey) and is_valid_cred(project_id)
    
    if has_watsonx_env:
        st.success("🟢 Live watsonx.ai API Active", icon="✅")
        sim_scenario = None
    else:
        st.info("ℹ️ Using local Watsonx.ai Vision Simulation", icon="🤖")
        sim_scenario = st.selectbox(
            "Simulated Scenario (Vision Mock)",
            options=[
                "Trafficking Attempt (Canada Job Ad)",
                "Trafficking Attempt (Dubai Domestic Worker Trap)",
                "Legitimate Local Freelance Offer",
                "Unrelated / Blank Image"
            ],
            key="sim_scenario_selector",
            help="Simulates the visual & textual understanding of Granite Vision on any uploaded image."
        )

    # ── Stats snapshot ───────────────────────────────────────────────────
    st.divider()
    st.markdown("**📊 Network Stats**")
    total = get_total_count()
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🚫 Flagged", total)
    with col_b:
        st.metric("👁️ Protected", f"{total * 340:,}")

    # ── IBM credit ───────────────────────────────────────────────────────
    st.markdown(
        """<div class="sidebar-credit">
        🤖 Powered by<br>
        <strong>IBM Granite · watsonx.ai</strong><br>
        <span style='font-size:0.65rem;opacity:0.7;'>ibm/granite-7b-instruct</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero-wrap">
    <h1 class="hero-title">🛡️ SiraSafe AI</h1>
    <p class="hero-sub">
        AI-Powered Human Trafficking Prevention · IBM Call for Code 2026<br>
        Dual-Pipeline: Social Media Scan · Community Blacklist · Edge AI
    </p>
    <div class="hero-stats">
        <span class="stat-pill">🤖 IBM Granite-7B</span>
        <span class="stat-pill">🇲🇱 Bambara · Fulfulde · Français</span>
        <span class="stat-pill">📱 Edge AI Ready</span>
        <span class="stat-pill">🌍 West Africa Focus</span>
        <span class="stat-pill">{"⚡ OFFLINE MODE" if offline_mode else "☁️ Cloud API"}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs([
    "📸  Scan Social Media Ad / Screenshot",
    "🗂️  Centralized Blacklist & Query",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCAN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

with tab1:

    st.markdown("""
    <div class="card">
        <p class="card-title">📌 Pipeline 1 · IBM Granite Vision Analysis</p>
        Upload a screenshot or paste text from a recruitment ad.
        The AI engine will analyze the content for human trafficking red flags,
        and automatically register the recruiter in the community blacklist.
    </div>
    """, unsafe_allow_html=True)

    # ── Input options (Image Upload or Direct Text Input) ──────────────────
    uploaded_file = st.file_uploader(
        label   = "📎 Upload Screenshot (WhatsApp / Facebook / Telegram ad)",
        type    = ["png", "jpg", "jpeg"],
        key     = "screenshot_upload",
        help    = "Upload a screenshot of a suspicious recruitment offer.",
    )

    st.markdown("<p style='text-align: center; color: #555577; margin: 0.5rem 0;'>— OR —</p>", unsafe_allow_html=True)

    manual_text = st.text_area(
        label       = "✍️ Paste Recruitment Text Directly",
        placeholder = "Paste the text of the job offer or message here to analyze it dynamically...",
        height      = 120,
        key         = "manual_text_input",
    )

    active_input = None
    if uploaded_file is not None:
        active_input = "image"
    elif manual_text.strip():
        active_input = "text"

    if active_input is not None:
        if active_input == "image":
            # ── Image preview + metadata ────────────────────────────────────
            col_img, col_meta = st.columns([1, 2], gap="medium")
            with col_img:
                st.image(uploaded_file, caption="📎 Uploaded Screenshot", use_container_width=True)
            with col_meta:
                st.markdown(f"""
                <div class="card">
                    <p class="card-title">📋 File Details</p>
                    <p style="font-size:0.82rem;color:#aaaacc;margin:0.15rem 0;">
                        📁 <strong style="color:#e8e8f0;">{uploaded_file.name}</strong>
                    </p>
                    <p style="font-size:0.82rem;color:#aaaacc;margin:0.15rem 0;">
                        📦 Size: {uploaded_file.size / 1024:.1f} KB
                    </p>
                    <p style="font-size:0.82rem;color:#aaaacc;margin:0.15rem 0;">
                        🌍 Language: {lang_flags[selected_language]} {selected_language}
                    </p>
                    <p style="font-size:0.82rem;color:#aaaacc;margin:0.15rem 0;">
                        🤖 Engine: {"📱 Edge AI (Simulated)" if offline_mode else "☁️ IBM Granite Vision (Cloud)"}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✍️ Analyzing manually pasted text...", icon="📝")

        st.divider()

        # ── AI Processing ───────────────────────────────────────────────
        with st.spinner("🔍 Processing with IBM Granite Vision & Threat Engine..."):
            if active_input == "image":
                sim_val = st.session_state.get("sim_scenario_selector", None)
                analysis: dict = analyze_with_ibm_granite(image_file=uploaded_file, sim_scenario=sim_val)
            else:
                analysis: dict = analyze_with_ibm_granite(text=manual_text)

        # ── Unpack dynamic results — all values come from the AI engine ──
        risk_score      = analysis["risk_score"]
        phone_detected  = analysis["phone_number_detected"]  # "N/A" if not in text
        flags_dict      = analysis["flags"]                   # {flag_key: bool}
        verdict_english = analysis["verdict_english"]

        # Convenience list of triggered flag display names (for rendering)
        _FLAG_LABELS = {
            "hidden_fees":        "Hidden Upfront Fees",
            "passport_retention": "Passport / Document Retention",
            "vague_destination":  "Vague / Unspecified Destination",
            "urgency_tactics":    "Extreme Urgency / Pressure Tactics",
            "unrealistic_salary": "Unrealistic / Implausible Salary",
        }
        flags_triggered = [_FLAG_LABELS[k] for k, v in flags_dict.items() if v]

        # ── Check if we got a visual fallback warning ────────────────────
        if active_input == "image" and "No text or threats could be extracted" in verdict_english:
            st.warning(
                f"⚠️ **{verdict_english}**",
                icon="⚠️",
            )
            st.stop()

        st.divider()

        # ══════════════════════════════════════════════════════════════
        # RISK VERDICT — HIGH RISK path
        # ══════════════════════════════════════════════════════════════

        if risk_score >= HIGH_RISK_THRESHOLD:

            # ── Big red alert banner ────────────────────────────────────
            st.markdown(f"""
            <div class="risk-critical">
                <p class="risk-title">🚨 HIGH RISK TRAFFICKING ATTEMPT DETECTED</p>
                <p class="risk-sub">
                    IBM Granite confidence: <strong>{risk_score}/100</strong> ·
                    {len(flags_triggered)} of 5 red flags triggered ·
                    Recruiter registered in community blacklist
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Also display native Streamlit error for accessibility ───
            st.error(
                f"🚨 **HIGH RISK TRAFFICKING ATTEMPT DETECTED** — "
                f"Risk Score: **{risk_score}/100** · Flags: **{len(flags_triggered)}/5**"
            )

        else:
            # ── Low / medium risk path ──────────────────────────────────
            if risk_score >= 45:
                st.warning(f"⚠️ **MEDIUM RISK** — Score: {risk_score}/100 · {len(flags_triggered)} flag(s) detected.")
            else:
                st.success(f"✅ **LOW RISK** — Score: {risk_score}/100 · No strong trafficking patterns detected.")

        # ── Metric strip ────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Risk Score",   f"{risk_score} / 100")
        with m2: st.metric("Flags Found",  f"{len(flags_triggered)} / 5")
        with m3: st.metric("Phone Number", phone_detected)
        with m4: st.metric("AI Model",     "Granite-7B")

        # ── Flags detected ──────────────────────────────────────────────
        if flags_triggered:
            st.markdown("""
            <p class="card-title" style="margin-top:1rem;">🚩 Red Flags Detected</p>
            """, unsafe_allow_html=True)
            for flag_name in flags_triggered:
                st.markdown(f"""
                <div class="flag-item">
                    ⛔ <strong>{flag_name}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <p class="card-title" style="margin-top:1rem;">✅ No Red Flags</p>
            """, unsafe_allow_html=True)

        # ── Verdict text ─────────────────────────────────────────────────
        st.markdown(f"""
        <div class="card" style="margin-top:1rem; border-color: rgba(255,214,0,0.2);">
            <p class="card-title">📋 IBM Granite Verdict (English — for IBM Jury)</p>
            <p style="font-size:0.88rem; line-height:1.7; margin:0;">{verdict_english}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Raw JSON expander ────────────────────────────────────────────
        with st.expander("🔩 Raw IBM Granite JSON Response"):
            # Show only the public schema fields (strip internal _ prefixed keys)
            st.json({k: v for k, v in analysis.items() if not k.startswith("_")})

        st.divider()

        # ── Auto-add to blacklist ────────────────────────────────────────
        if phone_detected not in ("N/A", "None", "") and risk_score >= HIGH_RISK_THRESHOLD:
            scam_reason = ", ".join(flags_triggered) if flags_triggered else "Suspicious recruitment ad"
            db_record   = add_to_blacklist(phone_detected, scam_reason)
            st.success(
                f"✅ **Blacklist Updated** — `{phone_detected}` has been automatically "
                f"registered in the SiraSafe community database.  \n"
                f"This number now has **{db_record['reports']} report(s)** · "
                f"Status: **{db_record['status']}**"
            )

        # ── Audio alert section ──────────────────────────────────────────
        st.markdown("""
        <div class="card" style="border-color: rgba(6,214,160,0.3); margin-top:0.5rem;">
            <p class="card-title">🎙️ Spoken Alert · Écouter l'alerte</p>
        </div>
        """, unsafe_allow_html=True)

        risk_level  = "danger" if risk_score >= HIGH_RISK_THRESHOLD else "safe"
        audio_key   = (selected_language, risk_level)
        audio_path  = Path(AUDIO_FILES.get(audio_key, ""))

        if audio_path.exists():
            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/m4a")
        else:
            # Clean fallback — no crash
            st.markdown(f"""
            <div style="background:rgba(255,214,0,0.08); border:1px solid rgba(255,214,0,0.3);
                        border-radius:10px; padding:1rem 1.2rem;">
                <p style="margin:0; font-size:0.83rem; color:#FFD600;">
                    🎙️ <strong>Audio file not found:</strong>
                    <code>{audio_path or AUDIO_FILES.get(audio_key, "N/A")}</code><br>
                    <span style="color:#8888aa; font-size:0.75rem;">
                        Place <strong>{Path(AUDIO_FILES.get(audio_key,"x")).name}</strong>
                        in the <code>audio/</code> folder to enable spoken alert playback.<br>
                        In production, IBM Watson Text-to-Speech synthesises this alert
                        in {selected_language} in &lt;200 ms via Edge AI.
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Show the spoken alert text as a fallback
            alert_texts = {
                ("Bambara",  "danger"): "⚠️ Kunnafoni: Sɛbɛn nin bɛ gafe dɔ yira. Ɲɛ a la!",
                ("Bambara",  "safe"):   "✅ Sɛbɛn nin bɛ ɲuman kɔ. Hali tɔ, i ka jate.",
                ("Fulfulde", "danger"): "⚠️ Xatooji mawndi yiyaama e takko ngoo!",
                ("Fulfulde", "safe"):   "✅ Takko ngoo hollaaki xatooji mawndi.",
                ("French",   "danger"): "⚠️ ALERTE: Ce message présente un risque critique de traite humaine!",
                ("French",   "safe"):   "✅ Aucun signal d'alarme critique détecté dans ce message.",
            }
            spoken = alert_texts.get(audio_key, "⚠️ Audio alert unavailable.")
            st.info(f"📢 **Spoken Alert Preview** ({selected_language})\n\n{spoken}")

    else:
        # ── Idle state ───────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; opacity:0.55;">
            <span style="font-size:5rem;">📱</span>
            <p style="font-size:1rem; color:#8888aa; margin-top:1rem;">
                Upload a screenshot of a suspicious WhatsApp or Facebook<br>
                recruitment offer to start the AI analysis.<br>
                <small>Supported formats: PNG · JPG · JPEG</small>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BLACKLIST & QUERY
# ════════════════════════════════════════════════════════════════════════════

with tab2:

    st.markdown("""
    <div class="card">
        <p class="card-title">🗂️ Pipeline 2 · Centralized Community Blacklist</p>
        The SiraSafe AI community network — every detected fraudulent recruiter
        is registered here, creating a collective safety shield for all users
        across Mali and West Africa.
    </div>
    """, unsafe_allow_html=True)

    # ── Manual phone number search ───────────────────────────────────────
    st.markdown("#### 🔎 Verify a Recruiter Phone Number")

    col_input, col_btn = st.columns([4, 1], gap="small")
    with col_input:
        search_phone = st.text_input(
            label       = "Phone Number",
            placeholder = "e.g. +22376543210 or 76543210",
            key         = "blacklist_search",
            label_visibility = "collapsed",
        )
    with col_btn:
        do_search = st.button("🔍 Check", use_container_width=True, key="search_btn")

    # ── Search result ────────────────────────────────────────────────────
    if search_phone and (do_search or search_phone):
        result = check_phone_number(search_phone)

        if result is not None:
            # ── FOUND — danger warning ──────────────────────────────────
            st.markdown(f"""
            <div class="match-banner">
                <p class="match-title">
                    🚨 BLACKLISTED NUMBER — DO NOT CONTACT
                </p>
                <div class="match-detail">
                    📞 <strong>{result['phone_number']}</strong><br>
                    🏷️ Scam Type: <strong>{result['scam_type']}</strong><br>
                    📅 First Flagged: {result['date_flagged']}<br>
                    📊 Reports: <strong>{result['reports']}</strong> community reports<br>
                    🔴 Status: <strong>{result['status']}</strong><br>
                    🤖 Source: {result['added_by']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.error(
                f"🚨 **DANGER** — `{result['phone_number']}` is confirmed in the "
                f"SiraSafe blacklist as: **{result['scam_type']}**  \n"
                f"Do **NOT** send money or documents to this number."
            )
        else:
            if search_phone.strip():
                st.success(
                    f"✅ **`{search_phone}`** is **NOT** in the SiraSafe blacklist.  \n"
                    "This number has not been reported by the community — "
                    "always remain vigilant and verify job offers through official channels."
                )

    st.divider()

    # ── Full blacklist dataframe ─────────────────────────────────────────
    all_records = get_all_records()
    total_count = get_total_count()

    st.markdown(f"""
    <p class="blacklist-header">
        🌍 Active Blacklist — {total_count} Confirmed Fraudulent Recruiters
    </p>
    <p style="font-size:0.8rem; color:#8888aa; margin-bottom:1rem;">
        Community network effect: each entry protects an estimated 340+ potential
        victims across West Africa. Real-time sync powered by IBM Cloud.
    </p>
    """, unsafe_allow_html=True)

    if all_records:
        # Build display DataFrame
        df = pd.DataFrame(all_records)

        # Rename columns for jury presentation
        df = df.rename(columns={
            "phone_number":  "📞 Phone Number",
            "scam_type":     "🏷️ Scam Type",
            "date_flagged":  "📅 Date Flagged",
            "reports":       "📊 Reports",
            "status":        "🔴 Status",
            "added_by":      "🤖 Source",
        })

        # Style the dataframe
        st.dataframe(
            df,
            use_container_width = True,
            hide_index          = True,
            column_config       = {
                "📊 Reports": st.column_config.NumberColumn(
                    "📊 Reports",
                    help   = "Number of independent community reports",
                    format = "%d ⚠️",
                ),
                "🔴 Status": st.column_config.TextColumn(
                    "🔴 Status",
                    help = "CONFIRMED = verified by multiple sources",
                ),
            },
        )

        # ── Community impact metrics ────────────────────────────────────
        st.divider()
        st.markdown("#### 📊 Community Network Impact")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🚫 Flagged Numbers", total_count)
        with c2:
            st.metric("👁️ Estimated Protected", f"{total_count * 340:,}")
        with c3:
            confirmed = sum(1 for r in all_records if r["status"] == "CONFIRMED")
            st.metric("✅ Confirmed Entries", confirmed)
        with c4:
            total_reports = sum(r["reports"] for r in all_records)
            st.metric("📋 Total Reports", total_reports)

    else:
        st.info("The blacklist is currently empty. Upload a screenshot in Tab 1 to start building the community safety network.")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pass   # Streamlit runs via `streamlit run app.py`
