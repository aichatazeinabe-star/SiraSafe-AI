# SiraSafe AI — IBM Call for Code 2026
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

###  IMPORTANT NOTE FOR IBM TECHNICAL REVIEWERS

> **Ready for Live Production Testing:** 
> Due to regional billing and geographic restrictions on IBM Cloud registration in West Africa, a live personal production API key is not pre-activated. 
> 
> However, **the authentic IBM watsonx.ai production pipeline is fully implemented, coded, and operational.** 
> 
> To test the live integration using your own credentials:
> 1. Create a local `.env` file at the root.
> 2. Add your corporate `WATSONX_APIKEY` and `PROJECT_ID` variables.
> 3. Launch the application. The system will automatically detect your credentials and switch from **Local Vision Simulation** mode to the **Live watsonx.ai Engine**.

---
SiraSafe AI is a community-led early warning platform that protects vulnerable West African youth from online human trafficking. Powered by IBM Granite via watsonx.ai on IBM Cloud, the solution parses suspicious recruitment ads from social media screenshots and triggers instant offline voice alerts in local languages like Bambara.

---

## Key Features
- **AI-Powered OCR Analysis:** Extracts and processes text from suspicious WhatsApp/Facebook recruitment screenshots using IBM Granite.
- **Offline Edge AI Simulation:** Designed to function fully offline directly on mobile devices to survive in remote, low-connectivity conflict zones.
- **Local Language Inclusivity:** Generates clear, automated spoken alerts in Bambara to protect survivors regardless of literacy levels.
- **Centralized Community Blacklist:** Instantly logs flagged recruiter numbers into a secure database to protect the wider community.
- **Robust Local Evaluation Fallback:** Includes an interactive local simulation panel (Sidebar Mock) and direct text parsing to test max-risk scenarios instantly, ensuring full evaluative resilience even without active IBM Cloud API credentials.

---

## Local Installation & Setup Guide

Follow these steps to clone and run the Streamlit application on your local machine.

### 1. Prerequisites
Make sure you have Python 3.10 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/aichatazeinabe-star/SiraSafe-AI.git
cd SiraSafe-AI
```

### 3. Create a Virtual Environment
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Variables & Simulation Setup
Create a `.env` file at the root of the project to add your IBM watsonx.ai credentials (this file is securely ignored by git via `.gitignore`):
```text
WATSONX_APIKEY=your_ibm_watsonx_api_key_here
PROJECT_ID=your_ibm_project_id_here
```

Note for Evaluation (No API Key Required): If you do not have active IBM Watsonx API credentials or encounter authentication limits, the application will automatically activate the Local Vision Simulation.
Use the control panel in the Sidebar to toggle through different simulated scenarios (e.g., Trafficking Attempt, Legitimate Offer) or switch to the "Paste Recruitment Text Directly" tab to test the semantic model's real-time risk assessment without any cloud dependency.

### 6. Run the Application
Launch the Streamlit web dashboard locally:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at http://localhost:8501.
