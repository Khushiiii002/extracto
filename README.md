# extracto — Invoice Intelligence

> Parse any invoice in seconds using Vision AI. Drop a file, get clean structured data instantly.

## 🌐 Live Demo
👉 **[Try it live](https://extracto-production.up.railway.app/)**

---

## ✨ What It Does

Upload any invoice as a JPG, PNG, or PDF and instantly extract:

| Field | Example |
|-------|---------|
| Invoice Number | INV-2024-0042 |
| Date | January 15, 2025 |
| Vendor Name | Acme Corporation |
| Customer Name | TechStart Inc. |
| Total Amount | $4,280.00 |
| Line Items | Description · Qty · Unit Price · Total |
| Notes | Net 30 payment terms |

---

## 🚀 Features

- ⚡ **Fast** — Results in under 5 seconds
- 🤖 **Vision AI** — Powered by Qwen2-VL-2B model
- 📁 **Multi-format** — Accepts JPG, PNG, and PDF up to 10MB
- 🌍 **Multi-currency** — Auto-detects USD, EUR, GBP, INR, JPY, AED and more
- 🔁 **Reliable** — 3x auto-retry with exponential backoff
- 🔒 **Safe** — File type validation, size limits, deduplication

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend & Proxy | FastAPI + Python 3.11 |
| Deployment | Railway (Docker) |
| AI Model | Qwen2-VL-2B Instruct |
| GPU Inference | Lightning AI — A10G GPU |
| Libraries | PyTorch · Transformers · httpx · pdf2image |
| File Handling | Pillow · python-multipart |

---

## 🏗️ Architecture
User (Browser)
↓ HTTPS
FastAPI Proxy (Railway)
→ validates file type & size
→ SHA-256 deduplication
→ 3x retry with backoff
↓ HTTPS
Lightning AI GPU Server
→ Qwen2-VL Vision Model
→ extracts all invoice fields
→ returns clean JSON
↓
Result displayed in UI

---

## 📂 Project Structure
extracto/
├── main.py          # FastAPI proxy server + frontend
├── server.py        # AI inference server (Qwen2-VL)
├── Dockerfile       # Docker config for Railway
├── requirements.txt # Python dependencies
└── static/
└── index.html   # Frontend UI

---

## ⚙️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Khushiiii002/Extracto.git
cd Extracto
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open in browser
http://localhost:8000
> ⚠️ Note: The AI inference server (`server.py`) requires a GPU and runs separately on Lightning AI.

---

## 👩‍💻 Made By

**Khushi** — [@Khushiiii002](https://github.com/Khushiiii002)

---

⭐ If you found this useful, give it a star!
