from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from PIL import Image
from pdf2image import convert_from_bytes
import torch
import io
import json
import threading

app = FastAPI()

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    # Use flash attention if your GPU supports it (A10/A100/H100)
    # attn_implementation="flash_attention_2",
)
model.eval()

# Compile model graph once at startup for faster repeated inference
# Uncomment if on PyTorch 2.0+ and want ~20% faster inference after warmup
# model = torch.compile(model, mode="reduce-overhead")

PROMPT = """You are an expert invoice and freight document extraction system with advanced OCR capabilities.

Your job is to scan the ENTIRE document image systematically and extract complete structured data.

Return ONLY valid JSON. No explanation, no markdown fences, no preamble.

════════════════════════════════════════
SCAN ORDER — MANDATORY
════════════════════════════════════════
Scan in this exact sequence before extracting:
  1. TOP-LEFT   → Company logo, vendor name, letterhead
  2. TOP-RIGHT  → Invoice number, date, reference codes
  3. TOP-CENTER → Document title (Invoice / Tax Invoice / Receipt / Bill of Lading)
  4. MIDDLE     → Line items table, service descriptions
  5. RIGHT MARGIN → Barcodes, QR codes, shipper/container codes
  6. BOTTOM     → Totals, tax breakdown, amount due
  7. FOOTER     → Payment terms, banking details, contact info

════════════════════════════════════════
VENDOR NAME — LOGO EXTRACTION RULES
════════════════════════════════════════
- PRIMARY: Read the largest, most prominent text at the top of the document
- LOGO TEXT: If a graphic logo is present, read the text embedded within or directly adjacent to it
- LETTERHEAD: Brand name printed in header area counts as vendor name even if styled decoratively
- TAGLINES: Ignore slogans ("Since 1990", "Quality You Trust") — use only the company name
- REGISTRATION: Ignore company numbers/tax IDs — those are not the vendor name
- MULTI-LINE: If name spans two lines (e.g. "GLOBAL\nLOGISTICS LTD"), join them
- FALLBACK: If logo is purely graphical with no readable text, return null — do NOT guess

════════════════════════════════════════
REFERENCE NUMBER — FREIGHT & TRANSACTION IDs
════════════════════════════════════════
Look for ALL of the following label types, in priority order:
  HIGHEST:  BOL No / Bill of Lading No / B/L No
            Freight No / Freight Order No
            Container No / CNTR No
            Shipper No / Consignment No
  HIGH:     PO No / Purchase Order No / Order ID
            Receipt No / Voucher No
  MEDIUM:   Job No / Booking Ref / Shipment Ref
            Delivery Note No / DN No
  LOW:      Customer Ref / Your Ref / Our Ref

Rules:
  - NEVER use Invoice Number as reference_number — they are separate fields
  - If multiple references exist, pick the highest-priority one from the list above
  - If two references share equal priority, pick the most visually prominent
  - Strip labels — return the VALUE only (e.g. "BOL-2024-00891", not "BOL No: BOL-2024-00891")

════════════════════════════════════════
CURRENCY DETECTION (CRITICAL)
════════════════════════════════════════
Detect currency from these sources in order:
  1. Explicit symbol before/after numbers: $, £, €, ¥, ₹, ₦, ₩, ฿, RM, AED, SAR, etc.
  2. Currency code label: "USD", "GBP", "EUR", "INR", "AED", "SGD", "MYR", etc.
  3. Country/region context: vendor address, customer address, bank account details
  4. Document header or footer stating "All amounts in [currency]"

Rules:
  - Return the ISO 4217 currency code (USD, GBP, EUR, INR, etc.)
  - If mixed currencies exist, use the currency of the final total amount
  - If genuinely indeterminate, return null — do NOT default to USD
  - Store detected currency in the "currency" field
  - All numeric amount fields (unit_price, total, total_amount) must be raw numbers — strip currency symbols

════════════════════════════════════════
LINE ITEMS — TABLE EXTRACTION
════════════════════════════════════════
Line items appear as a TABLE in the middle section. Extract EVERY row.

Column mapping (labels vary — match by meaning):
  description → Item, Description, Product, Service, Particulars, Narration
  qty         → Qty, Quantity, Units, Pcs, No., Nos
  unit_price  → Unit Price, Rate, MRP, Price Each, Per Unit
  total       → Amount, Line Total, Ext. Price, Net Amount

Rules:
  - Extract ALL visible rows without skipping
  - If table borders are missing, infer rows by vertical alignment
  - If a column is missing entirely, use null for every item in that column
  - NEVER default missing numeric values to 0
  - Subtotal rows, tax rows, and discount rows → do NOT include in line_items
  - If a row spans multiple lines (long description), merge into one description string
  - Preserve the order of rows as they appear top-to-bottom

════════════════════════════════════════
TOTAL AMOUNT
════════════════════════════════════════
Look ONLY in the bottom section. Match these labels (descending priority):
  Grand Total > Total Due > Amount Due > Balance Due > Net Payable > Total Payable > Total

Rules:
  - Always pick the LARGEST / FINAL amount — this is what the customer owes
  - Ignore subtotals, pre-tax amounts, and line totals
  - Prioritize bold, boxed, underlined, or right-aligned numbers
  - Return raw number only — no currency symbols
  - NEVER return 0 unless that value is explicitly printed and makes contextual sense

════════════════════════════════════════
TAX & CHARGES
════════════════════════════════════════
Extract any tax or additional charge lines visible in the totals section:
  - Label: "GST", "VAT", "Tax", "Service Charge", "Freight Charge", "Handling Fee", etc.
  - Return as a list of { "label": "...", "amount": ... } objects
  - If none visible, return empty array []

════════════════════════════════════════
DATES
════════════════════════════════════════
  - invoice_date: The primary document date (labeled "Date", "Invoice Date", "Issue Date")
  - due_date: Payment deadline (labeled "Due Date", "Payment Due", "Pay By")
  - Return dates exactly as printed — do not reformat

════════════════════════════════════════
NOTES — MEANINGFUL CONTENT ONLY
════════════════════════════════════════
Include ONLY business-critical information:
  ✓ Bank details / payment instructions (account no, IFSC, SWIFT, IBAN)
  ✓ Payment terms (Net 30, COD, advance required)
  ✓ Tax registration numbers (GSTIN, VAT No, TRN, EIN)
  ✓ Return / refund policy
  ✓ Delivery / shipping instructions
  ✓ Contact for disputes or queries
  ✓ Late payment penalties or early payment discounts
  ✓ Special terms, warranties, or service conditions

Exclude:
  ✗ Thank you messages ("Thank you for your business")
  ✗ Greetings or pleasantries
  ✗ Branding slogans or mission statements
  ✗ Decorative or legal boilerplate with no actionable content

If no meaningful notes exist, return null.

════════════════════════════════════════
GENERAL EXTRACTION RULES
════════════════════════════════════════
- Do NOT guess values that are completely invisible
- If partially visible or obscured, extract best possible reading
- Use null only when there is absolutely no visual evidence
- Preserve original formatting of IDs and codes (uppercase, hyphens, slashes)
- For addresses: combine multi-line addresses into a single comma-separated string

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════
{
  "invoice_number": "string or null",
  "invoice_date": "string or null",
  "due_date": "string or null",
  "currency": "ISO 4217 code or null",
  "reference_number": "string or null",
  "reference_type": "e.g. BOL No / PO No / Container No / Receipt No — or null",
  "vendor_name": "string or null",
  "vendor_address": "string or null",
  "vendor_tax_id": "string or null",
  "customer_name": "string or null",
  "customer_address": "string or null",
  "customer_tax_id": "string or null",
  "line_items": [
    {
      "description": "string or null",
      "qty": null,
      "unit_price": null,
      "total": null
    }
  ],
  "tax_and_charges": [
    {
      "label": "string",
      "amount": null
    }
  ],
  "total_amount": null,
  "note": "string or null"
}

Return ONLY JSON. No extra text."""



def load_image(file_bytes: bytes, content_type: str) -> Image.Image:
    if content_type == "application/pdf":
        # Lower DPI (150 vs 200) — still readable, ~2x faster PDF conversion
        pages = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=1)
        return pages[0].convert("RGB")
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    # Resize large images — VLM doesn't need >1280px on longest side
    max_side = 1280
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _run_inference(inputs) -> str:
    with torch.no_grad():
        # max_new_tokens=384 (was 512) — JSON invoices rarely need more
        # do_sample=False — greedy, faster and deterministic
        # use_cache=True  — KV cache reuse across decode steps
        output_ids = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _parse_json(result: str):
    # Strip markdown fences if present
    if result.startswith("```"):
        parts = result.split("```")
        result = parts[1] if len(parts) > 1 else result
        if result.startswith("json"):
            result = result[4:]
        result = result.strip()

    start = result.find("{")
    end   = result.rfind("}") + 1
    if start == -1 or end == 0:
        return {"error": "Model did not return valid JSON", "raw": result}

    try:
        return json.loads(result[start:end])
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}", "raw": result[start:end]}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    file_bytes = await file.read()

    if not file_bytes:
        return {"error": "Uploaded file is empty"}

    try:
        image = load_image(file_bytes, file.content_type)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": PROMPT}
        ]
    }]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # pin_memory=False avoids extra copy for GPU tensors
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(model.device)

        result = _run_inference(inputs)
        return _parse_json(result)

    except Exception as e:
        return {"error": f"Inference failed: {e}"}


# Warm-up endpoint — call this once after deploy to pre-load CUDA kernels
# so the very first real request isn't slow
@app.on_event("startup")
async def warmup():
    dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
    msgs  = [{"role": "user", "content": [
        {"type": "image", "image": dummy},
        {"type": "text",  "text": "say hi"}
    ]}]
    text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, _ = process_vision_info(msgs)
    inp = processor(text=[text], images=img_in, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=5, do_sample=False)


@app.get("/health")
def health():
    return {"status": "ok"}
