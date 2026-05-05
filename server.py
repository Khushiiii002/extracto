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

PROMPT = """You are an expert invoice OCR extraction system. Your ONLY job is to read the image carefully and return a JSON object.

CRITICAL FAILURE MODES TO AVOID:
- Returning null for fields that ARE visible in the image
- Missing the date because it is labeled differently (e.g. "Dated", "Bill Date", "Tax Date")
- Missing the total because it appears at the bottom of a long document
- Skipping line items because the table has no visible borders
- Leaving notes empty when payment or banking details are present

════════════════════════════════════════
STEP 1 — FULL IMAGE SCAN (DO THIS FIRST)
════════════════════════════════════════
Before extracting anything, scan these 7 zones in order:

  ZONE 1 → TOP-LEFT     : Logo, company name, letterhead
  ZONE 2 → TOP-RIGHT    : Invoice no, date, reference codes, PO number
  ZONE 3 → TOP-CENTER   : Document title (Invoice / Receipt / Tax Invoice / Bill of Lading)
  ZONE 4 → LEFT BLOCK   : Bill To / Ship To — customer name and address
  ZONE 5 → MIDDLE TABLE : Line items — every row in the table
  ZONE 6 → BOTTOM RIGHT : Subtotal, tax, grand total, amount due, balance due
  ZONE 7 → FOOTER       : Payment terms, bank details, tax numbers, contact info

Do NOT skip any zone, even if the document looks simple.

════════════════════════════════════════
STEP 2 — DATE EXTRACTION (COMMONLY MISSED)
════════════════════════════════════════
The date may appear under ANY of these labels — check all:
  "Date", "Invoice Date", "Bill Date", "Tax Date", "Dated", "Issue Date",
  "Document Date", "Billing Date", "Date of Issue", "Dt."

Location: Almost always in ZONE 2 (top-right) or directly under the invoice number.

Rules:
  - Return the date EXACTLY as printed — do not reformat
  - If multiple dates exist, prefer "Invoice Date" or "Bill Date"
  - due_date is separate — look for "Due Date", "Payment Due", "Pay By", "Valid Till"
  - NEVER return null if any date is visible anywhere on the document

════════════════════════════════════════
STEP 3 — LINE ITEMS EXTRACTION (COMMONLY MISSED)
════════════════════════════════════════
Line items are a TABLE in ZONE 5. The table may or may not have visible borders.

Even if there are NO borders, rows can be identified by:
  - Consistent vertical alignment of numbers
  - Alternating white/grey background rows
  - Items listed one per line with amounts on the right

Column headers vary — map them like this:
  description  <- "Description", "Item", "Particulars", "Product", "Service", "Narration", "Details"
  qty          <- "Qty", "Quantity", "Units", "Nos", "Pcs", "No."
  unit_price   <- "Rate", "Unit Price", "Price", "MRP", "Per Unit", "Price Each"
  total        <- "Amount", "Total", "Line Total", "Ext. Price", "Net", "Value"

MANDATORY RULES:
  - Extract EVERY data row — do not stop after the first row
  - Do NOT skip rows that have long descriptions
  - Do NOT merge multiple items into one
  - Subtotal / Tax / Discount rows → exclude from line_items array
  - If a column is absent, use null — NEVER use 0 as a substitute for missing data
  - If the table has 10 rows, your line_items array must have 10 objects

════════════════════════════════════════
STEP 4 — TOTAL AMOUNT EXTRACTION (COMMONLY MISSED)
════════════════════════════════════════
The total is ALWAYS in ZONE 6 (bottom-right of the document).

Scan for these labels (pick the highest match):
  1st priority → "Grand Total", "Total Due", "Amount Due", "Net Payable", "Balance Due", "Total Payable"
  2nd priority → "Total", "Invoice Total", "Final Amount", "Net Amount"
  3rd priority → The largest right-aligned number in the bottom section

Visual clues:
  - It is usually BOLD or in a larger font
  - It may be BOXED or have a border around it
  - It is the LAST number in the totals section
  - It is the LARGEST of all the totals shown

Rules:
  - Return raw number only — strip all currency symbols, remove commas
  - NEVER return 0 unless 0 is explicitly and clearly printed as the final total
  - NEVER return null if any total number is visible at the bottom of the document
  - If tax is added, total_amount must INCLUDE tax (it is the final payable amount)

════════════════════════════════════════
STEP 5 — CURRENCY DETECTION
════════════════════════════════════════
Detect from (in order of priority):
  1. Symbol next to amounts: $, £, €, ₹, ¥, ₦, ₩, ฿, RM, AED, SAR, S$, HK$
  2. Currency code in header/footer: "USD", "GBP", "EUR", "INR", "AED", "SGD", "MYR"
  3. Vendor or customer country from address
  4. Bank account currency in payment details

Return the ISO 4217 code (USD / GBP / EUR / INR / AED / SGD etc.)
If genuinely unclear, return null. Do NOT default to USD.

════════════════════════════════════════
STEP 6 — VENDOR NAME FROM LOGO
════════════════════════════════════════
  - Read the largest text at the top of the document
  - If a logo image exists, read any text printed inside it or directly next to it
  - Decorative letterhead text = vendor name
  - Ignore: taglines, slogans, registration numbers, tax IDs
  - Multi-line name → join into one string
  - Purely graphical logo with zero readable text → return null (do not guess)

════════════════════════════════════════
STEP 7 — REFERENCE NUMBER
════════════════════════════════════════
Priority order (pick highest available):
  TIER 1: BOL No / Bill of Lading / B/L No / Freight No / Container No / Consignment No / Shipper No
  TIER 2: PO No / Purchase Order / Order ID / Receipt No / Voucher No
  TIER 3: Job No / Booking Ref / Shipment Ref / Delivery Note No
  TIER 4: Customer Ref / Your Ref / Our Ref

  - NEVER use Invoice Number as reference_number
  - Return the value only — strip the label
  - Also return reference_type (e.g. "BOL No", "PO No", "Container No")

════════════════════════════════════════
STEP 8 — TAX AND CHARGES
════════════════════════════════════════
In ZONE 6, extract any lines between subtotal and grand total:
  e.g. GST, VAT, IGST, CGST, SGST, Service Tax, Freight Charge, Handling Fee, Discount

Return as array: [{ "label": "GST 18%", "amount": 540.00 }]
If none exist, return []

════════════════════════════════════════
STEP 9 — NOTES (MEANINGFUL CONTENT ONLY)
════════════════════════════════════════
Scan ZONE 7 (footer) carefully. Extract ONLY these types of content:

  INCLUDE:
  - Bank name, account number, IFSC, SWIFT, IBAN, routing number
  - Payment instructions ("Pay within 30 days", "NEFT/RTGS to above account")
  - Payment terms ("Net 30", "50% advance, 50% on delivery", "COD")
  - Tax registration: GSTIN, VAT No, TRN, PAN, EIN, CRN
  - Late payment fee or early payment discount
  - Return, refund, or warranty policy
  - Delivery or shipping instructions
  - Dispute or query contact (email, phone)
  - Any other terms with financial or operational impact

  EXCLUDE:
  - "Thank you for your business" and similar pleasantries
  - Greetings, sign-offs, warm wishes
  - Marketing taglines or brand slogans
  - Generic boilerplate with no actionable meaning

If meaningful content exists → return it as a clean string.
If only thank-you messages exist → return null.
Do NOT return null just because the footer looks short — read it fully.

════════════════════════════════════════
OUTPUT FORMAT — RETURN EXACTLY THIS
════════════════════════════════════════
{
  "invoice_number": "string or null",
  "invoice_date": "string or null",
  "due_date": "string or null",
  "currency": "ISO 4217 code or null",
  "reference_number": "string or null",
  "reference_type": "e.g. BOL No / PO No / Container No or null",
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

Return ONLY the JSON object. No markdown. No explanation. No extra text."""


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
        # max_new_tokens=512 — enough for detailed invoices with many line items
        # do_sample=False — greedy, faster and deterministic
        # use_cache=True  — KV cache reuse across decode steps
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
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
