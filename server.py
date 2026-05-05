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

PROMPT= """You are an expert invoice data extraction system. Extract structured data from the invoice image and return ONLY a valid JSON object.

Do NOT include explanations, markdown, or extra text.

────────────────────────────
STRICT RULES
────────────────────────────
- NEVER guess or hallucinate values
- NEVER confuse vendor (issuer) with customer (recipient)
- Use null for missing or unreadable fields (NEVER use 0 unless explicitly shown)
- Amounts: extract exact numeric values only (remove currency symbols if present)
- Dates: copy exactly as shown in the invoice
- Do NOT reformat business names or addresses

────────────────────────────
VENDOR IDENTIFICATION (CRITICAL)
────────────────────────────
- Vendor = entity issuing the invoice (usually TOP section)
- Primary sources (in order of priority):
  1. Top-left / top-center header text
  2. Text next to or under a logo (ONLY if readable)
  3. Labels like: "From", "Seller", "Issued By", "Supplier"
- Logo alone is NOT enough → use nearby readable text
- Vendor address must come from explicit text blocks near vendor name
- Ignore footer, bank details, and payment sections for vendor identification
- If unclear, set vendor fields to null

────────────────────────────
REFERENCE NUMBER RULES
────────────────────────────
- reference_number includes ONLY:
  Ref No, PO Number, Purchase Order, Receipt No, Shipper No, Container No, Booking No, Order ID
- NEVER use Invoice Number / Bill Number as reference_number
- Priority if multiple exist:
  PO Number > Order ID > Receipt No > Shipping IDs
- If none exist, return null

────────────────────────────
AMOUNT EXTRACTION (CRITICAL FIX)
────────────────────────────
- Extract numbers exactly as shown in:
  "Total", "Grand Total", "Amount Due", "Balance Due"
- DO NOT default missing values to 0
- Only use 0 if invoice explicitly shows 0
- If unclear or partially visible → use null
- line_items totals must match visible invoice values exactly

────────────────────────────
NOTE EXTRACTION
────────────────────────────
Include meaningful operational text only.

INCLUDE:
- Payment instructions
- Contact details (phone/email/support)
- Terms & conditions
- Tax/legal disclaimers
- Customer support / query instructions

EXCLUDE:
- "Thank you for your business"
- Greetings / slogans
- Decorative branding text

If only generic phrases exist → return null

────────────────────────────
OUTPUT FORMAT (STRICT JSON)
────────────────────────────
{
  "invoice_number": "Invoice No / Bill No",
  "date": "invoice date",
  "reference_number": "Ref/PO/Receipt/Shipper/etc or null",
  "vendor_name": "issuer name",
  "vendor_address": null,
  "customer_name": null,
  "customer_address": null,
  "line_items": [
    {
      "description": "...",
      "qty": null,
      "unit_price": null,
      "total": null
    }
  ],
  "total_amount": null,
  "note": null
}

Return ONLY the JSON.
No additional text allowed.
"""



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
