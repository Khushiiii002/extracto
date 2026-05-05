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

PROMPT = """You are an expert invoice document extraction system.

Your job is to carefully scan the entire image and extract structured invoice data.

Return ONLY valid JSON. No explanation, no markdown.

────────────────────────────
CRITICAL INSTRUCTION
────────────────────────────
You MUST scan the entire image in this order:
1. Top section (vendor + header)
2. Middle section (line items / table)
3. Bottom section (totals / footer)
4. Right side / margins (reference numbers, codes)

Do NOT skip any region.

────────────────────────────
STRICT RULES
────────────────────────────
- Do NOT guess completely invisible values
- If partially visible, extract best possible value
- Use null only if absolutely no visual evidence exists
- NEVER default numeric fields to 0 unless explicitly written
- Extract numbers exactly as seen (remove currency symbols only)

────────────────────────────
VENDOR RULES
────────────────────────────
- Vendor is in TOP section (usually largest text or near logo)
- Logo text alone is NOT sufficient unless readable text exists near it
- Address is usually directly under vendor name
- Combine multi-line addresses

────────────────────────────
REFERENCE NUMBER (VERY IMPORTANT)
────────────────────────────
Look carefully in:
- Top header
- Right side near invoice number
- Near barcode / QR codes
- Labels like: Ref No, PO No, Order ID, Shipper No, Container No, Receipt No

Rules:
- NEVER use Invoice Number as reference_number
- If multiple exist, choose most transaction-related:
  PO Number > Order ID > Receipt No > Shipping IDs
- If unclear, pick the most visually prominent non-invoice identifier

────────────────────────────
LINE ITEMS (CRITICAL TABLE EXTRACTION)
────────────────────────────
- Line items are usually in a TABLE format in the middle section
- Each row = one item
- Columns may include: description, qty, unit price, total

Rules:
- Extract ALL visible rows
- If table borders are missing, still infer rows by alignment
- If qty/price missing → use null (NOT 0)
- Do NOT merge rows

────────────────────────────
TOTAL AMOUNT (CRITICAL)
────────────────────────────
- Look ONLY in bottom section of invoice
- Labels: Total, Grand Total, Amount Due, Balance Due
- If multiple totals exist, choose the LARGEST valid amount
- Prioritize bold / boxed / right-aligned numbers
- Never return 0 unless explicitly printed

────────────────────────────
NOTE FIELD
────────────────────────────
Include only meaningful business info:
- Payment instructions
- Contact details
- Terms / tax notes
- Support info

Exclude:
- Thank you messages
- Greetings
- Branding slogans

────────────────────────────
OUTPUT FORMAT
────────────────────────────
{
  "invoice_number": "string or null",
  "date": "string or null",
  "reference_number": "string or null",
  "vendor_name": "string or null",
  "vendor_address": "string or null",
  "customer_name": "string or null",
  "customer_address": "string or null",
  "line_items": [
    {
      "description": "string or null",
      "qty": null,
      "unit_price": null,
      "total": null
    }
  ],
  "total_amount": null,
  "note": "string or null"
}

Return ONLY JSON.
No extra text."""


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
