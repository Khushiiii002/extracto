from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import httpx
import asyncio
import time
import hashlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")


LIGHTNING_URL = "https://8000-01kqknda0a0j57k3kjtm42r489.cloudspaces.litng.ai/extract"

ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB hard cap

# ─── In-flight request deduplication ───────────────────────────────────────────
# If two identical files are uploaded simultaneously, only one request is sent
# to Lightning; the second awaits the first result.
_in_flight: dict[str, asyncio.Future] = {}

# ─── Shared persistent HTTP/2 client ───────────────────────────────────────────
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global _client
    _client = httpx.AsyncClient(
        verify=False,
        timeout=httpx.Timeout(
            connect=8.0,
            read=300.0,
            write=30.0,
            pool=5.0,
        ),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=60,
        ),
        http2=True,
    )


@app.on_event("shutdown")
async def shutdown():
    if _client:
        await _client.aclose()


def _file_hash(data: bytes) -> str:
    """SHA-256 fingerprint for dedup key."""
    return hashlib.sha256(data).hexdigest()


async def _call_lightning(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """Single attempt to call Lightning AI; raises httpx exceptions on failure."""
    response = await _client.post(
        LIGHTNING_URL,
        files={"file": (filename, file_bytes, content_type)},
    )
    response.raise_for_status()
    return response.json()


@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    # ── Validate type ──────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, or PDF files are allowed.")

    # ── Read & size-check ──────────────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    key = _file_hash(file_bytes)
    t_start = time.perf_counter()

    # ── Deduplication: reuse in-flight future for identical files ──────────────
    if key in _in_flight:
        try:
            result = await asyncio.shield(_in_flight[key])
            elapsed = round((time.perf_counter() - t_start) * 1000)
            if isinstance(result, dict):
                result = {**result, "_elapsed_ms": elapsed, "_cache_hit": True}
            return result
        except Exception:
            pass  # fall through and retry fresh

    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    _in_flight[key] = fut

    last_exc = None
    result = None

    try:
        for attempt in range(3):
            try:
                result = await _call_lightning(file_bytes, file.filename, file.content_type)
                break  # success

            except httpx.TimeoutException as e:
                last_exc = e
                if attempt == 2:
                    raise HTTPException(status_code=504, detail="Model inference timed out. Please try again.")

            except httpx.HTTPStatusError as e:
                last_exc = e
                # 4xx errors are not retryable
                if e.response.status_code < 500:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Upstream error: {e.response.text[:200]}"
                    )
                if attempt == 2:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Lightning AI returned {e.response.status_code}. Try again shortly."
                    )

            except httpx.RequestError as e:
                last_exc = e
                if attempt == 2:
                    raise HTTPException(status_code=502, detail=f"Could not reach model server: {e}")

            # Exponential backoff: 0.5s → 1s → 2s
            await asyncio.sleep(0.5 * (2 ** attempt))

        if result is None:
            raise HTTPException(status_code=502, detail=f"All retries failed: {last_exc}")

        elapsed = round((time.perf_counter() - t_start) * 1000)
        if isinstance(result, dict):
            result["_elapsed_ms"] = elapsed

        fut.set_result(result)
        return result

    except HTTPException:
        fut.cancel()
        raise

    except Exception as e:
        fut.cancel()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        _in_flight.pop(key, None)


@app.get("/health")
def health():
    return {"status": "ok"}
