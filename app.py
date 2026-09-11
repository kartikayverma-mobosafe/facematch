"""FastAPI service wrapping compare_faces."""

import base64
import binascii
import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

import compare_faces

# Optional shared secret. When FACE_MATCH_API_KEY is set in the environment
# every endpoint except /health requires a matching X-Api-Key header. Left
# unset the service is open, which is only safe when it is bound to localhost
# or a private interface.
API_KEY = os.environ.get("FACE_MATCH_API_KEY", "").strip()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pay the one-off model load at boot rather than inside whichever request
    # happens to arrive first. Without this the first caller of the morning
    # waits several seconds and usually times out.
    compare_faces.get_model()
    yield


app = FastAPI(title="Face Compare API", version="1.1.0", lifespan=lifespan)


class CompareResponse(BaseModel):
    cosine_similarity: float
    similarity_score: float
    threshold: float
    same_person: bool


class DetectResponse(BaseModel):
    faces: int
    usable: bool
    width: int
    height: int
    face_width: int
    face_height: int


class Base64CompareRequest(BaseModel):
    image1: str = Field(..., description="Raw base64 or data:image/*;base64,... URI")
    image2: str = Field(..., description="Raw base64 or data:image/*;base64,... URI")
    threshold: float = Field(compare_faces.DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class Base64DetectRequest(BaseModel):
    image: str = Field(..., description="Raw base64 or data:image/*;base64,... URI")


def _require_key(supplied: Optional[str]) -> None:
    if not API_KEY:
        return
    if not supplied or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail={
            "code": "unauthorized", "message": "A valid X-Api-Key is required"})


def _strip_data_uri(s: str) -> str:
    if s.startswith("data:") and ";base64," in s:
        return s.split(";base64,", 1)[1]
    return s


def _decode_b64_field(value: str, field_name: str) -> bytes:
    stripped = _strip_data_uri(value).strip()
    try:
        return base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "bad_base64",
                "message": f"{field_name} is not valid base64: {e}",
            },
        )


def _fail(e: ValueError) -> HTTPException:
    """
    Turn an engine ValueError into a coded 400.

    The caller has to tell "the live photograph has no face in it" (retake it)
    apart from "the enrolled photograph has no face in it" (fix the employee
    master). Both used to come back as prose, which left the decision to
    string matching at the far end.
    """
    message = str(e)
    if "image1" in message:
        code = "no_face_image1"
    elif "image2" in message:
        code = "no_face_image2"
    elif "No face detected" in message:
        code = "no_face"
    elif "decode" in message or "Empty image" in message:
        code = "bad_image"
    else:
        code = "compare_failed"
    return HTTPException(status_code=400, detail={"code": code, "message": message})


@app.get("/health")
def health():
    return {"status": "ok", "model": "buffalo_sc", "loaded": compare_faces.is_loaded()}


@app.post("/compare", response_model=CompareResponse)
async def compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    threshold: float = Form(compare_faces.DEFAULT_THRESHOLD, ge=0.0, le=1.0),
    x_api_key: Optional[str] = Header(default=None),
):
    _require_key(x_api_key)
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()
    try:
        return compare_faces.run_pair(img1_bytes, img2_bytes, threshold=threshold)
    except ValueError as e:
        raise _fail(e)


@app.post("/compare/base64", response_model=CompareResponse)
def compare_base64(
    payload: Base64CompareRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_key(x_api_key)
    img1_bytes = _decode_b64_field(payload.image1, "image1")
    img2_bytes = _decode_b64_field(payload.image2, "image2")
    try:
        return compare_faces.run_pair(
            img1_bytes, img2_bytes, threshold=payload.threshold
        )
    except ValueError as e:
        raise _fail(e)


@app.post("/detect", response_model=DetectResponse)
async def detect(
    image: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    """Whether one image holds a face worth enrolling. No comparison."""
    _require_key(x_api_key)
    data = await image.read()
    try:
        return compare_faces.detect(data)
    except ValueError as e:
        raise _fail(e)


@app.post("/detect/base64", response_model=DetectResponse)
def detect_base64(
    payload: Base64DetectRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_key(x_api_key)
    try:
        return compare_faces.detect(_decode_b64_field(payload.image, "image"))
    except ValueError as e:
        raise _fail(e)
