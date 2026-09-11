"""
Face similarity comparison using InsightFace buffalo_sc.
Lightweight model, ~16MB, ~400-550MB peak RAM.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import sys
import os
import threading

# The model is loaded once per process and reused.
#
# It used to be constructed inside run_pair(), which meant every HTTP request
# re-read ~16MB of ONNX weights and re-allocated the session: several seconds
# of latency and a 400-550MB allocation spike per call. Under a morning's
# attendance marking - many supervisors photographing many people inside a few
# minutes - that reliably exhausted the box. Loading is idempotent and the
# session is safe to share, so it is done once behind a lock and then reused.
_MODEL = None
_MODEL_LOCK = threading.Lock()

DEFAULT_THRESHOLD = 0.5


def get_model():
    """The process-wide FaceAnalysis instance, built on first use."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = _build_model()
    return _MODEL


def is_loaded():
    """Whether the model is already resident, for the health endpoint."""
    return _MODEL is not None


def _build_model():
    """Initialize buffalo_sc (lightweight model)."""
    print("Loading buffalo_sc model (first run downloads ~16MB)...", flush=True)
    app = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(320, 320))
    print("Model loaded.\n", flush=True)
    return app


def load_model():
    """Backwards-compatible alias; the model is a process-wide singleton."""
    return get_model()


def decode_image_bytes(data):
    """Decode raw image bytes to a BGR numpy array."""
    if not data:
        raise ValueError("Empty image data")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def read_image_safe(path):
    """Read image from disk, handles Unicode paths on Windows."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    return decode_image_bytes(data)


def _largest_first(faces):
    """Biggest face first: the subject of the photograph, not a bystander."""
    if len(faces) > 1:
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
    return faces


def get_face_embedding_from_array(app, img, label="image"):
    """Detect face in a numpy image and return (embedding, bbox)."""
    faces = app.get(img)

    if len(faces) == 0:
        raise ValueError(f"No face detected in {label}")

    faces = _largest_first(faces)

    return faces[0].embedding, faces[0].bbox


def get_face_embedding(app, image_path):
    """Detect face from a file path and return (embedding, bbox). CLI helper."""
    img = read_image_safe(image_path)
    return get_face_embedding_from_array(app, img, label=image_path)


def cosine_similarity(emb1, emb2):
    """Cosine similarity between two embeddings."""
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))


def detect(image_bytes):
    """
    How many faces are in one image, and how big the largest one is.

    Used to check an enrolment photograph the moment it is uploaded. A master
    photograph with no detectable face fails every attendance match afterwards,
    and the person who can fix it is the one uploading it, not the supervisor
    standing in front of the employee weeks later.
    """
    app = get_model()
    img = decode_image_bytes(image_bytes)
    faces = _largest_first(app.get(img))
    height, width = img.shape[:2]

    if not faces:
        return {
            "faces": 0,
            "usable": False,
            "width": width,
            "height": height,
            "face_width": 0,
            "face_height": 0,
        }

    bbox = faces[0].bbox
    return {
        "faces": len(faces),
        "usable": True,
        "width": width,
        "height": height,
        "face_width": int(bbox[2] - bbox[0]),
        "face_height": int(bbox[3] - bbox[1]),
    }


def run_pair(image1_bytes, image2_bytes, threshold=DEFAULT_THRESHOLD):
    """Compare two images given as raw bytes. Pure function, no I/O side-effects."""
    app = get_model()
    img1 = decode_image_bytes(image1_bytes)
    img2 = decode_image_bytes(image2_bytes)

    emb1, _ = get_face_embedding_from_array(app, img1, label="image1")
    emb2, _ = get_face_embedding_from_array(app, img2, label="image2")

    similarity = cosine_similarity(emb1, emb2)
    similarity_pct = (similarity + 1) / 2 * 100

    return {
        "cosine_similarity": round(similarity, 4),
        "similarity_score": round(similarity_pct, 2),
        "threshold": threshold,
        "same_person": similarity >= threshold,
    }


def compare_faces(image1_path, image2_path, threshold=DEFAULT_THRESHOLD):
    """CLI entry point. Prints results."""
    app = get_model()

    print(f"Processing: {image1_path}")
    emb1, bbox1 = get_face_embedding(app, image1_path)
    print(f"  Face bbox: {bbox1.astype(int).tolist()}")

    print(f"Processing: {image2_path}")
    emb2, bbox2 = get_face_embedding(app, image2_path)
    print(f"  Face bbox: {bbox2.astype(int).tolist()}\n")

    similarity = cosine_similarity(emb1, emb2)
    similarity_pct = (similarity + 1) / 2 * 100

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Cosine similarity:  {similarity:.4f}")
    print(f"Similarity score:   {similarity_pct:.2f}%")
    print(f"Threshold:          {threshold}")
    print(f"Same person:        {'YES' if similarity >= threshold else 'NO'}")
    print("=" * 50)

    return {
        "similarity": similarity,
        "similarity_pct": similarity_pct,
        "is_match": similarity >= threshold,
    }


if __name__ == "__main__":
    if len(sys.argv) == 3:
        img1, img2 = sys.argv[1], sys.argv[2]
    else:
        img1 = "face1.jpg"
        img2 = "face2.jpg"

    try:
        compare_faces(img1, img2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
