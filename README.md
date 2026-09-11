# Face match engine

InsightFace `buffalo_sc` behind FastAPI. Compares two photographs and returns
how alike the faces are. Used by HRMS attendance: the live photograph taken
while marking somebody present is compared against the photograph on their
employee master.

## Endpoints

| Method | Path             | Body                                   | Returns |
| ------ | ---------------- | -------------------------------------- | ------- |
| GET    | `/health`        | -                                      | `{status, model, loaded}` |
| POST   | `/compare`       | multipart: `image1`, `image2`, `threshold` | `{cosine_similarity, similarity_score, threshold, same_person}` |
| POST   | `/compare/base64`| json: `{image1, image2, threshold}`    | same as above |
| POST   | `/detect`        | multipart: `image`                     | `{faces, usable, width, height, face_width, face_height}` |
| POST   | `/detect/base64` | json: `{image}`                        | same as above |

`similarity_score` is the cosine similarity rescaled to 0-100 for display:
`(cosine + 1) / 2 * 100`. `same_person` compares the **cosine** value against
`threshold`, so a 0.40 threshold is a 70.0% score. Do not compare the
percentage against the threshold directly.

A photograph with no detectable face is a `400` carrying
`{"detail": {"code": "no_face_image1" | "no_face_image2", "message": ...}}`, so
the caller can say whether it is the live shot or the enrolled one that needs
redoing.

## Running it

```bash
./run_api.sh            # creates .venv on first run, serves on 0.0.0.0:9000
```

For a real deployment use `facematch.service` instead, which binds to
localhost and sets the shared secret.

## Installing

```bash
git clone https://github.com/kartikayverma-mobosafe/facematch.git /opt/facematch
cd /opt/facematch
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.lock.txt
```

Install from `requirements.lock.txt`, not `requirements.txt`. The floating set
resolves to whatever is current, and insightface has already gone 0.7.3 → 2.0
under this code once; the lock file is the set actually verified to work.

Needs only Python 3 with `venv` — no compiler, and no system OpenCV libraries.

## Notes that matter

- **The model is loaded once per process**, and warmed at startup so the first
  request of the morning is not the one that pays for it. It used to be built
  inside each request handler, which cost seconds of latency per call.
- **Measured memory: ~195MB resident** (Python 3.12, insightface 2.0,
  `buffalo_sc`, two phone photographs in flight). The figure of 400-550MB in
  the original script's docstring was wrong — it is well under a quarter of a
  gigabyte, which is what makes this practical to co-host.
- **One worker.** Every extra uvicorn worker is another full copy of the model
  in RAM.
- **Bind to localhost** and reach it over the SSH tunnel or a private
  interface. If it must listen publicly, set `FACE_MATCH_API_KEY` and put TLS
  in front of it — the images on the wire are photographs of employees.
- First run downloads the model into `~/.insightface`, so the service account
  needs a writable `HOME`.

## Calibrating the threshold

`same_person` is `cosine >= threshold`. Measured on three real photographs of
two people:

| Pair | Cosine | Score |
| --- | --- | --- |
| Same person, two photos | **+0.681** | 84.05% |
| Different people | +0.169 | 58.47% |
| Different people | +0.143 | 57.16% |

The gap between a genuine match and a stranger is wide, and a 0.40 threshold
(a 70% score) sits in the middle of it. Widen or narrow it against your own
photographs rather than trusting a default — every comparison the HRMS makes
stores the raw cosine it scored, so the data to do that accumulates from the
first day.
