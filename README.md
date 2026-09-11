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

## Notes that matter

- **The model is loaded once per process.** It is ~16MB on disk but ~500MB
  resident, and takes seconds to build. It is warmed at startup so the first
  request of the day is not the one that pays for it.
- **One worker.** Every extra uvicorn worker is another full copy of the model
  in RAM.
- **Bind to localhost** and reach it over the SSH tunnel or a private
  interface. If it must listen publicly, set `FACE_MATCH_API_KEY` and put TLS
  in front of it — the images on the wire are photographs of employees.
- First run downloads the model into `~/.insightface`, so the service account
  needs a writable `HOME`.
