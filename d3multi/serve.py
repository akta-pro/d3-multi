"""Add the `multi` question type to any System One server.

python -m d3multi.serve --upstream http://localhost:8100 --port 8200

Requests go to POST /v1/systemone as usual. A question with "type": "multi" is sent upstream as a
`choice` question with the cutoff option added, and its answer comes back as
{"type": "multi", "selected": [...], "scores": {...}, "cutoff": p}. Every other question passes
through unchanged.
"""

from __future__ import annotations

import argparse
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from d3multi import multi


def translate(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    questions = body.get("questions")
    if not isinstance(questions, dict):
        return body, {}
    offsets, sent = {}, {}
    for name, q in questions.items():
        if isinstance(q, dict) and q.get("type") == "multi":
            sent[name] = multi.to_choice(q)
            offsets[name] = float(q.get("offset", 0.0))
        else:
            sent[name] = q
    return {**body, "questions": sent}, offsets


def decode(response: dict[str, Any], offsets: dict[str, float]) -> dict[str, Any]:
    answers = response.get("answers", {})
    for name, offset in offsets.items():
        if name in answers and "probabilities" in answers[name]:
            answers[name] = multi.decode(answers[name]["probabilities"], offset)
    return response


def app(upstream: str, timeout: float) -> FastAPI:
    api = FastAPI()
    client = httpx.AsyncClient(base_url=upstream, timeout=timeout)

    @api.post("/v1/systemone")
    async def systemone(request: Request):
        body = await request.json()
        try:
            sent, offsets = translate(body)
        except ValueError as e:
            return JSONResponse({"error": {"message": str(e), "type": "BadRequestError", "code": 400}}, status_code=400)
        resp = await client.post("/v1/systemone", json=sent)
        if resp.status_code != 200:
            return JSONResponse(resp.json(), status_code=resp.status_code)
        return decode(resp.json(), offsets)

    @api.get("/v1/models")
    async def models():
        resp = await client.get("/v1/models")
        return JSONResponse(resp.json(), status_code=resp.status_code)

    return api


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    uvicorn.run(app(a.upstream, a.timeout), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
