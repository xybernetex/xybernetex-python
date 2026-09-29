"""The Xybernetex policy service (api.xybernetex.com): which follow-up an ended
run gets (POST /intervene), and what happened after it (POST /outcome). The
OpenClaw plugin's createRemoteDecider and createOutcomeSender (src/
interventions.js, src/outcomes.js), with the same wire format.

Only the run's shape travels: success, whether a failure looks retriable,
the tool-call count and the model id for a decision; labels and counts for an
outcome. Never prompts, files, command text or error text. A decision falls
back to the local rule (logged as such) whenever the service can't answer,
so a network problem can never stop a follow-up or loop one.

Standard library only: requests are blocking; the adapter runs them off the
event loop.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from .deaths import retriable
from .followups import RunSummary, decide_local

DEFAULT_ENDPOINT = "https://api.xybernetex.com"
ACTIONS = ("none", "retry", "verify")

# post(url, api_key, body, timeout) -> (HTTP status, parsed JSON body or {})
Post = Callable[[str, str, dict, float], tuple[int, dict]]


def base_url(endpoint: str) -> str:
    """The service root, from either the root or the plugin-style .../evaluate URL."""
    url = endpoint.rstrip("/")
    return url[: -len("/evaluate")] if url.endswith("/evaluate") else url


def post_json(url: str, api_key: str, body: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers={
        "authorization": f"Bearer {api_key}", "content-type": "application/json",
        "user-agent": "xybernetex-python"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as res:
            status, raw = res.status, res.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        out = json.loads(raw) if raw else {}
    except ValueError:
        out = {}
    return status, out if isinstance(out, dict) else {}


def wire_summary(summary: RunSummary) -> dict:
    """What /intervene is told about a run - nothing else."""
    return {"success": bool(summary.success), "retriable": bool(summary.retriable or retriable(summary.error)),
            "toolCalls": int(summary.tool_calls), "model": summary.model or None}


class RemoteDecider:
    """decide(summary) backed by POST /intervene, falling back to the local rule."""

    def __init__(self, api_key: str, endpoint: str = DEFAULT_ENDPOINT, *, timeout: float = 3.0,
                 fallback: Callable[[RunSummary], dict] = decide_local, post: Post = post_json) -> None:
        self.url = base_url(endpoint) + "/intervene"
        self._api_key, self._timeout, self._fallback, self._post = api_key, timeout, fallback, post

    def __call__(self, summary: RunSummary) -> dict:
        try:
            status, out = self._post(self.url, self._api_key, {"summary": wire_summary(summary)}, self._timeout)
            if not 200 <= status < 300 or out.get("action") not in ACTIONS:
                raise RuntimeError(f"HTTP {status} {out.get('error') or ''}".strip())
            return {"action": out["action"], "probability": out.get("probability"), "rule": out.get("rule"),
                    "policy": out.get("policy")}
        except Exception as err:  # noqa: BLE001 - any failure means: decide locally
            local = self._fallback(summary)
            return {**local, "rule": f"fallback:{local['rule']}", "fallbackReason": str(err)[:120]}


class OutcomeSender:
    """send(episode) for the outcome tracker: POST /outcome, labels and counts only."""

    def __init__(self, api_key: str, endpoint: str = DEFAULT_ENDPOINT, *, timeout: float = 3.0,
                 post: Post = post_json) -> None:
        self.url = base_url(endpoint) + "/outcome"
        self._api_key, self._timeout, self._post = api_key, timeout, post

    def __call__(self, episode: dict) -> None:
        status, out = self._post(self.url, self._api_key, {"episode": episode}, self._timeout)
        if not 200 <= status < 300:
            raise RuntimeError(f"HTTP {status} {out.get('error') or ''}".strip())
