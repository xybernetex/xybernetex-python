"""Contracts: what "done" means for a run, checked by code instead of guessed
by a model - the original Xybernetex's deliverable manifest, as a layer over
any agent framework.

A contract is a short list of acceptance checks. Each check is a shell
command that passes when it exits 0 (and, optionally, its output matches a
pattern): `test -f report.csv`, `python3 -m pytest -q`, `python3 -c "..."`.
The developer can supply one; otherwise one call on the agent's own model
writes it from the request (CONTRACT_PROMPT, parse_generated), so neither the
request nor the checks ever need to leave the customer's environment.

When a run ends, the checks run where the agent's commands run (an executor
the adapter is given), cost no tokens, and give the same answer every time.
An executor should run them on a throwaway copy of the working folder when
it can: a check that runs the deliverable (`python3 kv.py set a 1`) changes
state, and the copy keeps the real work exactly as the agent left it.
All pass: the run is done, no check-your-work turn. Any fail: one follow-up
naming exactly what failed (failure_message), then the checks again.

Checks are read-only as far as a command line shows: anything the gate's
risk classifier doesn't rate "none" is refused, and so is any visible write -
a redirect onto a file, tee, cp, mv, mkdir, sed -i, installs, git beyond
reading - so a wrong or poisoned contract can't delete, push or overwrite the
work it checks. Code inside `python3 -c` can't be read this way; like any
command, it runs with the agent's own permissions. The log identifies a
contract by its hash and records counts, never the check text (it echoes the
request).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .followups import MARKER
from .risk import _segments, _strip_quoted, _words, classify_shell_command

MAX_CHECKS = 12
MAX_COMMAND = 2000
MAX_NAME = 120
MAX_TIMEOUT = 600
OUTPUT_TAIL = 1500
MESSAGE_LIMIT = 6000

# command, timeout seconds -> (exit code, combined output). The adapter's way
# of running a command where the agent's commands run.
Executor = Callable[[str, int], "tuple[int, str]"]


# Commands that change files or install things, as a check's command line shows them.
_WRITERS = frozenset({"tee", "touch", "mkdir", "cp", "mv", "ln", "chmod", "chown", "install", "truncate", "dd",
                      "patch", "rsync", "pip", "pip3", "npm", "npx", "yarn", "pnpm", "apt", "apt-get", "unzip", "tar",
                      "curl", "wget"})
_READ_ONLY_GIT = frozenset({"status", "diff", "log", "show", "ls-files", "rev-parse", "branch", "cat-file", "grep"})
_REDIRECT = re.compile(r"(?<![<>&\d])\d?>{1,2}\s*([^\s;&|]+)")
_NULL_TARGET = re.compile(r"^(/dev/null|&\d|nul)$", re.I)


def _writes(command: str) -> str | None:
    """What visible write a command makes, or None."""
    for target in _REDIRECT.findall(_strip_quoted(command)):
        if not _NULL_TARGET.match(target):
            return "redirects output into a file"
    for segment in _segments(command):
        cmd, args = _words(segment)
        if cmd in _WRITERS:
            return f"runs {cmd}"
        if cmd in ("sed", "perl") and any(a == "-i" or a.startswith("-i") for a in args):
            return f"edits files in place ({cmd} -i)"
        if cmd == "git" and next((a for a in args if not a.startswith("-")), None) not in _READ_ONLY_GIT:
            return "runs git beyond reading"
    return None


class ContractError(ValueError):
    """A contract with no usable checks."""


@dataclass(frozen=True)
class Check:
    name: str
    command: str
    expect: str | None = None   # regex the combined output must also match
    timeout: int = 60


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    source: str = "developer"   # developer | generated
    refused: tuple[str, ...] = ()  # why checks were dropped (no command text)

    @property
    def hash(self) -> str:
        """Stable id: sha256 of the canonical checks, first 16 hex - like the gate's params hash."""
        canon = json.dumps([[c.name, c.command, c.expect, c.timeout] for c in self.checks],
                           separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


@dataclass
class CheckResult:
    check: Check
    passed: bool
    exit_code: int | None
    output: str = ""
    seconds: float = 0.0
    error: str | None = None


@dataclass
class Verdict:
    contract: Contract
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> dict:
        """For the log: counts and per-check pass/fail by position, no text."""
        return {"contract": self.contract.hash, "checks": len(self.results),
                "passed": sum(r.passed for r in self.results), "failedAt": [i for i, r in enumerate(self.results) if not r.passed]}


def parse_contract(raw: Any, source: str = "developer") -> Contract:
    """{"checks": [{"name", "command", "expect"?, "timeout"?}, ...]} (or the bare list) -> Contract.
    Malformed or unsafe checks are dropped and counted in `refused`; none left raises ContractError."""
    items = raw.get("checks") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ContractError('a contract is {"checks": [...]}')
    checks: list[Check] = []
    refused: list[str] = []
    for i, item in enumerate(items):
        if len(checks) >= MAX_CHECKS:
            refused.append(f"check {i}: over {MAX_CHECKS} checks")
            continue
        if not isinstance(item, dict) or not isinstance(item.get("command"), str) or not item["command"].strip():
            refused.append(f"check {i}: needs a command")
            continue
        command = item["command"].strip()
        if len(command) > MAX_COMMAND:
            refused.append(f"check {i}: command over {MAX_COMMAND} characters")
            continue
        tier = classify_shell_command(command)
        if tier != "none":
            refused.append(f"check {i}: {tier} - checks must be read-only")
            continue
        write = _writes(command)
        if write:
            refused.append(f"check {i}: {write} - checks must be read-only")
            continue
        expect = item.get("expect")
        if expect is not None:
            if not isinstance(expect, str):
                refused.append(f"check {i}: expect must be a regex string")
                continue
            try:
                re.compile(expect)
            except re.error:
                refused.append(f"check {i}: expect is not a valid regex")
                continue
        timeout = item.get("timeout", 60)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= MAX_TIMEOUT:
            timeout = 60
        name = str(item.get("name") or command)[:MAX_NAME]
        checks.append(Check(name=name, command=command, expect=expect or None, timeout=timeout))
    if not checks:
        raise ContractError("no usable checks" + (f" ({'; '.join(refused)})" if refused else ""))
    return Contract(checks=tuple(checks), source=source, refused=tuple(refused))


CONTRACT_PROMPT = """You write acceptance checks for an automated coding task. Do not do the task.

Task, exactly as the user gave it:
<<<
{request}
>>>

Write between 2 and {max_checks} checks that will all pass if and only if the task is fully and correctly done.
Each check is a shell command run from the task's working folder after the agent finishes. It passes when it
exits 0 and, if you give "expect", its output also matches that regular expression.

Rules:
- Read-only: never create, change, move or delete files, and never use the network.
- Check what the user asked for: files exist where requested, outputs have the requested format and values,
  programs run and behave as described. Use python3 for anything beyond test/grep.
- Prefer exact facts stated in the task (names, columns, values, formats) over guesses.
- If the folder may contain tests the user mentioned, include running them.
- Keep each command short and self-contained.

Reply with JSON only, no prose:
{{"checks": [{{"name": "short description", "command": "shell command", "expect": "optional regex"}}]}}"""


def contract_prompt(request: str) -> str:
    return CONTRACT_PROMPT.format(request=request.strip()[:8000], max_checks=MAX_CHECKS)


def parse_generated(text: str) -> Contract:
    """A model's reply to contract_prompt -> Contract (tolerates code fences and stray prose)."""
    if not isinstance(text, str):
        raise ContractError("no reply")
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        raise ContractError("the reply holds no JSON object")
    try:
        raw = json.loads(body[start:end + 1])
    except ValueError as e:
        raise ContractError(f"the reply isn't valid JSON: {e}") from e
    return parse_contract(raw, source="generated")


def run_checks(contract: Contract, executor: Executor) -> Verdict:
    """Every check, in order, through the executor. An executor error fails that check."""
    verdict = Verdict(contract)
    for check in contract.checks:
        t0 = time.time()
        try:
            code, output = executor(check.command, check.timeout)
            output = output or ""
            passed = code == 0 and (check.expect is None or re.search(check.expect, output) is not None)
            verdict.results.append(CheckResult(check, passed, code, output[-OUTPUT_TAIL:], round(time.time() - t0, 2)))
        except Exception as e:  # noqa: BLE001 - a check that can't run hasn't passed
            verdict.results.append(CheckResult(check, False, None, "", round(time.time() - t0, 2),
                                               f"{type(e).__name__}: {e}"[:300]))
    return verdict


def failure_message(verdict: Verdict) -> str:
    """The follow-up for a run whose contract failed: exactly what failed, nothing generic."""
    lines = [f"{MARKER} The task isn't finished yet. These acceptance checks fail when run in your working folder:"]
    for r in verdict.failed:
        why = (r.error or (f"exited {r.exit_code}" if r.exit_code != 0 else f"output doesn't match /{r.check.expect}/"))
        tail = r.output.strip()[-600:]
        lines.append(f"\n- {r.check.name}\n  command: {r.check.command}\n  result: {why}" +
                     (f"\n  output (end):\n{tail}" if tail else ""))
    lines.append("\nFix the work so these pass - change your deliverables, not the checks - then give your final "
                 "answer again in full.")
    message = "\n".join(lines)
    return message if len(message) <= MESSAGE_LIMIT else message[:MESSAGE_LIMIT - 40] + "\n...[more failures omitted]"
