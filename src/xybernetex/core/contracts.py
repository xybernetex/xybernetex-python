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
    basis: str | None = None    # v2: the words of the request this check enforces (not part of the hash)


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    source: str = "developer"   # developer | generated
    refused: tuple[str, ...] = ()  # why checks were dropped (no command text)
    overturned: int = 0          # v2: checks a judge ruled wrong and removed (not part of the hash)

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
        if self.results:
            return all(r.passed for r in self.results)
        # Never vacuously met - except a contract a judge emptied: every check was overturned as wrong.
        return not self.contract.checks and self.contract.overturned > 0

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> dict:
        """For the log: counts and per-check pass/fail by position, no text."""
        return {"contract": self.contract.hash, "checks": len(self.results),
                "passed": sum(r.passed for r in self.results), "failedAt": [i for i, r in enumerate(self.results) if not r.passed]}


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`\"'“”‘’]", "", text)).strip().lower()


def cites(basis: Any, request: str) -> bool:
    """v2: a check's basis must be words the request actually contains (quotes and spacing aside)."""
    if not isinstance(basis, str):
        return False
    b = _plain(basis).strip(" .,:;!?-")
    return len(b) >= 3 and b in _plain(request)


def parse_contract(raw: Any, source: str = "developer", request: str | None = None) -> Contract:
    """{"checks": [{"name", "command", "expect"?, "timeout"?}, ...]} (or the bare list) -> Contract.
    Malformed or unsafe checks are dropped and counted in `refused`; none left raises ContractError.
    With `request` (contract v2), every check must also quote the request in "basis", or it's dropped:
    a check can only enforce something the user actually said."""
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
        basis = item.get("basis")
        if request is not None and not cites(basis, request):
            refused.append(f"check {i}: its basis isn't a quote from the request")
            continue
        name = str(item.get("name") or command)[:MAX_NAME]
        checks.append(Check(name=name, command=command, expect=expect or None, timeout=timeout,
                            basis=str(basis)[:300] if request is not None else None))
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


#: Contract v2. The first benchmark showed v1's failure: the writer computed expected answers in its head
#: (anchor slugs, sort orders, totals) and hard-coded them, so checks failed correct work. v2 checks
#: only what the request states, each quoting the words it enforces ("basis", verified against the request).
CONTRACT_PROMPT_V2 = """You write acceptance checks for an automated coding task. Do not do the task.

Task, exactly as the user gave it:
<<<
{request}
>>>

Write between 2 and {max_checks} checks. Each check is a shell command run from the task's working folder after
the agent finishes. It passes when it exits 0 and, if you give "expect", its output also matches that regex.

Checks must only test what the task explicitly states. Rules:
- Every check has a "basis": the exact words from the task that it enforces, copied verbatim. A check whose basis
  isn't in the task is thrown away.
- Never hard-code an answer you worked out yourself: no expected totals, orderings, formatted strings, slugs or
  counts unless the task states that exact value. Test properties instead: a file exists where the task says, a
  column or flag the task names is present, the program runs and exits 0, the output is valid JSON or CSV, rows
  are in the order the task names, a round trip returns the input, a stated rule holds when the check recomputes
  it from the input files.
- Don't test details the task leaves open (tie-breaking, exact wording, formatting the task doesn't specify,
  edge cases it doesn't mention).
- Read-only: never create, change, move or delete files, and never use the network. Use python3 beyond test/grep.
- Keep each command short and self-contained. Fewer, certain checks beat many doubtful ones.

Reply with JSON only, no prose:
{{"checks": [{{"name": "short description", "basis": "exact words from the task", "command": "shell command",
"expect": "optional regex"}}]}}"""


def contract_prompt(request: str, version: str = "v1") -> str:
    template = CONTRACT_PROMPT_V2 if version == "v2" else CONTRACT_PROMPT
    return template.format(request=request.strip()[:8000], max_checks=MAX_CHECKS)


JUDGE_PROMPT = """You review one automatically written acceptance check that failed. The check was written by a model
from the user's request before the work was done, and such checks are often wrong: they guess exact values,
formats or conventions the request never stated, or test details the request leaves open.

The user's request:
<<<
{request}
>>>

The check: {name}
It claims to enforce: "{basis}"
Command: {command}
Result: {why}
Output (end):
{tail}
{dispute}
Does this failure show that the work clearly fails to do something the request states ("work")? Or could the check
be wrong, stricter than the request, or testing something the request leaves open ("check")? Answer "work" only
when the output shows a clear violation of what the request says.

Reply with JSON only: {{"verdict": "work" or "check", "why": "one sentence"}}"""


def _why(r: "CheckResult") -> str:
    return r.error or (f"exited {r.exit_code}" if r.exit_code != 0 else f"output doesn't match /{r.check.expect}/")


def judge_prompt(request: str, result: "CheckResult", dispute: str | None = None) -> str:
    c = result.check
    return JUDGE_PROMPT.format(request=request.strip()[:6000], name=c.name, basis=c.basis or "(none given)",
                               command=c.command, why=_why(result), tail=result.output.strip()[-1200:] or "(none)",
                               dispute=f"\nThe agent disputes this check: {dispute[:600]}\n" if dispute else "")


def parse_judgment(text: Any) -> str:
    """A judge's reply -> "work" (the work is wrong: keep the check) or "check" (drop it). Unreadable -> "work",
    so a broken judge never silently waives a check."""
    if not isinstance(text, str):
        return "work"
    m = re.search(r'"verdict"\s*:\s*"(work|check)"', text, re.I)
    return m.group(1).lower() if m else "work"


_DISPUTE = re.compile(r"^\s*DISPUTE:\s*(.+?)\s*(?::|-|—)\s+(.+)$", re.M)


def parse_disputes(text: Any, contract: Contract) -> list[tuple[Check, str]]:
    """`DISPUTE: <check name>: <why>` lines in an agent's reply -> the checks they name, with the reason."""
    if not isinstance(text, str):
        return []
    found: list[tuple[Check, str]] = []
    for m in _DISPUTE.finditer(text):
        named = _plain(m.group(1))
        for c in contract.checks:
            if c not in [f[0] for f in found] and (_plain(c.name) == named or _plain(c.name).startswith(named)
                                                   or named.startswith(_plain(c.name))):
                found.append((c, m.group(2).strip()))
                break
    return found


def without(contract: Contract, dropped: list[Check]) -> Contract:
    """The contract minus checks a judge overturned (possibly none left: met)."""
    keep = tuple(c for c in contract.checks if c not in dropped)
    return Contract(checks=keep, source=contract.source, refused=contract.refused,
                    overturned=contract.overturned + len(contract.checks) - len(keep))


def restrict(verdict: "Verdict", contract: Contract) -> "Verdict":
    """A verdict's results for the checks still in `contract` (no re-run)."""
    return Verdict(contract, [r for r in verdict.results if r.check in contract.checks])


def parse_generated(text: str, request: str | None = None) -> Contract:
    """A model's reply to contract_prompt -> Contract (tolerates code fences and stray prose).
    With `request` (v2), checks must quote it (parse_contract)."""
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
    return parse_contract(raw, source="generated", request=request)


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


def failure_message(verdict: Verdict, disputable: bool = False) -> str:
    """The follow-up for a run whose contract failed: exactly what failed, nothing generic. With
    `disputable` (v2), the agent is told the checks can be wrong and how to dispute one."""
    lines = [f"{MARKER} The task isn't finished yet. These acceptance checks fail when run in your working folder:"]
    for r in verdict.failed:
        tail = r.output.strip()[-600:]
        lines.append(f"\n- {r.check.name}\n  command: {r.check.command}\n  result: {_why(r)}" +
                     (f"\n  checks this part of the request: \"{r.check.basis}\"" if r.check.basis else "") +
                     (f"\n  output (end):\n{tail}" if tail else ""))
    if disputable:
        lines.append("\nThese checks were written automatically from the request before you started, and they can be "
                     "wrong. Fix any real gap between your work and what the request asks. If a check demands something "
                     "the request didn't ask for, don't change correct work to satisfy it: leave that part as it is "
                     "and add a line `DISPUTE: <check name>: <why>` to your final answer. Then give your final answer "
                     "again in full.")
    else:
        lines.append("\nFix the work so these pass - change your deliverables, not the checks - then give your final "
                     "answer again in full.")
    message = "\n".join(lines)
    return message if len(message) <= MESSAGE_LIMIT else message[:MESSAGE_LIMIT - 40] + "\n...[more failures omitted]"
