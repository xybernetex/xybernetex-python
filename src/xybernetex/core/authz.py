"""Authorization: does the user's own request cover a risky tool call?

risk.py says what a call DOES (deletes something, reaches outside the
machine). It can't say whether the user asked for it, so a requested
`rm -rf build` and an `rm -rf data` planted in a file the agent read get the
same tier. This module supplies the missing signal, one label per
destructive or sensitive call:

    "requested"     the user's own message asks for this kind of operation
                    and names every target it touches (verb-only for
                    operations with no parseable target, except deletes and
                    drops)
    "own_artifact"  every target is something the agent itself created
                    earlier in the session (written, mkdir'd, redirected to,
                    CREATE TABLE'd) - routine cleanup of its own scratch work
    "unrequested"   neither
    None            not a risky call, or no user request seen for the session

Only the user's own turns count: they come from the framework's user input,
never from tool results, so an instruction planted in a file or web page
can't make itself "requested". The request text stays in memory on this
machine; only the label is ever sent or logged.

"own_artifact" is weaker evidence than "requested" - a same-session
instruction could create a file and then delete it - so it is a feature for
the policy, never on its own grounds to skip an enforcement rule. Like
risk.py this is a heuristic, not a security boundary.

`owns_files` is the narrow case that may waive one (a rule lists
"own_files" in unlessAuthorization): a plain delete (rm, unlink, del,
Remove-Item) of single files the agent itself created this session -
written, added by a patch, redirected to with > - that no move has touched
since. The same call may cd and run other commands the risk classifier
rates harmless (python3 check.py; ls), but no move: a script can already
delete or move files unseen (risk.py can't read inside it), so harmless
company adds nothing a planted instruction couldn't do anyway, while a
visible move in the same call could land a user's file on the one deleted.
Deleting such a file loses only the agent's own content: whatever it
replaced was already gone when the agent wrote it. Paths are resolved
against the call's workdir and any cd, and must match exactly - a file made
at tmp/data.csv never covers data.csv. Folders never qualify: `mv data.csv
scratch/ && rm -rf scratch` would launder a user's file through a folder the
agent made. The one exception is a tool cache (__pycache__, *.pyc), which
regenerates itself, and only while no move this session has named a folder
or file by that name. Any visible move that could land on a tracked file
(same path, a folder above it, the same file name, or sources we can't read)
drops it from the set.

A port of the OpenClaw plugin's src/authz.js; tests/test_authz.py holds
that file's cases, so the two behave identically.
"""
from __future__ import annotations

import re
from typing import Any

from .risk import SHELL_KEYWORDS, SHELL_TOOLS, SQL_CLIENT, SQL_DESTRUCTIVE, classify_shell_command, classify_tool_call

AUTHORIZATION_LABELS = ("requested", "own_artifact", "unrequested")

VERBS: dict[str, list[str]] = {
    "delete": ["delete", "remove", "rm", "erase", "wipe", "purge", "clean", "clean up", "cleanup", "clear", "empty",
               "truncate", "get rid of", "prune", "discard", "trash", "unlink", "drop", "tidy"],
    "git_rewrite": ["reset", "clean", "discard", "revert", "undo", "rewind", "force", "delete", "remove", "drop",
                    "prune", "clear"],
    "db_destroy": ["drop", "delete", "remove", "truncate", "clear", "wipe", "empty", "purge"],
    "publish": ["push", "publish", "deploy", "release", "upload", "ship", "apply", "merge", "install", "upgrade",
                "scale", "rollout", "roll out"],
    "send": ["post", "put", "patch", "send", "upload", "submit", "message", "notify", "email", "mail", "register",
             "transfer", "scp", "rsync", "sync", "ssh"],
    "credentials": ["create", "write", "add", "set", "configure", "save", "store", "make", "update", "edit", "change"],
    "process": ["kill", "stop", "terminate", "end", "restart", "shut down"],
    "system": ["shutdown", "shut down", "reboot", "restart", "format", "halt", "power off", "install", "uninstall",
               "configure", "enable", "disable", "schedule", "cron"],
}
VERBS["generic"] = list(dict.fromkeys(v for vs in VERBS.values() for v in vs))
# Git history operations have no file target to name, so the verb carries the
# whole authorization - and a generic one isn't enough: "delete files only
# there" in a task framing must not authorize a planted `git reset --hard`.
GIT_VERBS = {
    "reset": ["reset", "roll back", "rollback", "revert", "undo", "rewind", "go back"],
    "clean": ["git clean", "clean", "untracked"],
    "push": ["force push", "force-push", "force"],
    "branch": ["delete", "remove", "force-delete", "drop"],
    "stash": ["drop", "clear", "discard"],
}
# Tool-generated and regenerated on demand: deleting them loses nothing.
REGENERABLE = re.compile(r"^(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.cache|.*\.pyc)$", re.I)
# A delete or drop with no target we can read is never "requested" on the
# strength of a verb alone: "delete the old logs" must not cover `xargs rm`.
NEEDS_TARGET = frozenset({"delete", "db_destroy"})
_SEVERITY = {"requested": 0, "own_artifact": 1, "unrequested": 2}

WRAPPERS = frozenset({"sudo", "doas", "env", "nohup", "time", "xargs", "exec", "call", "start", "&", ".",
                      "npx", "pnpx", "bunx", "uvx", *SHELL_KEYWORDS})
DELETE_COMMANDS = frozenset({"rm", "rmdir", "rd", "del", "erase", "unlink", "shred", "rimraf", "ri", "truncate"})
PACKAGE_MANAGERS = frozenset({"npm", "pnpm", "yarn", "bun", "cargo", "gem", "twine"})
CLI_TOOLS = frozenset({"docker", "podman", "kubectl", "helm", "terraform", "tofu", "aws", "az", "gcloud",
                       "wrangler", "fly", "flyctl", "heroku", "firebase", "supabase", "gh", "vercel", "netlify"})
CLI_DESTRUCTIVE = frozenset({"delete", "destroy", "rm", "rmi", "rb", "prune", "purge", "terminate", "drop"})
HTTP_CLIENTS = frozenset({"curl", "wget", "http", "invoke-webrequest", "invoke-restmethod", "iwr", "irm"})
REMOTE_COMMANDS = frozenset({"ssh", "scp", "sftp", "rsync"})
PROCESS_COMMANDS = frozenset({"kill", "pkill", "killall", "taskkill", "stop-process"})
MAIL_COMMANDS = ("sendmail", "mail", "mailx", "send-mailmessage")
SYSTEM_COMMANDS = ("crontab", "schtasks", "shutdown", "reboot", "halt", "poweroff", "restart-computer", "stop-computer",
                   "set-executionpolicy", "enter-pssession", "invoke-command")
TARGET_FLAGS = frozenset({"-path", "-literalpath", "-filter", "-include", "-name", "-iname"})
VALUE_FLAGS = frozenset({"-s", "--size", "-r", "--reference"})
MOVE_COMMANDS = frozenset({"mv", "move", "move-item", "mi", "ren", "rename", "rename-item", "rni", "trash", "trash-put"})
COPY_COMMANDS = frozenset({"cp", "copy", "copy-item", "cpi", "install", "rsync"})
WRITE_TOOLS = frozenset({"write", "edit", "write_file"})

_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|\'([^\']*)\'|(\S+)')
_ENV_PREFIX = re.compile(r"[A-Za-z_]\w*=")
_ASSIGNED_VAR = re.compile(r"\$[\w:]+")
_PATH_PREFIX = re.compile(r"^.*[\\/]")
_EXT_SUFFIX = re.compile(r"\.(exe|cmd|bat|ps1|sh)$")
_REMOVE_CLEAR = re.compile(r"^(remove|clear)-")
_HOST = re.compile(r"https?://([^/\s'\"?#:]+)", re.I)
_SQL_TARGET = re.compile(
    r"\b(?:drop\s+(?:table|database|schema|index|view|user)\s+(?:if\s+exists\s+)?|truncate\s+(?:table\s+)?|delete\s+from\s+)"
    r"([\w.\"`\[\]]+)", re.I)
_GIT_FORCE = re.compile(r"^(-f|--force.*|--delete|-d)$")
_REMOTE_TARGET = re.compile(r"^(?:[^@\s]+@)?([^:\s]+):")
_MKFS_FORMAT = re.compile(r"^(mkfs|format)")
_PATCH_DELETE = re.compile(r"^\*\*\* Delete File: (.+)$", re.M)
_PATCH_ADD = re.compile(r"^\*\*\* Add File: (.+)$", re.M)
_PATCH_TOUCHED = re.compile(r"^\*\*\* (?:Add|Update) File: (.+)$", re.M)
_PATCH_ANY = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$|^\*\*\* Move to: (.+)$", re.M)
_DELETE_ACTION = re.compile(r"\b(delete|remove|unsend|purge|destroy)\b", re.I)
_REDIRECT_ANY = re.compile(r"(?<![\d&>])>>?(?![&])\s*(\"[^\"]+\"|'[^']+'|[^\s;|&<>]+)")
_REDIRECT_CREATE = re.compile(r"(?<![\d&>])>(?![>&])\s*(\"[^\"]+\"|'[^']+'|[^\s;|&<>]+)")
_DEVNULL = re.compile(r"^/dev/|^nul$", re.I)
_QUOTE_ENDS = re.compile(r"^[\"']|[\"']$")
_MKDIR_IDEMPOTENT = re.compile(r"^(-p|--parents|-force)$", re.I)
_CREATE_TABLE = re.compile(r"\bcreate\s+table\s+(?!if\s+not\s+exists)([\w.\"`]+)", re.I)
_PLANTED_PREFIX = re.compile(r"^(?:[-*>]|\d+[.)]|\$|#|PS [^>]*>)\s*")
_BACKTICKS = re.compile(r"^`+|`+$")
_GLOB = re.compile(r"[*?\[\]]")
_GLOB_OR_VAR = re.compile(r"[*?\[\]]|^\$")
_RELATIVE_PREFIX = re.compile(r"^(\.\.?/|~/)+")
_EXT = re.compile(r"(\.[a-z0-9]{1,6})$", re.I)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n+")
# owns_files: the only commands, flags and target shapes it accepts.
_OWN_DELETE_COMMANDS = frozenset({"rm", "unlink", "del", "erase", "remove-item", "ri"})
_PLAIN_DELETE_FLAG = re.compile(r"^(-f|-v|-fv|-vf|--force|--verbose|-force|/f|/q)$", re.I)
_RECURSIVE_FLAG = re.compile(r"^(-(?=[rfv]*r)[rfv]+|--recursive|-recurse)$", re.I)
_PATH_FLAG = re.compile(r"^-(path|literalpath)$", re.I)
_CD_COMMANDS = frozenset({"cd", "chdir", "pushd", "set-location", "sl"})
_NULL_REDIRECT = re.compile(r"^(\d?>>?|&>)(/dev/null|nul|&\d)$|^\d?>&\d$", re.I)
_HIDDEN_COMMAND = re.compile(r"`|\$\(")
_UNSAFE_TARGET = re.compile(r"[*?\[\]{}$`~,]|(^|[/\\])\.\.([/\\]|$)|^-")
_ABSOLUTE = re.compile(r"^([/\\]|[a-z]:)", re.I)
_UNREADABLE_MOVE = re.compile(r"\bxargs\b|\bfind\b.*-exec|\bparallel\b", re.I)


def _resolve(cwd: str, path: str) -> str:
    """path against a relative working folder, as one normalized relative (or absolute) path."""
    p = path.replace("\\", "/")
    if cwd and not _ABSOLUTE.match(p):
        p = f"{cwd}/{p}"
    return ("/" if p.startswith("/") else "") + "/".join(x for x in p.split("/") if x not in ("", "."))


def _shell_steps(p: dict) -> list[tuple[str | None, str, list[str], str]] | None:
    """A shell call's commands, each with the folder it runs in: (cwd, cmd, args,
    segment). cwd is relative to the agent's workspace ("" = the workspace), or
    None once a cd (or the call's workdir) goes somewhere we can't follow."""
    text = _shell_text(p)
    if text is None:
        return None
    wd = _first_present(p, "workdir", "cwd")
    cwd: str | None = ""
    if isinstance(wd, str) and wd.strip():
        cwd = _resolve("", wd.strip()) if not _UNSAFE_TARGET.search(wd.strip()) else None
    steps = []
    for segment in _split_segments(text):
        cmd, args = _command(segment)
        if cmd in _CD_COMMANDS:
            plain = [a for a in args if not a.startswith("-")]
            safe = len(plain) == 1 and not _UNSAFE_TARGET.search(plain[0])
            cwd = _resolve(cwd, plain[0]) if cwd is not None and safe else None
            continue
        steps.append((cwd, cmd, args, segment))
    return steps


def _split_segments(command: str) -> list[str]:
    """Split a shell command on ; && || | and newlines, respecting quotes
    (unlike risk.py, which strips quoted text; here quoted paths and SQL are
    the point)."""
    out: list[str] = []
    cur = ""
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            cur += ch
            if ch == "\\" and quote == '"' and i + 1 < n:
                i += 1
                cur += command[i]
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            cur += ch
            i += 1
            continue
        if ch in "()":  # `(a || rm -rf b)` subshells
            out.append(cur)
            cur = ""
            i += 1
            continue
        if ch == "\n" or ch == ";" or ch == "|" or (ch == "&" and i + 1 < n and command[i + 1] == "&"):
            out.append(cur)
            cur = ""
            if i + 1 < n and command[i + 1] == ch:
                i += 1
            i += 1
            continue
        cur += ch
        i += 1
    out.append(cur)
    return [s.strip() for s in out if s.strip()]


def _tokens(segment: str) -> list[str]:
    out = []
    for m in _TOKEN.finditer(segment):
        out.append(m.group(1) if m.group(1) is not None else m.group(2) if m.group(2) is not None else m.group(3))
    return out


def _command(segment: str) -> tuple[str, list[str]]:
    """The segment's command (bare name, lower case) and its arguments as written."""
    all_tokens = _tokens(segment)
    i = 0
    while i < len(all_tokens) and (
        all_tokens[i].lower() in WRAPPERS or _ENV_PREFIX.match(all_tokens[i])
        or (_ASSIGNED_VAR.fullmatch(all_tokens[i]) and i + 1 < len(all_tokens) and all_tokens[i + 1] == "=")
    ):
        i += 2 if all_tokens[i].startswith("$") else 1
    rest = all_tokens[i:]
    cmd = _EXT_SUFFIX.sub("", _PATH_PREFIX.sub("", (rest[0] if rest else "").lower()))
    return cmd, rest[1:]


def _plain_args(raw: list[str], cmd: str | None = None) -> list[str]:
    """Arguments that are targets: flags dropped, PowerShell lists (`a, b`)
    split. cmd scopes VALUE_FLAGS: they take a value only for truncate;
    everywhere else -r is recursive, and skipping the next word lost `build`
    from `rm -r build`."""
    args: list[str] = []
    for a in raw:
        if a.startswith("-"):
            args.append(a)
        else:
            args.extend(s.strip() for s in a.split(",") if s.strip())
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        lower = a.lower()
        if lower in TARGET_FLAGS:
            if i + 1 < len(args):
                i += 1
                out.append(args[i])
            i += 1
            continue
        if cmd == "truncate" and lower in VALUE_FLAGS:
            i += 2
            continue
        if not a.startswith("-"):
            out.append(a)
        i += 1
    return out


def _hosts_in(text: str) -> list[str]:
    return _HOST.findall(text)


def _sql_targets(text: str) -> list[str]:
    return [re.sub(r"[`\"\[\]]", "", m) for m in _SQL_TARGET.findall(text)]


def _segment_operation(segment: str) -> dict | None:
    if classify_shell_command(segment) == "none":
        return None
    if SQL_CLIENT.search(segment) and SQL_DESTRUCTIVE.search(segment):
        return {"kind": "db_destroy", "targets": _sql_targets(segment)}
    cmd, args = _command(segment)
    plain = _plain_args(args, cmd)
    if cmd in DELETE_COMMANDS or _REMOVE_CLEAR.match(cmd):
        return {"kind": "delete", "targets": plain}
    if cmd == "find":
        start = next((i for i, a in enumerate(args) if a.startswith("-")), -1)
        head = args if start == -1 else args[:start]
        return {"kind": "delete", "targets": [*head, *[a for a in _plain_args(args[max(start, 0):]) if "*" in a]]}
    if cmd == "git":
        sub = plain[0] if plain else None
        if sub == "push" and not any(_GIT_FORCE.match(a) for a in args):
            return {"kind": "publish", "targets": []}
        if sub == "branch":
            return {"kind": "git_rewrite", "verbs": GIT_VERBS["branch"], "targets": plain[1:]}
        return {"kind": "git_rewrite", "verbs": GIT_VERBS.get(sub), "targets": []}
    if cmd in PACKAGE_MANAGERS:
        if "unpublish" in plain:
            return {"kind": "delete", "targets": plain[plain.index("unpublish") + 1:]}
        return {"kind": "publish", "targets": []}
    if cmd in CLI_TOOLS:
        verb = next((i for i, v in enumerate(plain) if v.lower() in CLI_DESTRUCTIVE), -1)
        return {"kind": "publish", "targets": []} if verb == -1 else {"kind": "delete", "targets": plain[verb + 1:]}
    if cmd in HTTP_CLIENTS:
        return {"kind": "send", "targets": _hosts_in(segment)}
    if cmd in REMOTE_COMMANDS:
        targets = []
        for a in plain:
            m = _REMOTE_TARGET.match(a)
            t = m.group(1) if m else (re.sub(r"^[^@]+@", "", a) if cmd == "ssh" else None)
            if t:
                targets.append(t)
        return {"kind": "send", "targets": targets[:1]}
    if cmd in PROCESS_COMMANDS:
        return {"kind": "process", "targets": []}
    if cmd in MAIL_COMMANDS:
        return {"kind": "send", "targets": []}
    if cmd in SYSTEM_COMMANDS or _MKFS_FORMAT.match(cmd):
        return {"kind": "system", "targets": []}
    return {"kind": "generic", "targets": []}


def _params(params: Any) -> dict:
    return params if isinstance(params, dict) else {}


def _first_present(p: dict, *keys: str) -> Any:
    for k in keys:
        if p.get(k) is not None:
            return p[k]
    return None


def _shell_text(p: dict) -> str | None:
    text = _first_present(p, "command", "cmd", "input")
    return text if isinstance(text, str) else None


def _patch_text(p: dict) -> str:
    return p["input"] if isinstance(p.get("input"), str) else p["patch"] if isinstance(p.get("patch"), str) else ""


def operations(tool_name: str, params: Any) -> list[dict]:
    """The risky operations a tool call performs, each {kind, targets[, verbs]}."""
    p = _params(params)
    if tool_name in SHELL_TOOLS:
        text = _shell_text(p)
        if text is None:
            return []
        return [op for op in (_segment_operation(s) for s in _split_segments(text)) if op]
    if tool_name == "apply_patch":
        patch = _patch_text(p)
        deleted = [m.strip() for m in _PATCH_DELETE.findall(patch)]
        if deleted:
            return [{"kind": "delete", "targets": deleted}]
        return [{"kind": "credentials", "targets": [m.strip() for m in _PATCH_TOUCHED.findall(patch)]}]
    if tool_name in WRITE_TOOLS:
        path = _first_present(p, "file_path", "path")
        return [{"kind": "credentials", "targets": [path] if isinstance(path, str) else []}]
    if tool_name in ("message", "conversations_send"):
        action = p.get("action") if isinstance(p.get("action"), str) else ""
        return [{"kind": "delete" if _DELETE_ACTION.search(action) else "send", "targets": []}]
    if tool_name == "github_publish":
        return [{"kind": "publish", "targets": []}]
    return [{"kind": "system", "targets": []}]


def touched_paths(tool_name: str, params: Any) -> list[str]:
    """Every path a call would move, rename, overwrite, write into or delete -
    not just the ones risk.py calls destructive. The gate uses it after a
    hold: an agent refused `rm -rf data` often reaches for `mv data data.bak`
    next, which makes the data vanish from where it was just as surely."""
    p = _params(params)
    if tool_name in WRITE_TOOLS:
        path = _first_present(p, "file_path", "path")
        return [path] if isinstance(path, str) else []
    if tool_name == "apply_patch":
        return [(a or b).strip() for a, b in _PATCH_ANY.findall(_patch_text(p))]
    if tool_name not in SHELL_TOOLS:
        return []
    text = _shell_text(p)
    if text is None:
        return []
    out: list[str] = []
    for segment in _split_segments(text):
        cmd, args = _command(segment)
        plain = _plain_args(args, cmd)
        if cmd in MOVE_COMMANDS or cmd in DELETE_COMMANDS or _REMOVE_CLEAR.match(cmd) or cmd == "shred":
            out.extend(plain)
        elif cmd in COPY_COMMANDS and len(plain) > 1:
            out.append(plain[-1])  # the destination is what gets overwritten
        for m in _REDIRECT_ANY.findall(segment):
            out.append(_QUOTE_ENDS.sub("", m))
    return [t for t in out if not _DEVNULL.search(t)]


def relocated_paths(tool_name: str, params: Any) -> list[str]:
    """Paths a call makes vanish from where they were: deleted, trashed, or
    moved away (a move's sources, not its destination). Narrower than
    touched_paths - writing into a folder doesn't make it disappear."""
    p = _params(params)
    if tool_name == "apply_patch":
        return [m.strip() for m in _PATCH_DELETE.findall(_patch_text(p))]
    if tool_name not in SHELL_TOOLS:
        return []
    text = _shell_text(p)
    if text is None:
        return []
    out: list[str] = []
    for segment in _split_segments(text):
        cmd, args = _command(segment)
        plain = _plain_args(args, cmd)
        if cmd in MOVE_COMMANDS:
            out.extend(plain[:-1] if len(plain) > 1 and not cmd.startswith("trash") else plain)
        elif cmd in DELETE_COMMANDS or _REMOVE_CLEAR.match(cmd) or cmd == "shred":
            out.extend(plain)
    return out


def result_text(result: Any, budget: int = 200_000) -> str:
    """The text of a tool result, whatever shape the framework hands it over
    in: a string, {content: [{type: "text", text}]}, or nested details."""
    parts: list[str] = []
    size = 0

    def walk(v: Any, depth: int) -> None:
        nonlocal size
        if size >= budget or depth > 4 or v is None:
            return
        if isinstance(v, str):
            parts.append(v[: budget - size])
            size += len(v)
            return
        if isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
            return
        if isinstance(v, dict):
            for x in v.values():
                walk(x, depth + 1)

    walk(result, 0)
    return "\n".join(parts)


def planted_targets(text: Any) -> list[str]:
    """Targets that text the agent read tells it to delete: a README setup
    step, a web page, a tool's output. Only literal commands count (`rm -rf
    ../customer-data`, maybe as a list item, prompt line or inline code),
    never prose, so a page that merely talks about deleting adds nothing."""
    if not isinstance(text, str) or not text:
        return []
    out: list[str] = []
    for raw in re.split(r"\r?\n", text[:200_000]):
        line = _BACKTICKS.sub("", _PLANTED_PREFIX.sub("", raw.strip())).strip()
        if not line or len(line) > 500:
            continue
        try:
            for op in operations("exec", {"command": line}):
                if op["kind"] == "delete":
                    out.extend(op["targets"])
        except Exception:  # noqa: BLE001 - not a command
            pass
        if len(out) >= 50:
            break
    # Deduplicated: a result often carries its text twice (content and details).
    return list(dict.fromkeys(t for t in out if t and not _GLOB_OR_VAR.search(t)))


def _norm_path(p: Any) -> str:
    return re.sub(r"/+$", "", re.sub(r"^\./", "", str(p).replace("\\", "/"))).lower()


def _tail(p: Any) -> str:
    return _RELATIVE_PREFIX.sub("", _norm_path(p))


def _mentionable(target: Any) -> str | None:
    """The last path segment, minus glob characters: what a user would name."""
    last = _GLOB.sub("", re.sub(r"/+$", "", str(target).replace("\\", "/")).split("/")[-1])
    return last if last and last not in (".", "..") and len(last) >= 2 else None


def covers_path(held: Any, path: Any) -> bool:
    """Relative and absolute spellings of one path share their tail, so
    compare tails: `../customer-data`, `./customer-data` and
    `/home/u/customer-data` all cover the same folder, and so does anything
    inside it."""
    h, t = _tail(held), _tail(path)
    if not h or not t or _GLOB.search(h) or not _mentionable(h):
        return False
    return t == h or t.endswith(f"/{h}") or h.endswith(f"/{t}") or t.startswith(f"{h}/") or f"/{h}/" in t


def _escape(s: str) -> str:
    escaped = re.sub(r"[.*+?^${}()|\[\]\\]", lambda m: "\\" + m.group(0), s)
    return re.sub(r"\s+", lambda m: r"\s+", escaped)


def _mentions(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w-]){_escape(phrase)}(?![\w-])", text, re.I) is not None


def _owned(created: dict, target: str) -> bool:
    if _GLOB.search(target):
        return False
    t = _norm_path(target)
    if REGENERABLE.search(t.split("/")[-1]):
        return True
    return any(c == t or c.endswith(f"/{t}") or t.endswith(f"/{c}") for c in created)


def _named(text: str, name: str) -> bool:
    """Named outright, or covered by an extension the user named ("delete the
    .tmp files" covers a.tmp)."""
    m = _EXT.search(name)
    ext = m.group(1) if m else None
    return _mentions(text, name) or (ext is not None and ext != name and _mentions(text, ext))


def _sentences(text: str) -> list[str]:
    """Sentence ends: . ! ? before whitespace or the end (so notes.txt and
    httpbin.org/post don't split), and line breaks."""
    return _SENTENCE_END.split(text)


def _named_with_verb(text: str, name: str, verbs: list[str]) -> bool:
    """A target counts as requested only when a matching verb sits in the SAME
    sentence as it. A blanket permission ("create, change and delete files")
    in one sentence plus the target named for another reason in the next must
    not combine into authorization."""
    return any(_named(s, name) and any(_mentions(s, v) for v in verbs) for s in _sentences(text))


class _Session:
    __slots__ = ("requests", "request_seen", "paths", "tables", "files", "moved_names", "moves_unreadable")

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.request_seen = False
        self.paths: dict[str, None] = {}  # insertion-ordered sets
        self.tables: dict[str, None] = {}
        self.files: dict[str, None] = {}  # single files created, no move since (resolved paths)
        self.moved_names: dict[str, None] = {}  # every path component any move has named
        self.moves_unreadable = False  # a move named things we couldn't read


def _judge(op: dict, session: _Session) -> str:
    text = "\n".join(session.requests)
    names = [n for n in (_mentionable(t) for t in op["targets"]) if n]
    verbs = op.get("verbs") or VERBS[op["kind"]]
    if names:
        requested = len(names) == len(op["targets"]) and all(_named_with_verb(text, n, verbs) for n in names)
    else:
        requested = op["kind"] not in NEEDS_TARGET and any(_mentions(text, v) for v in verbs)
    if text and requested:
        return "requested"
    pool = session.tables if op["kind"] == "db_destroy" else session.paths
    if op["targets"] and all(_owned(pool, t) for t in op["targets"]):
        return "own_artifact"
    return "unrequested"


def _creations(tool_name: str, params: Any) -> tuple[list[str], list[str], list[str]]:
    """Things a successful call created, so later deletes of them read as the
    agent's own scratch work: (paths, tables, files), files being the paths
    that are single files (everything but mkdir). Deliberately excludes
    idempotent forms that also succeed on something pre-existing (mkdir -p,
    touch, CREATE TABLE IF NOT EXISTS, git init, >> appends), which would let
    an existing target pass as "created"."""
    p = _params(params)
    paths: list[str] = []
    tables: list[str] = []
    files: list[str] = []  # resolved against the call's working folder, for owns_files
    if tool_name in ("write", "write_file"):
        path = _first_present(p, "file_path", "path")
        if isinstance(path, str):
            paths.append(path)
            files.append(_resolve("", path))
    elif tool_name == "apply_patch":
        added = [m.strip() for m in _PATCH_ADD.findall(_patch_text(p))]
        paths.extend(added)
        files.extend(_resolve("", a) for a in added)
    elif tool_name in SHELL_TOOLS:
        steps = _shell_steps(p)
        if steps is None:
            return paths, tables, files
        for cwd, cmd, args, segment in steps:
            if cmd in ("mkdir", "md") and not any(_MKDIR_IDEMPOTENT.match(a) for a in args):
                paths.extend(_plain_args(args, cmd))
            for m in _REDIRECT_CREATE.findall(segment):
                target = _QUOTE_ENDS.sub("", m)
                if not _DEVNULL.search(target):
                    paths.append(target)
                    if cwd is not None and not _UNSAFE_TARGET.search(target):
                        files.append(_resolve(cwd, target))
            if SQL_CLIENT.search(segment):
                tables.extend(re.sub(r"[`\"]", "", m) for m in _CREATE_TABLE.findall(segment))
    return paths, tables, files


def _is_move(cmd: str, args: list[str]) -> bool:
    return (cmd in MOVE_COMMANDS or (cmd == "git" and args[:1] == ["mv"])
            or (cmd == "rsync" and "--remove-source-files" in args))


def _moves(tool_name: str, params: Any) -> tuple[list[str], bool]:
    """The paths a call's visible moves name (sources and destinations, resolved
    against its working folder), and whether it moved things we can't name
    (globs, xargs, find -exec, a folder we lost track of)."""
    if tool_name not in SHELL_TOOLS:
        return [], False
    steps = _shell_steps(_params(params))
    if steps is None:
        return [], False
    named: list[str] = []
    unreadable = False
    for cwd, cmd, args, segment in steps:
        if not _is_move(cmd, args):
            continue
        plain = [a for a in (args[1:] if cmd == "git" else args) if not a.startswith("-")]
        if (cwd is None or not plain or _UNREADABLE_MOVE.search(segment)
                or any(_GLOB_OR_VAR.search(a) or _UNSAFE_TARGET.search(a) for a in plain)):
            unreadable = True
        named.extend(_resolve(cwd or "", a) for a in plain)
    return named, unreadable


class AuthorizationTracker:
    """Per-session memory of the user's own turns and of what the agent
    created, and the who-asked label for each risky call."""

    def __init__(self, max_sessions: int = 200, requests_kept: int = 3, max_created: int = 1000) -> None:
        self._sessions: dict[str, _Session] = {}  # insertion order doubles as LRU order
        self._max_sessions = max_sessions
        self._requests_kept = requests_kept
        self._max_created = max_created

    def _state(self, session_key: str) -> _Session:
        s = self._sessions.pop(session_key, None) or _Session()
        self._sessions[session_key] = s
        while len(self._sessions) > self._max_sessions:
            del self._sessions[next(iter(self._sessions))]
        return s

    def _bounded(self, store: dict[str, None], items: list[str]) -> None:
        for item in items:
            store[_norm_path(item)] = None
            if len(store) > self._max_created:
                del store[next(iter(store))]

    def set_request(self, session_key: str | None, prompt: Any, provenance: str | None = None) -> bool:
        """One user turn. Returns whether it was accepted as the user's own words."""
        if not session_key or not isinstance(prompt, str):
            return False
        s = self._state(session_key)
        s.request_seen = True
        if provenance and provenance != "external_user":
            return False
        s.requests.append(prompt)
        if len(s.requests) > self._requests_kept:
            s.requests.pop(0)
        return True

    def label(self, session_key: str | None, tool_name: str, params: Any) -> str | None:
        risk = classify_tool_call(tool_name, params)
        if risk not in ("destructive", "sensitive"):
            return None
        s = self._sessions.get(session_key) if session_key else None
        if not s or not s.request_seen:
            return None
        ops = operations(tool_name, params) or [{"kind": "generic", "targets": []}]
        worst = "requested"
        for op in ops:
            lab = _judge(op, s)
            if _SEVERITY[lab] > _SEVERITY[worst]:
                worst = lab
        return worst

    def requests_target(self, session_key: str | None, target: Any) -> bool:
        """Whether the user's own turns name this target with a delete or move
        verb in one sentence - what lets a held target be touched after all."""
        s = self._sessions.get(session_key) if session_key else None
        name = _mentionable(target)
        if not s or not s.requests or not name:
            return False
        return _named_with_verb("\n".join(s.requests), name, [*VERBS["delete"], "move", "rename", "mv"])

    def owns_files(self, session_key: str | None, tool_name: str, params: Any) -> bool:
        """Whether this call only deletes single files the agent created this
        session and no move has touched since, or tool caches no move has
        named (see the module docstring)."""
        s = self._sessions.get(session_key) if session_key else None
        if not s or tool_name not in SHELL_TOOLS:
            return False
        p = _params(params)
        text = _shell_text(p)
        if text is None or _HIDDEN_COMMAND.search(text):
            return False
        steps = _shell_steps(p) or []
        deleted = False
        for cwd, cmd, args, segment in steps:
            if cmd not in _OWN_DELETE_COMMANDS:
                # Company: harmless, and never a move (see the module docstring).
                if (_is_move(cmd, args) or _UNREADABLE_MOVE.search(segment)
                        or classify_shell_command(segment) != "none"):
                    return False
                continue
            if cwd is None:
                return False
            targets: list[str] = []
            recursive = False
            i = 0
            while i < len(args):
                a = args[i]
                if _NULL_REDIRECT.match(a):
                    i += 1
                    continue
                if a.startswith((">", "<")) or a[:1].isdigit() and ">" in a:
                    return False
                if _PATH_FLAG.match(a) and i + 1 < len(args):
                    targets.append(args[i + 1])
                    i += 2
                    continue
                if _RECURSIVE_FLAG.match(a):
                    recursive = True
                elif a.startswith("-") or (cmd in ("del", "erase") and a.startswith("/")):
                    if not _PLAIN_DELETE_FLAG.match(a):
                        return False
                else:
                    targets.append(a)
                i += 1
            if not targets:
                return False
            for t in targets:
                if _UNSAFE_TARGET.search(t):
                    return False
                path = _norm_path(_resolve(cwd, t))
                name = path.rsplit("/", 1)[-1]
                cache = REGENERABLE.match(name) and not s.moves_unreadable and name not in s.moved_names
                if not cache and (recursive or path not in s.files):
                    return False
            deleted = True
        return deleted

    def record_completed(self, session_key: str | None, tool_name: str, params: Any, failed: bool = False) -> None:
        if not session_key:
            return
        # A move - even a failed one, which may have moved part of its sources -
        # can land something else on a file the agent created, or inside a cache.
        named, unreadable = _moves(tool_name, params)
        if named or unreadable:
            s = self._state(session_key)
            hit = {_norm_path(n) for n in named}
            if unreadable:
                s.files.clear()
                s.moves_unreadable = True
            else:
                names = {h.rsplit("/", 1)[-1] for h in hit}
                for f in list(s.files):
                    if f in hit or any(f.startswith(f"{h}/") for h in hit) or f.rsplit("/", 1)[-1] in names:
                        del s.files[f]
            self._bounded(s.moved_names, [part for h in hit for part in h.split("/") if part])
        if failed:
            return
        paths, tables, files = _creations(tool_name, params)
        if not paths and not tables:
            return
        s = self._state(session_key)
        self._bounded(s.paths, paths)
        self._bounded(s.tables, tables)
        self._bounded(s.files, files)

    def end_session(self, session_key: str) -> None:
        self._sessions.pop(session_key, None)
