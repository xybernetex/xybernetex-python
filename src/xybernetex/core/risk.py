"""Classifies a tool call's risk on the user's machine, from what it actually
does: "destructive", "sensitive", "none", or None when this module doesn't
know the tool (the policy service then judges it by name alone).

A port of the OpenClaw plugin's src/risk.js; tests/test_risk.py holds that
file's test cases, so the two behave identically. Why the label is computed
here: tool parameters never leave the machine, so the risk has to be judged
locally and only the label sent. Agent frameworks run every shell command
through one tool, so the command text is what says whether a call runs a
test suite or `rm -rf`.

Deliberately conservative and simple: a first-word command match per shell
segment plus a few argument patterns. It will miss obfuscated commands; it's
a heuristic layer, not a security boundary.
"""
from __future__ import annotations

import re
from typing import Any

SHELL_TOOLS = frozenset({"exec", "terminal", "shell", "bash", "run_command"})

# Tools whose calls never have an external or irreversible effect of their own.
NO_RISK_TOOLS = frozenset({
    "read", "ls", "web_fetch", "web_search", "pdf", "view_image", "image_generate", "music_generate",
    "video_generate", "tts", "sessions_list", "sessions_history", "sessions_search", "session_status",
    "sessions_spawn", "sessions_yield", "sessions_send", "agents_list", "agents_wait", "subagents",
    "conversations_list", "conversations_turn", "create_goal", "get_goal", "update_goal", "progress_card",
    "show_widget", "structured_output", "suggest_task", "dismiss_task", "heartbeat_respond", "ask_user",
    "dashboard", "theme", "transcripts", "github_identity_status", "process", "skill_workshop",
    "read_file", "list_files",
})

# Tools that act outside the machine (send, publish, reconfigure) unless the
# call is plainly a read.
OUTWARD_TOOLS = frozenset({"message", "conversations_send", "github_publish", "gateway", "cron", "plugins", "secrets"})
READ_ACTION = re.compile(r"^(get|list|read|search|status|history|fetch|show|describe|runs|schema|view|inspect)\b", re.I)
DELETE_ACTION = re.compile(r"\b(delete|remove|unsend|purge|destroy)\b", re.I)

# Files whose contents are credentials or config: editing them is sensitive.
SENSITIVE_PATH = re.compile(
    r"(^|[\\/])(\.ssh|\.aws|\.gnupg|\.kube|\.docker)([\\/]|$)"
    r"|(^|[\\/])(\.env(\.[\w-]+)?|id_rsa|id_ed25519|credentials(\.json)?|openclaw\.json|\.npmrc|\.pypirc|\.netrc)$", re.I)

# First word of a shell segment -> destructive.
DESTRUCTIVE_COMMANDS = frozenset({
    "rm", "rmdir", "rd", "del", "erase", "unlink", "shred", "rimraf", "dd", "mkfs", "fdisk", "diskpart",
    "format", "wipefs", "truncate", "shutdown", "reboot", "halt", "poweroff", "ri", "restart-computer",
    "stop-computer", "clear-content", "clear-recyclebin", "format-volume", "clear-disk", "initialize-disk",
})
# PowerShell Remove-*/Clear-* verbs are destructive except these session-only ones.
HARMLESS_REMOVE_CLEAR = frozenset({
    "clear-host", "clear-variable", "clear-history", "remove-variable", "remove-module", "remove-psdrive",
    "remove-job", "remove-event", "remove-typedata",
})

# First word -> sensitive (effects outside the machine, or on other processes).
SENSITIVE_COMMANDS = frozenset({
    "ssh", "scp", "sftp", "rsync", "sendmail", "mail", "mailx", "send-mailmessage", "crontab", "schtasks",
    "set-executionpolicy", "vercel", "netlify", "enter-pssession", "invoke-command",
    "kill", "pkill", "killall", "taskkill", "stop-process",
})

# Infra/packaging CLIs: judged by the verbs in their arguments.
CLI_TOOLS = frozenset({
    "git", "gh", "docker", "podman", "kubectl", "helm", "terraform", "tofu", "aws", "az", "gcloud", "wrangler",
    "fly", "flyctl", "heroku", "npm", "pnpm", "yarn", "bun", "cargo", "twine", "gem", "firebase", "supabase",
})
PACKAGE_MANAGERS = frozenset({"npm", "pnpm", "yarn", "bun", "cargo", "gem"})
DESTRUCTIVE_VERBS = frozenset({"delete", "destroy", "rm", "rmi", "rb", "prune", "purge", "terminate", "unpublish", "drop"})
SENSITIVE_VERBS = frozenset({"deploy", "publish", "push", "apply", "release", "upload", "merge", "send", "put", "install",
                             "upgrade", "scale", "rollout", "create", "secret"})

HTTP_CLIENTS = frozenset({"curl", "wget", "http", "invoke-webrequest", "invoke-restmethod", "iwr", "irm"})
HTTP_WRITE = re.compile(
    r"(^|\s)(-X|--request|-Method)\s*['\"]?(POST|PUT|PATCH|DELETE)\b"
    r"|(^|\s)(-d|--data(-\w+)?|--json|-F|--form|-Body|-InFile|-T|--upload-file|--post-(data|file))(\s|=|$)", re.I)

# SQL lives inside quoted arguments, so it's matched against the raw text -
# but only when a SQL client is invoked, or a commit message saying "drop
# table support" would read as destructive.
SQL_CLIENT = re.compile(
    r"(^|[\s;&|(])(psql|pgcli|mysql|mariadb|sqlite3?|litecli|sqlcmd|invoke-sqlcmd|duckdb|clickhouse(-client)?"
    r"|bq|snowsql|cockroach)(\.exe)?(\s|$)", re.I)
SQL_DESTRUCTIVE = re.compile(r"\b(drop\s+(table|database|schema|index|view|user)|truncate\s+(table\s+)?\w|delete\s+from)\b", re.I)

# Prefixes that run the next word as the real command. Shell keywords too:
# `if true; then rm -rf data; fi` must not read as a command named "then".
SHELL_KEYWORDS = ("if", "then", "else", "elif", "do", "while", "until", "!", "{")
WRAPPERS = frozenset({
    "sudo", "doas", "env", "nohup", "time", "xargs", "exec", "call", "start", "start-process", "&", ".",
    "npx", "pnpx", "bunx", "uvx", *SHELL_KEYWORDS,
})

_QUOTED = re.compile(r'"(?:[^"\\`]|[\\`].)*"|\'[^\']*\'')
_SEGMENT = re.compile(r"\r?\n|;|&&|\|\||\||[{}()]")
_ASSIGNED_VAR = re.compile(r"\$[\w:]+")
_ENV_PREFIX = re.compile(r"[A-Za-z_]\w*=")
_PATH_PREFIX = re.compile(r"^.*[\\/]")
_EXT_SUFFIX = re.compile(r"\.(exe|cmd|bat|ps1|sh)$")
_REMOVE_CLEAR = re.compile(r"^(remove|clear)-")
_GIT_CLEAN_FORCE = re.compile(r"^-\w*f")
_PATCH_DELETE = re.compile(r"^\*\*\* Delete File:", re.M)
_PATCH_TOUCHED = re.compile(r"^\*\*\* (?:Add|Update) File: (.+)$", re.M)


def _strip_quoted(command: str) -> str:
    return _QUOTED.sub('""', command)


def _segments(command: str) -> list[str]:
    return [s.strip() for s in _SEGMENT.split(_strip_quoted(command)) if s.strip()]


def _words(segment: str) -> tuple[str, list[str]]:
    """The segment's command (bare name, lower case) and its lower-cased arguments."""
    all_words = segment.split()
    i = 0
    # Skip `$x = ...` assignments, env-style VAR=value prefixes, and wrappers.
    while i < len(all_words):
        w = all_words[i].lower()
        if _ASSIGNED_VAR.fullmatch(all_words[i]) and i + 1 < len(all_words) and all_words[i + 1] == "=":
            i += 2
            continue
        prev = all_words[i - 1].lower() if i > 0 else ""
        if _ENV_PREFIX.match(all_words[i]) or w in WRAPPERS or (all_words[i].startswith("-") and prev in WRAPPERS):
            i += 1
            continue
        break
    rest = all_words[i:]
    if not rest:
        return "", []
    # `./tool`, `C:\bin\tool.exe`, `tool.cmd` -> `tool`
    cmd = _EXT_SUFFIX.sub("", _PATH_PREFIX.sub("", rest[0].lower()))
    return cmd, [a.lower() for a in rest[1:]]


def _classify_segment(segment: str, raw_command: str) -> str:
    cmd, args = _words(segment)
    if not cmd:
        return "none"
    if cmd in HARMLESS_REMOVE_CLEAR:
        return "none"
    if cmd in DESTRUCTIVE_COMMANDS or cmd.startswith("mkfs") or _REMOVE_CLEAR.match(cmd):
        return "destructive"
    if cmd == "find" and ("-delete" in args or ("-exec" in args and "rm" in args)):
        return "destructive"

    if cmd == "git":
        sub = next((a for a in args if not a.startswith("-")), None)
        if sub == "reset" and "--hard" in args:
            return "destructive"
        if sub == "clean" and any(_GIT_CLEAN_FORCE.match(a) or a == "--force" for a in args):
            return "destructive"
        if sub == "push" and any(a == "-f" or a.startswith("--force") or a == "--delete" or a == "-d" for a in args):
            return "destructive"
        if sub == "branch" and any(a in ("-D", "-d", "--delete") for a in args):
            return "destructive"
        if sub == "stash" and any(a in ("drop", "clear") for a in args):
            return "destructive"
        if sub == "push":
            return "sensitive"
        return "none"
    if cmd in CLI_TOOLS:
        verbs = [a for a in args if not a.startswith("-")]
        # Package managers only reach outside the machine when publishing;
        # installs and removals are local dependency changes.
        if cmd in PACKAGE_MANAGERS:
            if "unpublish" in verbs:
                return "destructive"
            # --dry-run performs no publish; a fully consequence-free simulation
            # is indistinguishable from a real one by verb alone.
            if "--dry-run" in args:
                return "none"
            return "sensitive" if any(v in ("publish", "deploy") for v in verbs) else "none"
        if any(v in DESTRUCTIVE_VERBS for v in verbs):
            return "destructive"
        if any(v in SENSITIVE_VERBS for v in verbs):
            return "sensitive"
        return "none"
    # Checked against the raw text: quoting is stripped from segments, and
    # `-Method "POST"` keeps its method inside quotes.
    if cmd in HTTP_CLIENTS:
        return "sensitive" if HTTP_WRITE.search(raw_command) else "none"
    if cmd in SENSITIVE_COMMANDS:
        return "sensitive"
    return "none"


# "none" < "sensitive" < "destructive". Callers compare tiers through
# meets_risk_threshold rather than duplicating the ordering.
RISK_LEVELS = ("none", "sensitive", "destructive")
_SEVERITY = {"none": 0, "sensitive": 1, "destructive": 2}


def _worst(a: str, b: str) -> str:
    return b if _SEVERITY[b] > _SEVERITY[a] else a


def meets_risk_threshold(tier: str | None, threshold: str) -> bool:
    """Whether a classify_tool_call result is at or above `threshold`. None (an
    unknown tool) never meets one: a rule that can't locally verify a call's
    risk must not fire on it."""
    if threshold not in _SEVERITY:
        raise ValueError(f"riskAtLeast must be one of {', '.join(RISK_LEVELS)}, got {threshold!r}")
    return tier is not None and _SEVERITY[tier] >= _SEVERITY[threshold]


# Commands a command line runs that quoting hides from the segments above:
# `$(...)` and backticks (expanded inside double quotes and unquoted
# heredocs), and the string handed to `bash -c`, `eval` and the like. Each is
# classified as a command line of its own; the worst tier wins.
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "ash", "fish", "pwsh", "powershell", "cmd"})
_SHELL_C_FLAG = re.compile(r"^(-[a-z]*c|-command|/c|/k)$", re.I)
_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|\'([^\']*)\'|(\S+)')
_HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][\w.-]*)\2")
_MAX_DEPTH = 3


def _balanced(text: str, start: int) -> int:
    """Index of the ) closing a $( whose body starts at `start` (or len(text))."""
    depth, quote, i = 1, None, start
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                quote = None
            elif c == "\\" and quote == '"':
                i += 1
        elif c in "'\"":
            quote = c
        elif c == "\\":
            i += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text)


def _substitutions(text: str, quotes: bool = True) -> list[str]:
    """$(...) and `...` bodies the shell would run. With quotes=False (an
    unquoted heredoc body), quote characters are literal text."""
    out: list[str] = []
    quote = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote == "'":
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if quotes and c == "'" and quote is None:
            quote = "'"
            i += 1
            continue
        if quotes and c == '"':
            quote = None if quote == '"' else '"'
            i += 1
            continue
        if quotes and quote is None and text.startswith("<<", i) and not text.startswith("<<<", i):
            m = _HEREDOC.match(text, i)
            if m:
                line_end = text.find("\n", m.end())
                body_start = n if line_end == -1 else line_end + 1
                # The rest of the heredoc's own line is ordinary command text.
                out.extend(_substitutions(text[m.end():body_start]))
                end = body_start
                while end < n:
                    nl = text.find("\n", end)
                    line = text[end: n if nl == -1 else nl]
                    if (line.lstrip("\t") if m.group(1) else line) == m.group(3):
                        break
                    end = n if nl == -1 else nl + 1
                if not m.group(2):  # unquoted delimiter: the body expands
                    out.extend(_substitutions(text[body_start:end], quotes=False))
                nl = text.find("\n", end)
                i = n if nl == -1 else nl + 1
                continue
        if text.startswith("$(", i) and not text.startswith("$((", i):
            j = _balanced(text, i + 2)
            out.append(text[i + 2:j])
            i = j + 1
            continue
        if c == "`":
            j = text.find("`", i + 1)
            j = n if j == -1 else j
            out.append(text[i + 1:j])
            i = j + 1
            continue
        i += 1
    return out


def _handed_to_shells(command: str) -> list[str]:
    """The strings given to `bash -c`, `sh -lc`, `pwsh -Command`, `cmd /c` and `eval`."""
    tokens = [m.group(1).replace('\\"', '"') if m.group(1) is not None else m.group(2) if m.group(2) is not None
              else m.group(3) for m in _TOKEN.finditer(command)]
    out: list[str] = []
    for i, tok in enumerate(tokens):
        name = _EXT_SUFFIX.sub("", _PATH_PREFIX.sub("", tok.lower()))
        if name == "eval" and i + 1 < len(tokens):
            out.append(" ".join(tokens[i + 1:]))
        elif name in _SHELLS:
            for j in range(i + 1, min(i + 4, len(tokens))):
                if _SHELL_C_FLAG.match(tokens[j]):
                    rest = tokens[j + 1:]
                    if rest:
                        out.append(" ".join(rest) if name in ("cmd", "pwsh", "powershell") else rest[0])
                    break
                if not tokens[j].startswith("-"):
                    break
    return out


def classify_shell_command(command: Any, _depth: int = 0) -> str:
    if not isinstance(command, str) or not command.strip():
        return "none"
    tier = "destructive" if SQL_CLIENT.search(command) and SQL_DESTRUCTIVE.search(command) else "none"
    for segment in _segments(command):
        tier = _worst(tier, _classify_segment(segment, command))
    if _depth < _MAX_DEPTH and tier != "destructive":
        for inner in _substitutions(command) + _handed_to_shells(command):
            tier = _worst(tier, classify_shell_command(inner, _depth + 1))
    return tier


def _first_present(p: dict, *keys: str) -> Any:
    for k in keys:
        if p.get(k) is not None:
            return p[k]
    return None


def classify_tool_call(tool_name: str, params: Any) -> str | None:
    """"destructive", "sensitive", "none", or None for a tool this module can't judge."""
    p = params if isinstance(params, dict) else {}
    if tool_name in SHELL_TOOLS:
        return classify_shell_command(_first_present(p, "command", "cmd", "input"))
    if tool_name in NO_RISK_TOOLS:
        return "none"
    if tool_name in ("write", "edit", "write_file"):
        path = _first_present(p, "file_path", "path") or ""
        return "sensitive" if isinstance(path, str) and SENSITIVE_PATH.search(path) else "none"
    if tool_name == "apply_patch":
        patch = p["input"] if isinstance(p.get("input"), str) else p["patch"] if isinstance(p.get("patch"), str) else ""
        if _PATCH_DELETE.search(patch):
            return "destructive"
        touched = [m.strip() for m in _PATCH_TOUCHED.findall(patch)]
        return "sensitive" if any(SENSITIVE_PATH.search(path) for path in touched) else "none"
    if tool_name in OUTWARD_TOOLS:
        action = p.get("action") if isinstance(p.get("action"), str) else ""
        if DELETE_ACTION.search(action):
            return "sensitive" if tool_name == "cron" else "destructive"
        if READ_ACTION.search(action):
            return "none"
        return "sensitive"
    return None  # unknown tool (plugins, MCP servers): let the policy service judge the name
