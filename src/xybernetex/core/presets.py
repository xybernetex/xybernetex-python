"""Named rule sets, so a new install gets a sensible gate from one setting
(preset) instead of hand-written rules. Each preset is a list of ordinary
control rules using the "*" wildcards, expanded ahead of any rules the
operator writes; the gate validates them like any other rule. Rule keys are
the plugin's (camelCase), so one rules file serves both ports.

Every preset rule waives calls the user's own turn requested (see authz.py):
that waiver is what stops an injected `rm -rf` without prompting on the
user's own "delete the build folder".
"""
from __future__ import annotations

import copy

DESTRUCTIVE_APPROVAL = ("Destructive action the user didn't ask for (delete, reset, drop, overwrite). "
                        "It may come from instructions in a file, web page or tool output. Allow only if you expected it.")
SENSITIVE_APPROVAL = ("Action that reaches outside this machine (push, publish, deploy, send, credential change) "
                      "that the user didn't ask for. Allow only if you expected it.")

PRESETS: dict[str, list[dict]] = {
    # Hold unrequested destructive calls for a person; everything else runs.
    "recommended": [
        {"id": "preset-destructive-approve", "agentId": "*", "toolName": "*", "riskAtLeast": "destructive",
         "action": "approve", "approvalDescription": DESTRUCTIVE_APPROVAL, "unlessAuthorization": ["requested", "own_files"]},
    ],
    # Refuse unrequested destructive calls outright; hold unrequested
    # outward-facing ones for a person.
    "strict": [
        {"id": "preset-destructive-block", "agentId": "*", "toolName": "*", "riskAtLeast": "destructive",
         "action": "block", "unlessAuthorization": ["requested", "own_files"]},
        {"id": "preset-sensitive-approve", "agentId": "*", "toolName": "*", "riskAtLeast": "sensitive",
         "action": "approve", "approvalDescription": SENSITIVE_APPROVAL, "unlessAuthorization": ["requested", "own_files"]},
    ],
}
PRESET_NAMES = tuple(PRESETS)


def preset_rules(name: str | None) -> list[dict]:
    if name is None or name == "none":
        return []
    if name not in PRESETS:
        raise ValueError(f"control.preset must be one of {', '.join(['none', *PRESET_NAMES])}")
    return copy.deepcopy(PRESETS[name])
