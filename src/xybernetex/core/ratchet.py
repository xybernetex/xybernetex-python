"""The ratchet: a fix turn may only move a run forward.

Before each fix turn the workspace is snapshotted; after it, the contract's
checks run again and `judge` compares the set of passing checks with the best
so far:

  improved   - more checks pass and none that passed now fails: keep it
  same       - the same checks pass: keep it (a round without progress)
  regressed  - something that passed now fails, whatever else improved:
               restore the snapshot and tell the agent what was undone

Sets, not counts: a fix that repairs the totals but breaks the header has
traded a known-good result for an unknown one. The run ends with the best
workspace it ever reached.

Snapshots are supplied by whoever runs the agent, like the check executor:
an object with snapshot() -> token, restore(token) and discard(token).
FolderSnapshots does it for a plain folder; the benchmark runners do it
inside the sandbox container. Only the workspace rolls back - anything a turn
did outside it (a database, the network) is the gate's to stop, not this.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol

from .contracts import Verdict, failure_message
from .followups import MARKER

IMPROVED, SAME, REGRESSED, ROLLED_BACK = "improved", "same", "regressed", "rolled-back"


class Snapshots(Protocol):
    def snapshot(self) -> Any: ...
    def restore(self, token: Any) -> None: ...
    def discard(self, token: Any) -> None: ...


def passing(verdict: Verdict | None) -> frozenset[int]:
    """Positions of the checks that passed."""
    if verdict is None:
        return frozenset()
    return frozenset(i for i, r in enumerate(verdict.results) if r.passed)


def judge(best: Verdict | None, new: Verdict) -> str:
    """improved | same | regressed, comparing which checks pass (see the module docstring)."""
    before, after = passing(best), passing(new)
    if best is not None and not before <= after:
        return REGRESSED
    return IMPROVED if after > before or best is None else SAME


def rollback_message(best: Verdict, new: Verdict) -> str:
    """The follow-up after a rolled-back fix: what broke, that it was undone, and what still fails."""
    broke = [best.results[i].check.name for i in sorted(passing(best) - passing(new))]
    head = (f"{MARKER} Your last change was undone: it made checks fail that passed before - "
            + "; ".join(broke) + ". The working folder is back exactly as it was before that change.")
    if best.passed:
        return head
    rest = failure_message(best).split("\n", 1)[1] if "\n" in failure_message(best) else ""
    return head + "\nThese checks still fail, and need fixing without breaking the ones above:" + rest


class FolderSnapshots:
    """Snapshots of a plain folder: a full copy per snapshot in a temp directory.
    restore() replaces the folder's contents with the copy - meant for a task's
    working folder, not a large repository."""

    def __init__(self, folder: str | os.PathLike):
        self.folder = Path(folder)
        self._root = Path(tempfile.mkdtemp(prefix="xyb-snap-"))
        self._n = 0

    def snapshot(self) -> Path:
        self._n += 1
        dest = self._root / str(self._n)
        shutil.copytree(self.folder, dest, symlinks=True)
        return dest

    def restore(self, token: Path) -> None:
        for child in self.folder.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in Path(token).iterdir():
            target = self.folder / child.name
            if child.is_dir() and not child.is_symlink():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target, follow_symlinks=False)

    def discard(self, token: Path) -> None:
        shutil.rmtree(token, ignore_errors=True)

    def close(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
