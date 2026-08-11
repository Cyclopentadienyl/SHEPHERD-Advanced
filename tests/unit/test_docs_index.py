"""
`docs/README.md` must actually index every Markdown document in the repository.
==============================================================================
The index opens by claiming it covers every tracked Markdown document. That claim
was false within one commit of being written — the index omitted itself — which is
exactly the failure this branch spent eight commits removing: a sentence asserting
something nothing checks.

So it is checked. Two directions, because either one alone rots:

  - every tracked ``*.md`` is linked from the index (a new document cannot be added
    without an entry);
  - every link in the index resolves to a file that exists (a renamed or deleted
    document cannot leave a dangling entry).

Deliberate exclusions are listed in EXCLUDED below and stated in the index's own
opening paragraph, so the two cannot drift apart silently.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX = REPO_ROOT / "docs" / "README.md"

# The index does not list itself. Nothing else is exempt; images are not Markdown
# and so never enter the tracked-.md set in the first place.
EXCLUDED = {"docs/README.md"}

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def _tracked_markdown() -> set:
    """Repo-relative paths of every tracked Markdown file.

    `git ls-files` rather than a filesystem walk: the claim is about the repository,
    so an untracked scratch note should not fail the suite.
    """
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "*.md"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return {p for p in completed.stdout.split("\0") if p.endswith(".md")}


def _index_link_targets() -> set:
    """Link targets in the index, normalised to repo-relative paths."""
    targets = set()
    for raw in LINK_RE.findall(INDEX.read_text(encoding="utf-8")):
        if raw.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (INDEX.parent / raw).resolve()
        try:
            targets.add(resolved.relative_to(REPO_ROOT).as_posix())
        except ValueError:  # points outside the repository
            targets.add(raw)
    return targets


try:
    TRACKED = _tracked_markdown()
except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
    pytest.skip(f"git unavailable: {exc}", allow_module_level=True)

LINKED = _index_link_targets()


def test_every_tracked_markdown_document_is_indexed():
    """A document added without an index entry fails here, not silently."""
    missing = sorted(TRACKED - LINKED - EXCLUDED)
    assert not missing, (
        "these Markdown files are tracked but not linked from docs/README.md:\n"
        + "\n".join(f"  {p}" for p in missing)
        + "\nAdd an entry with a status label, or add it to EXCLUDED here and say so "
        "in the index's opening paragraph."
    )


@pytest.mark.parametrize("target", sorted(LINKED), ids=sorted(LINKED))
def test_index_link_resolves(target):
    """A renamed or deleted document must not leave a dangling entry behind."""
    assert (REPO_ROOT / target).exists(), (
        f"docs/README.md links to {target}, which does not exist"
    )


def test_exclusions_are_declared_in_the_index_itself():
    """The index must say what it leaves out, so the code and the prose agree."""
    opening = INDEX.read_text(encoding="utf-8").split("## ", 1)[0]
    assert "this index itself" in opening, (
        "docs/README.md excludes itself from the listing; its opening paragraph must "
        "say so, otherwise the 'every Markdown document' claim is again untrue"
    )
