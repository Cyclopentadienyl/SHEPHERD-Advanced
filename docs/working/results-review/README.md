# Results review — specifications

Design for the clinician-facing results review: how a diagnosis result is
snapshotted, reopened, sorted, filtered and explained, and what the deployment
must be secured against before any of it is exposed.

**Status:** under review. Spec 2 §§5 and 7 freeze once the closure audit below
completes; everything else is draft.

| File | Scope |
|---|---|
| `SPEC_0_INDEX.md` | Index, requirements traceability, gates, open institutional values, backlog |
| `SPEC_1_RESULTS_REVIEW.md` | Scorer/view authority, SP sort/filter policy, decomposition, two-surface UX, limits |
| `SPEC_2_SNAPSHOT_REPOSITORY.md` | Payload schema, rotation, retention, atomic publication, access decision. **§11 carries the six normative invariants and the six reopen triggers** |
| `SPEC_3_EVIDENCE_AND_AUDIT.md` | Constraints any Gate 3 evidence design must satisfy |
| `SPEC_4_DEPLOYMENT_SECURITY.md` | Route inventory, bind modes, authentication, CORS, risk acceptance |
| `ARCHIVE_rev1_6.md` | Historical record of revisions 1–6. **Not review surface**, not an implementation authority |

**Authority above all of these:** `docs/DISEASE_SCORER_POLICY.md` and
`docs/SP_SCORE_GUIDE.md`.

---

## The closure audit

Sections were previously quoted into review submissions by a generator whose
extraction rule ended a block at the next heading of *any* level. A parent
section was therefore cut off at its first subsection, and submissions were
overwritten in place, so which of them were incomplete cannot be reconstructed.

The rule is fixed — a block ends at the first heading of level **≤** its own —
and carries regression tests. What remains is to confirm that no finding was
missed because its section arrived truncated. That audit is bounded to the
sections the two rules disagree on:

```
cd tools
python affected_sections.py            # the 13 sections, and the five file hashes
python affected_sections.py --show 2:7 # any one of them, complete
python spec_extract_tests.py           # the extractor's regression tests
```

Both rules live in `tools/affected_sections.py`; the old one exists only for that
comparison and produces no output anywhere else. The list is derived at run time
rather than written down, so it cannot claim coverage it does not have.

**Scope of the audit:** hidden subsections, contradictions, and findings that
depend on text that was not visible. Settled decisions are not reopened, and the
archive is not part of it.

## After the freeze

Spec 2 §§5 and 7 reopen only on triggers T1–T6, listed in that document's §11.2.
Helper structure, branch mechanics and naming are code-review matters and do not
open a specification round.
