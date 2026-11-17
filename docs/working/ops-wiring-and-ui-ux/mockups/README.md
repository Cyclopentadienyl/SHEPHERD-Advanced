# The round's UI mockups (T01), made before the screens were built

**Self-checked, not owner-approved.** These are the static mockups the ops-wiring-and-ui-ux round made
first, on 2026-11-16 at base `ba6e85e`, before its UI work built the screens they show: D01 the navigation
(built in T04), D04 and D05 the shared model list and the Diagnosis Load drawer (T05), D06 and D07
Settings › Data locations (T03). Their author checked them against the WebUI skill and recorded the
numbers; nobody else has reviewed them, and the owner has not approved them. They do not count as the
owner's visual acceptance of any screen (B27-P).

What is here:

- `NOTES.md`: what each mockup shows, the choices it makes, its self-check against the skill, and the
  findings it recorded (F1 to F4), as written when the mockups were made.
- `scripts/`: the mockups' sources. Each `*.cjs` opens the running app in Chromium (Playwright) and
  injects the mockup's CSS and DOM into the real page, then takes its screenshots and measures it
  (`common.cjs` holds the shared tokens and the self-check). `train_families.py` and `dump_family.py`
  built the multi-family demo the model list draws from.
- `shots/*_check.json`: the self-check's measurements behind `NOTES.md`'s numbers.
- `work/family.json`: the server's family listing of that demo, which `tree.cjs` draws from.

Not here: the PNG screenshots (about 4 MB), which stayed in the session's scratch directory. Running a
script against a demo service makes them again: the scripts expect the service at
`http://127.0.0.1:8871/ui/` (mode M1, data folders set), `8872` (M1, nothing set) and `8873` (not M1).

The skill's §12 checklist now asks for this step on every new screen or layout change (a mockup first,
saying who checked it) and for before and after screenshots when the work is delivered.
