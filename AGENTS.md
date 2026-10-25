# Working rules for agents

For every agent working on this repository: the author (Claude) and the reviewer (Codex) alike. Adopted by the owner on
2026-10-24. `CLAUDE.md` loads this file into every Claude Code session.

## The scale of a fix

1. **A small fix follows one path:** a few sentences stating its scope and its pass condition; reproduce the problem;
   fix it at its shared root cause; run the related tests and the checks next to them; one external review. No large
   plan document.
2. **Verification has a stop condition:** the original problem is caught by a regression test, the fix passes it, and
   no related failure is left unexplained. Then stop: the same checks are not rerun for more confidence.
3. **Verification widens only for a concrete reason:** a newly found dependency, a cross-module contract or a high-risk
   behaviour gets the checks that match it. That does not automatically mean several agents and the full suite.
4. **A small fix that runs long checks its method first:** past about 30–45 minutes, report briefly where it is stuck
   and what question the next check answers. A progress report, not a request for approval.

A minimal change at the shared root cause is a root-cause fix: it does not require refactoring the code around it.

## WebUI work

The WebUI's design rules are the project skill `.claude/skills/webui-design/SKILL.md`. The rules above come first: the
skill says how to do the UI work a task asks for, picks its checks by risk, and never widens the task's scope.
