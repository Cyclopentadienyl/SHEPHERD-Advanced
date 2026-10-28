"""
P2-NEW-1: a late Training poll never shows, or lets Start use, another dataset
==============================================================================
The reviewer's independent reproduction (incremental review of `2887506`,
``test_training_poll_display.py``), kept as a regression test through the
pages served by Gradio: one session chooses ``demo``; the fast poll's read of
the job runner is held; meanwhile the session chooses ``resampled`` and its
queued render ends; then the poll is released. Before the fix its answer
brought back ``demo``'s select value, chart token and an enabled Start, while
the server's selection, and so Start, was ``resampled``.

Its assertions are kept; two lines are adapted to the fix's mechanism: the
poll has the session's own record as a sixth input (``polled``), so it is
asked with one value per input, and the select, the chart and Start are no
longer its outputs at all (it writes only the pill), so what its answer
writes into them is read with `_written` (nothing, for a component it does
not output) instead of ``poll.outputs.index``. Added: what the browser shows
after every answer (the select, the chart token's run, Start), the server's
selection and the directory `_on_start` hands the job runner all agree with
``resampled``, and the late answer hands nothing over.

Module: tests/unit/test_training_poll_display.py
"""
import json
import threading

from tests.unit.test_training_page import (  # noqa: F401 - fixtures
    ServedSession,
    _after,
    _answered,
    _datasets_and_training,
    _job_in_resampled,
    _shown,
    built,
    operator,
    root,
    served,
    shell,
    tc,
)


def _written(fn, answer, block):
    """What an answer of `fn` writes into `block`: nothing (``{}``) when
    `block` is not one of its outputs."""
    return answer[fn.outputs.index(block)] if block in fn.outputs else {}


def test_delayed_full_poll_cannot_restore_the_old_dataset_label(root, operator, monkeypatch):  # noqa: F811
    job = _job_in_resampled(root, monkeypatch)
    demo, ctx, named, by_id = _datasets_and_training()
    poll, opened, choose = (named[n] for n in ('_poll', '_open', '_on_dataset'))
    ws, cache = poll.inputs[1], poll.inputs[3]
    render_after_choice = _after(demo, choose, '_render_view')
    with served(demo) as base:
        browser = ServedSession(base, demo)
        try:
            assert browser.wait(browser.join(opened, [None, None]))
            first = browser.join(choose, ['demo', None])
            assert browser.wait(first)
            drawn_demo = browser.join(render_after_choice, [None] * 4)
            assert browser.wait(drawn_demo)
            browser.set_state(ctx.location, shell.TRAINING)
            before = browser.state(cache)
            before['ticks'] = tc.FULL_SEND_TICKS - 1
            browser.set_state(cache, before)
            job.hold.set()
            answers = []
            ticking = threading.Thread(
                target=lambda: answers.append(browser.predict(poll, [None] * len(poll.inputs))),
                daemon=True)
            ticking.start()
            assert job.entered.wait(20)
            second = browser.join(choose, ['resampled', None])
            assert browser.wait(second)
            drawn_resampled = browser.join(render_after_choice, [None] * 4)
            assert browser.wait(drawn_resampled)
            assert browser.state(ws)['name'] == 'resampled'
            job.release.set()
            ticking.join(20)
            assert not ticking.is_alive()
            (answer,) = answers
            dataset = _written(poll, answer, by_id['shep-train-dataset'])
            chart = _written(poll, answer, by_id['shep-train-charts'])
            start = _written(poll, answer, by_id['shep-train-start'])
            calls = []
            monkeypatch.setattr(tc.training_manager, 'start_training', lambda config, **kw: calls.append(config) or {'success': True})
            tc._on_start(operator, browser.state(ws), '', 'cpu', *tc.base_form_defaults())
            print(json.dumps({'old_poll_dropdown': dataset, 'old_poll_chart': chart,
                              'start_button': start, 'server_workspace': browser.state(ws)['name'],
                              'start_passed_to_manager': calls}, default=str))
            assert dataset.get('value', 'resampled') == 'resampled', 'late poll displays demo while Start uses resampled'

            # The late answer writes no piece drawn from the selection, and
            # hands nothing over: it read demo and the record of demo's render.
            assert chart.get('run') is None and 'interactive' not in start
            handoff = _written(poll, answer, by_id[tc.CHANGED_ID])
            assert handoff.get('__type__') == 'update', handoff
            # What the browser shows once every answer is applied, in the
            # order they came: one projection of resampled.
            shown = _shown([_answered(browser, choose, first),
                            _answered(browser, render_after_choice, drawn_demo),
                            _answered(browser, choose, second),
                            _answered(browser, render_after_choice, drawn_resampled),
                            (poll, answer)])
            resampled = browser.state(ws)
            assert shown['shep-train-dataset']['value'] == 'resampled'
            assert shown['shep-train-charts']['value']['run'].startswith('resampled/')
            assert resampled['row']['dataset_fingerprint'][:12] in shown['shep-train-chip']['value']
            assert shown['shep-train-start']['interactive'] is True
            # Start's target is the selection shown.
            assert [c['data_dir'] for c in calls] == [str(root / 'resampled')]
            # The next tick, from the selection as it is: nothing to hand over.
            answer = browser.predict(poll, [None] * len(poll.inputs))
            assert _written(poll, answer, by_id[tc.CHANGED_ID]).get('__type__') == 'update'
        finally:
            job.release.set()
            browser.close()
