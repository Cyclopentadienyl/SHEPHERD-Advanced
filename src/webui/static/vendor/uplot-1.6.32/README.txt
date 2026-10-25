# uPlot 1.6.32 (vendored)

The dashboard's charts (`src/webui/components/charts.py`, `src/webui/static/js/shep_charts.js`)
draw with uPlot by Leon Sorokin, <https://github.com/leeoniya/uPlot>, under the MIT licence
(`LICENSE`, copied from the package). Served by the API at `/static/vendor/uplot-1.6.32/`;
nothing is loaded from a CDN.

**Version:** 1.6.32, the latest stable release (npm `dist-tags.latest`, published
2025-03-14T02:51:47Z). The repository's master is an unreleased rewrite and is not used.

**Source:** the npm tarball <https://registry.npmjs.org/uplot/-/uplot-1.6.32.tgz>, checked
against the registry's values before copying:

- integrity `sha512-KIMVnG68zvu5XXUbC4LQEPnhwOxBuLyW1AHtpm6IKTXImkbLgkMy+jabjLgSLMasNuGGzQm/ep3tOkyTxpiQIw==`
- shasum (SHA-1) `c800a63b432bad692d6d746f44f0882aa73a49ae`

**Files, byte-identical to the tarball** (`package/dist/…` and `package/LICENSE`):

| File | SHA-256 |
|---|---|
| `uPlot.iife.min.js` | `19c8d4c6ad88929a79f4ae49d6f7161566dfd0ba3d15cc495e974f787eb78f1f` |
| `uPlot.min.css` | `df630c6a8d6f8eeaff264b50f73ce5b114f646ffd9a0bb74f049b0a00135fa04` |
| `LICENSE` | `8f989229699b4fe2f1a0432d0e9edc338a8a911e250e2d1b01ecd770a5f5b1bd` |

`tests/unit/test_webui_charts.py` checks these digests and that each file is served.

**Never edited.** Two 1.6.32 defects are worked around in our own code instead:

- log axes get their own tick splits (`logSplits` in `shep_charts.js`): 1.6.32's
  `logAxisSplits` loops until it throws or hangs for a minimum that is not a power of ten
  (upstream issue #1098, fixed only after 1.6.32);
- `shep_charts_locale.js`, loaded before uPlot, gives a browser whose `navigator.language`
  `Intl` refuses (`en-US@posix`) the locale `en-US`: 1.6.32 calls
  `new Intl.NumberFormat(navigator.language)` as it loads and never loads otherwise
  (upstream #1053).

When a later release has been out for two weeks, check whether both workarounds and the
disabled mousedown binding (`cursor.bind`, a 1.6.32 listener leak on destroy) can go, and
re-run the browser checks before replacing these files.
