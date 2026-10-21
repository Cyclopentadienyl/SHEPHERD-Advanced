/*
 * Loaded before uPlot (the chart component's head=, in order): uPlot 1.6.32
 * runs `new Intl.NumberFormat(navigator.language)` as it loads, which throws
 * for a locale Intl refuses (en-US@posix, reported by some Linux browsers),
 * and uPlot then never loads (upstream #1053, fixed after 1.6.32). The
 * vendored file is never edited; this guard gives such a browser "en-US".
 */
try { new Intl.NumberFormat(navigator.language); } catch (e) { Object.defineProperty(navigator, "language", { value: "en-US", configurable: true }); }
