"""
# ==============================================================================
# Module: tests/unit/test_pyg_native_check.py
# ==============================================================================
# Purpose: Unit tests for src/utils/pyg_native_check.py
#
# Tests:
#   - Probing returns a status for every tracked extension
#   - Result caching (and force re-probe)
#   - format_status_line / get_missing_extensions
#   - get_fallback_warning: None when all present, message when any missing
#   - log_pyg_native_status log level (INFO vs WARNING)
#
# Note: torch-free by design — these run anywhere, including environments
#       without the native extensions installed.
# ==============================================================================
"""
import logging

from src.utils.pyg_native_check import (
    ExtensionStatus,
    check_pyg_native_extensions,
    format_status_line,
    get_fallback_warning,
    get_missing_extensions,
    log_pyg_native_status,
)

_EXTENSIONS = ("pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster")


def _all_present():
    """A synthetic 'everything installed' status map."""
    return {name: ExtensionStatus(name, True, "9.9.9", None) for name in _EXTENSIONS}


def _some_missing():
    """pyg_lib missing, the rest present."""
    statuses = _all_present()
    statuses["pyg_lib"] = ExtensionStatus("pyg_lib", False, None, "No module named 'pyg_lib'")
    return statuses


class TestProbe:
    def test_returns_status_for_every_extension(self):
        statuses = check_pyg_native_extensions(force=True)
        assert set(statuses.keys()) == set(_EXTENSIONS)
        for st in statuses.values():
            assert isinstance(st, ExtensionStatus)
            assert isinstance(st.available, bool)

    def test_result_is_cached(self):
        first = check_pyg_native_extensions(force=True)
        second = check_pyg_native_extensions()
        assert second is first  # cached object reused

    def test_force_reprobes(self):
        first = check_pyg_native_extensions(force=True)
        second = check_pyg_native_extensions(force=True)
        assert second is not first  # force builds a fresh dict
        assert set(second.keys()) == set(_EXTENSIONS)


class TestSummaries:
    def test_format_status_line_marks_present_and_missing(self):
        line = format_status_line(_some_missing())
        assert "pyg_lib MISSING" in line
        assert "torch_scatter 9.9.9 OK" in line

    def test_get_missing_extensions(self):
        assert get_missing_extensions(_all_present()) == []
        assert get_missing_extensions(_some_missing()) == ["pyg_lib"]


class TestFallbackWarning:
    def test_none_when_all_present(self):
        assert get_fallback_warning(_all_present()) is None

    def test_message_when_missing(self):
        msg = get_fallback_warning(_some_missing())
        assert msg is not None
        assert "pyg_lib" in msg
        # Should be reassuring (still runs) but actionable (how to diagnose).
        assert "still run" in msg
        assert "validate_pyg_ext.py" in msg


class TestLogging:
    def test_info_level_when_all_present(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.utils.pyg_native_check"):
            log_pyg_native_status(_all_present())
        records = [r for r in caplog.records if r.name == "src.utils.pyg_native_check"]
        assert records and all(r.levelno == logging.INFO for r in records)

    def test_warning_level_when_missing(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.utils.pyg_native_check"):
            log_pyg_native_status(_some_missing())
        records = [r for r in caplog.records if r.name == "src.utils.pyg_native_check"]
        assert records and records[-1].levelno == logging.WARNING
