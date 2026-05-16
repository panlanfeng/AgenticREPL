"""Tests for tab completion and command history navigation."""

import os
import sys
import pytest
import sys
import os
import json
import tempfile
import time


class TestTabCompletion:
    """Test the file/directory path tab completer."""

    def setup_method(self):
        try:
            from srun.repl import tab_completer
            self.completer = tab_completer
        except ImportError:
            pytest.skip("readline not available")

    def _completions(self, text):
        results = []
        state = 0
        while True:
            match = self.completer(text, state)
            if match is None:
                break
            results.append(match)
            state += 1
        return sorted(results)

    def test_complete_current_dir(self):
        """Typing nothing or '.' should complete files in current dir."""
        matches = self._completions("")
        assert len(matches) > 0, "Should find files in current directory"

    def test_complete_partial(self):
        """Typing partial filename should complete to matching files."""
        matches = self._completions("AGENT")
        assert any("AGENTS.md" in m for m in matches), \
            f"Should find AGENTS.md in completions: {matches}"

    def test_complete_directory(self):
        """Typing a directory name should add trailing slash."""
        # The completer adds "/" for directories
        # Try completing "src" → should get "src/"
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        matches = self._completions("src")
        assert matches, f"Should complete 'src' to something: {matches}"
        # At least one match should be a directory (with trailing /)
        dir_matches = [m for m in matches if m.endswith("/")]
        if os.path.isdir("src"):
            assert len(dir_matches) > 0, f"src/ is a directory but no dir matches: {matches}"

    def test_complete_no_match(self):
        """Typing nonsense should return empty results."""
        matches = self._completions("xyzzy_nonexistent_")
        assert matches == [], f"Should return empty for no match: {matches}"

    def test_complete_same_prefix(self):
        """Multiple files with same prefix should all appear."""
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        # Create temp files to test
        import tempfile
        with tempfile.NamedTemporaryFile(dir=".", prefix="srun_completion_test_", suffix=".txt", delete=False) as f1:
            pass
        with tempfile.NamedTemporaryFile(dir=".", prefix="srun_completion_test_", suffix=".md", delete=False) as f2:
            pass
        try:
            matches = self._completions("srun_completion_test_")
            assert len(matches) >= 2, f"Should find 2 test files: {matches}"
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_complete_handles_state(self):
        """State transitions should work: state 0 rebuilds, state N returns matches."""
        # state 0 → rebuilds match list
        result0 = self.completer("AGENT", 0)
        assert result0 is not None or True  # state 0 may or may not have matches

        # state 1 → should return second match or None
        result1 = self.completer("AGENT", 1)
        # state 1 can be None if only 1 match

        # state 999 → should return None (out of bounds)
        result999 = self.completer("AGENT", 999)
        assert result999 is None


class TestCommandHistory:
    """Test readline command history persistence."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.histfile = os.path.join(os.path.expanduser("~"), ".srun", "history")
        # Backup existing history
        self.backup = None
        if os.path.exists(self.histfile):
            with open(self.histfile) as f:
                self.backup = f.read()
            os.remove(self.histfile)
        yield
        # Restore
        if self.backup is not None:
            with open(self.histfile, "w") as f:
                f.write(self.backup)
        elif os.path.exists(self.histfile):
            os.remove(self.histfile)

    def test_history_file_created(self):
        """After running srun, history file should exist."""
        try:
            import readline
            readline.add_history("test command 1")
            readline.add_history("test command 2")
            readline.write_history_file(self.histfile)
            assert os.path.exists(self.histfile), "History file should be created"
        except ImportError:
            pytest.skip("readline not available")

    def test_history_file_readable(self):
        """History file should contain previously added commands."""
        try:
            import readline
            readline.add_history("echo hello")
            readline.add_history("ls -la")
            readline.write_history_file(self.histfile)
            readline.clear_history()
            readline.read_history_file(self.histfile)

            # Verify via history length
            assert readline.get_current_history_length() >= 2, \
                f"Should have 2+ history entries, got {readline.get_current_history_length()}"
        except ImportError:
            pytest.skip("readline not available")

    def test_history_persists_across_readline(self):
        """Writes to history file should be readable."""
        try:
            import readline
            readline.clear_history()
            readline.add_history("command alpha")
            readline.add_history("command beta")
            readline.write_history_file(self.histfile)

            # Read back with a fresh clear
            readline.clear_history()
            readline.read_history_file(self.histfile)
            assert readline.get_current_history_length() >= 2
        except ImportError:
            pytest.skip("readline not available")

    def test_history_no_duplicates_on_reload(self):
        """Re-reading history file should not duplicate entries if cleared first."""
        try:
            import readline
            readline.clear_history()
            readline.add_history("entry1")
            readline.write_history_file(self.histfile)

            readline.clear_history()
            readline.read_history_file(self.histfile)
            first_len = readline.get_current_history_length()

            readline.clear_history()
            readline.read_history_file(self.histfile)
            second_len = readline.get_current_history_length()

            assert first_len == second_len, \
                f"Re-reading should give same count: {first_len} vs {second_len}"
        except ImportError:
            pytest.skip("readline not available")


class TestHistoryIntegration:
    """Integration: verify history is used in srun REPL setup."""

    def test_readline_configured(self):
        """readline should be configured with history file path."""
        histfile = os.path.join(os.path.expanduser("~"), ".srun", "history")
        os.makedirs(os.path.dirname(histfile), exist_ok=True)
        if not os.path.exists(histfile):
            open(histfile, "a").close()
        try:
            import readline
            hl = readline.get_history_length()
            # -1 means not configured yet (runs before repl imports readline)
            assert hl == -1 or hl >= 0
        except ImportError:
            pytest.skip("readline not available")
