"""Tests for CLI module."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mankinds_eval.cli import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestVersionCommand:
    """Tests for version command."""

    def test_version_shows_version(self) -> None:
        """Test that version command shows version."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "mankinds-eval" in result.output

    def test_version_includes_version_number(self) -> None:
        """Test that version output includes version number."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        # Version should contain a number (e.g., 0.1.0)
        # The output should have format like "mankinds-eval X.Y.Z"
        lines = result.output.strip().split("\n")
        assert len(lines) >= 1
        version_line = lines[0]
        assert "." in version_line  # Version numbers have dots


class TestInitCommand:
    """Tests for init command."""

    def test_init_shows_default_models(self) -> None:
        """Test that init command shows default models."""
        # Use --no-input mode by sending 'n' to skip customization
        result = runner.invoke(app, ["init"], input="n\n")

        assert result.exit_code == 0
        assert "Configuration Setup" in result.output
        assert "default models" in result.output.lower()

    def test_init_displays_model_types(self) -> None:
        """Test that init displays available model types."""
        result = runner.invoke(app, ["init"], input="n\n")

        assert result.exit_code == 0
        assert "embeddings" in result.output.lower()


class TestRunCommand:
    """Tests for run command."""

    def test_run_missing_config(self) -> None:
        """Test that run command errors when config is missing."""
        result = runner.invoke(
            app,
            ["run", "-c", "/nonexistent/config.yaml", "-d", "/some/data.json"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_run_missing_data(self, tmp_path: Path) -> None:
        """Test that run command errors when data file is missing."""
        # Create a valid config file
        config_file = tmp_path / "scorer.yaml"
        config_file.write_text("""
name: test_scorer
methods:
  - type: heuristic.ExactMatch
""")

        result = runner.invoke(
            app,
            ["run", "-c", str(config_file), "-d", "/nonexistent/data.json"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_run_requires_config_option(self) -> None:
        """Test that run command requires config option."""
        result = runner.invoke(app, ["run", "-d", "data.json"])

        # Should error because --config is required
        assert result.exit_code != 0

    def test_run_requires_data_option(self) -> None:
        """Test that run command requires data option."""
        result = runner.invoke(app, ["run", "-c", "config.yaml"])

        # Should error because --data is required
        assert result.exit_code != 0

    def _make_run_fixtures(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create minimal config and data files for run tests."""
        config_file = tmp_path / "scorer.yaml"
        config_file.write_text("name: test_scorer\nmethods:\n  - type: heuristic.ExactMatch\n")
        data_file = tmp_path / "data.json"
        data_file.write_text('[{"input": "Q", "output": "A", "expected": "A"}]')
        return config_file, data_file

    def _assert_ran(self, result: object) -> None:
        """Assert the run command actually executed (not just showed help)."""
        output = _strip_ansi(getattr(result, "output", ""))
        assert getattr(result, "exit_code", -1) == 0, (
            f"exit_code={getattr(result, 'exit_code', '?')}, output={output[:500]}"
        )
        assert "Usage:" not in output, f"Command showed help instead of running: {output[:500]}"

    def test_run_successful_evaluation(self, tmp_path: Path) -> None:
        """Test successful run command execution."""
        config_file, data_file = self._make_run_fixtures(tmp_path)

        result = runner.invoke(
            app,
            ["run", "-c", str(config_file), "-d", str(data_file)],
        )

        self._assert_ran(result)
        assert "Evaluation complete" in result.output
        assert "Samples:" in result.output

    def test_run_with_verbose_flag(self, tmp_path: Path) -> None:
        """Test run command with verbose flag."""
        config_file, data_file = self._make_run_fixtures(tmp_path)

        result = runner.invoke(
            app,
            ["run", "-c", str(config_file), "-d", str(data_file), "-v"],
        )

        self._assert_ran(result)
        assert "Loading" in result.output

    def test_run_with_output_file(self, tmp_path: Path) -> None:
        """Test run command with output file option."""
        config_file, data_file = self._make_run_fixtures(tmp_path)
        output_file = tmp_path / "results.json"

        result = runner.invoke(
            app,
            [
                "run",
                "-c",
                str(config_file),
                "-d",
                str(data_file),
                "-o",
                str(output_file),
            ],
        )

        self._assert_ran(result)
        assert output_file.exists()
        assert "Results written to" in result.output

    def test_run_with_html_output(self, tmp_path: Path) -> None:
        """Test run command with HTML output option."""
        config_file, data_file = self._make_run_fixtures(tmp_path)
        html_file = tmp_path / "scorecard.html"

        result = runner.invoke(
            app,
            [
                "run",
                "-c",
                str(config_file),
                "-d",
                str(data_file),
                "--html",
                str(html_file),
            ],
        )

        self._assert_ran(result)
        assert html_file.exists()
        assert "Scorecard written to" in result.output


class TestConfigShowCommand:
    """Tests for config show command."""

    def test_config_show_runs(self) -> None:
        """Test that config show command runs without error."""
        result = runner.invoke(app, ["config", "show"])

        # Should succeed regardless of whether config exists
        assert result.exit_code == 0

    def test_config_show_displays_info(self) -> None:
        """Test that config show displays configuration info."""
        result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should show either config file path or default models
        output_lower = result.output.lower()
        assert "config" in output_lower or "model" in output_lower

    def test_config_show_no_config_file(self) -> None:
        """Test config show when no config file exists."""
        with patch("mankinds_eval.cli.commands.config.get_config_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/.mankinds_eval/config.yaml")
            with patch("mankinds_eval.cli.commands.config.load_config") as mock_load:
                mock_load.return_value = {}

                result = runner.invoke(app, ["config", "show"])

                assert result.exit_code == 0
                assert "default" in result.output.lower()


class TestConfigGetCommand:
    """Tests for config get command."""

    def test_config_get_missing_key(self) -> None:
        """Test config get with non-existent key."""
        result = runner.invoke(app, ["config", "get", "nonexistent.key"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestConfigSetCommand:
    """Tests for config set command."""

    def test_config_set_basic(self, tmp_path: Path) -> None:
        """Test config set command."""
        config_path = tmp_path / ".mankinds_eval" / "config.yaml"

        with patch("mankinds_eval.cli.commands.config.get_config_path") as mock_path:
            mock_path.return_value = config_path
            with patch("mankinds_eval.cli.commands.config.load_config") as mock_load:
                mock_load.return_value = {}
                with patch("mankinds_eval.cli.commands.config.save_config") as mock_save:
                    result = runner.invoke(
                        app,
                        ["config", "set", "models.embeddings", "custom-model"],
                    )

                    assert result.exit_code == 0
                    assert "custom-model" in result.output
                    mock_save.assert_called_once()


class TestHelpOutput:
    """Tests for CLI help output."""

    def test_main_help(self) -> None:
        """Test main help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        clean = _strip_ansi(result.output).lower()
        assert "mankinds-eval" in clean
        assert "evaluation" in clean

    def test_run_help(self) -> None:
        """Test run command help output."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "--config" in clean
        assert "--data" in clean

    def test_config_help(self) -> None:
        """Test config subcommand help output."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        clean = _strip_ansi(result.output).lower()
        assert "show" in clean
        assert "get" in clean
        assert "set" in clean
