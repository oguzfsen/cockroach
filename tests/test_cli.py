import os
import subprocess
import sys

import pytest

from cockroach.cli import main, parser


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--minutes", "-1"],
        ["train", "--minutes", "nan"],
        ["train", "--minutes", "inf"],
        ["train", "--steps", "0"],
        ["train", "--minutes", "1", "--steps", "2"],
    ],
)
def test_invalid_training_durations_are_rejected(args):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(args)
    assert error.value.code == 2


def test_missing_checkpoint_reports_a_readable_error(capsys):
    assert main(["evaluate", "--model", "/nonexistent/cockroach-model.npz"]) == 1
    assert "Hata:" in capsys.readouterr().out


def test_training_does_not_require_pygame_or_display():
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['pygame'] = None; "
            "from cockroach.training import rollout; "
            "from cockroach.config import WorldConfig; "
            "assert rollout(WorldConfig(day_seconds=1.6, episode_days=1), "
            "0, None)['days'] == 1",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
