"""Tests file that ensures deepaas service loads and starts without problems.
## Note: Currently it is only possible to test deepaas service by running it
as subprocess. DEEPaaS does not provide a `test` call that generates a mock
client with loaded configuration.

Additionally, oslo_config performs parsing at command line call, it does not
support configuration loading. Therefore testing with direct calls to deepaas
would require monkeypatching configuration loading which might lead to multiple
issues, see:
https://github.com/pytest-dev/pytest/discussions/5461#discussioncomment-85530

As main inconvenient when using subprocess, the test duration is cannot be
optimized, leading to a minimum `watch_time` during the execution logs are
evaluated in order to ensure there are no printed logs showing errors during
the start up.

Shutdown of the process is done by killing the process group, at the end of
the tests module. However, interrupting the process group might lead to
leaking processes occupying ports. If tests fail due to address already in
use, you can run `lsof -i :{PORT}` and to check which process is occupying
the port. Then, you can kill the process with `kill -9 {PID}`.
"""
# pylint: disable=redefined-outer-name
import subprocess
from subprocess import TimeoutExpired

import pytest


@pytest.fixture(scope="module")
def deepaas_process(request, config_file):
    """Fixture to start deepaas process and kill it after tests."""
    with subprocess.Popen(
        args=["deepaas-run", "--config-file", config_file],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        text=True,  # Capture as text
    ) as process:
        try:
            outs, errs = process.communicate(timeout=request.param)
        except TimeoutExpired:
            process.kill()
            outs, errs = process.communicate()
        except Exception as exc:
            process.kill()
            raise exc
    return {"stdout": outs, "stderr": errs}


@pytest.mark.parametrize("deepaas_process", [60], indirect=True)
def test_stdout_errors(deepaas_process):
    """Assert there are no errors in process stdout."""
    assert "ERROR" not in deepaas_process["stdout"]


@pytest.mark.parametrize("deepaas_process", [60], indirect=True)
def test_stderr_errors(deepaas_process):
    """Assert there are no errors in process stderr."""
    assert "ERROR" not in deepaas_process["stderr"]
