"""Tests of the control panel's logic (no ROS, no window needed):
    python3 -m pytest launcher/test_panel.py"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocohrip_panel as panel  # noqa: E402

FAKE = textwrap.dedent("""
    import signal, sys, time
    out = [a for a in sys.argv[1:] if a.startswith("/")][-1]
    def bye(*_):
        open(out, "w").write("SIGINT")
        sys.exit(0)
    signal.signal(signal.SIGINT, bye)
    open(out + ".ready", "w").write("")
    while True:
        time.sleep(0.05)
""")


def wait_ready(flag, timeout=5.0):
    """The fake launch has set up its Ctrl+C handler."""
    t = time.time()
    while not os.path.exists(str(flag) + ".ready") and time.time() - t < timeout:
        time.sleep(0.05)


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """A fake workspace + ROS setup file and a fake 'launch' script."""
    ws = tmp_path / "ws"
    (ws / "install").mkdir(parents=True)
    (ws / "install" / "setup.bash").write_text("export AMENT_PREFIX_PATH=%s/install/pkg\n" % ws)
    ros = tmp_path / "ros_setup.bash"
    ros.write_text("")
    script = tmp_path / "fake_launch.py"
    script.write_text(FAKE)
    monkeypatch.setattr(panel, "LOG_DIR", str(tmp_path / "logs"))
    return ws, ros, script, tmp_path


def test_signature_ignores_changing_arguments_and_matches_ros2_launch():
    cfg = panel.load_config("/nonexistent.json")
    sig = panel.signature(cfg["games"]["chess"]["game"])
    assert sig == ["ros2", "launch", "chess_game", "chess_game_launch.py"]
    argv = ["/usr/bin/python3", "/opt/ros/jazzy/bin/ros2", "launch", "chess_game",
            "chess_game_launch.py", "engine:=maia", "level:=2"]
    assert panel.matches(argv, sig)
    assert not panel.matches(argv, panel.signature(cfg["robot"]))
    assert not panel.matches(["bash", "-c", "exec ros2 launch chess_game chess_game_launch.py"], sig)


def test_part_starts_logs_and_stops_like_ctrl_c(cell):
    ws, ros, script, tmp = cell
    flag = tmp / "stopped"
    part = panel.Part("robot", "Robot software", f"{sys.executable} {script} robot {flag}",
                      str(ws), str(ros), log_name="t_robot")
    part.start()
    wait_ready(flag)
    assert part.running() and not part.external()
    pid = part.pids()[0]
    assert panel.runs_from(pid, str(ws)) is True
    assert panel.runs_from(pid, str(tmp / "other_ws")) is False
    assert part.stop(timeout=5)
    assert flag.read_text() == "SIGINT"            # stopped gently, as with Ctrl+C
    assert not part.running()
    assert open(part.log_path).read().startswith("$ ")


def test_part_sees_and_stops_a_process_started_elsewhere(cell):
    ws, ros, script, tmp = cell
    flag = tmp / "stopped2"
    cmd = f"{sys.executable} {script} control {flag}"
    other = subprocess.Popen(cmd.split(), start_new_session=True)
    try:
        part = panel.Part("control", "Robot control", cmd, str(ws), str(ros))
        wait_ready(flag)
        assert part.external()
        assert part.stop(timeout=5)
        other.wait(timeout=5)
        assert flag.read_text() == "SIGINT"
    finally:
        if other.poll() is None:
            other.kill()


def test_missing_workspace_is_reported(cell):
    _, ros, script, tmp = cell
    part = panel.Part("game", "Game", f"{sys.executable} {script}", str(tmp / "nope"), str(ros))
    with pytest.raises(RuntimeError, match="colcon build"):
        part.start()


def test_stop_everything_finds_all_parts(cell):
    ws, ros, script, tmp = cell
    cfg = panel.load_config("/nonexistent.json")
    cfg["robot"] = f"{sys.executable} {script} fake-robot {tmp / 'r'}"
    cfg["control"] = f"{sys.executable} {script} fake-control {tmp / 'c'}"
    cfg["games"] = {"chess": {"title": "Chess", "workspace": str(ws),
                              "game": f"{sys.executable} {script} fake-game {tmp / 'g'} engine:={{engine}}"}}
    procs = [subprocess.Popen(c.replace("{engine}", "maia").split(), start_new_session=True)
             for c in (cfg["robot"], cfg["control"], cfg["games"]["chess"]["game"])]
    try:
        for name in "rcg":
            wait_ready(tmp / name)
        assert panel.stop_everything(cfg, log=lambda *a: None) == [
            "Chess game", "Robot control", "Robot software"]
        for p in procs:
            p.wait(timeout=5)
        assert panel.stop_everything(cfg, log=lambda *a: None) == []
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()


def test_camera_usb_speed_from_sysfs(tmp_path):
    for name, vendor, speed in (("1-1", "8087", "480"), ("2-3", "20f7", "5000")):
        d = tmp_path / name
        d.mkdir()
        (d / "idVendor").write_text(vendor + "\n")
        (d / "speed").write_text(speed + "\n")
    assert panel.camera_usb_speed(str(tmp_path)) == 5000
    (tmp_path / "2-3" / "speed").write_text("480\n")
    assert panel.camera_usb_speed(str(tmp_path)) == 480
    assert panel.camera_usb_speed(str(tmp_path / "missing")) is None


def test_config_file_changes_only_what_it_names(tmp_path):
    f = tmp_path / "launcher.json"
    f.write_text('{"robot_ip": "10.0.0.9", "games": {"checkers": {"workspace": "~/cw"}}}')
    cfg = panel.load_config(str(f))
    assert cfg["robot_ip"] == "10.0.0.9"
    assert cfg["games"]["checkers"]["workspace"] == "~/cw"
    assert cfg["games"]["checkers"]["title"] == "Checkers"
    assert cfg["games"]["chess"] == panel.DEFAULTS["games"]["chess"]
