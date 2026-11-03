#!/usr/bin/env python3
"""Control panel for the robot cell: one window per game that starts the
robot software, robot_control and the game in the right order, shows what
the robot still needs (power, brakes, the pendant program, a stop), and
stops everything again.

    cocohrip_panel.py chess | checkers     the desktop icons
    cocohrip_panel.py --stop-all            the "Stop robot apps" icon

Start it through the wrapper `cocohrip-panel`, which gives it a clean ROS
environment (ROS + the game's workspace only).  Each part runs in its own
process group with its output in ~/.cocohrip/logs/<game>_<part>.log, and
keeps running if the panel is closed (the panel asks).  Parts started by
hand in a terminal are recognised too, and stopped like Ctrl+C would.

Settings (workspaces, commands, robot IP) can be changed in
~/.cocohrip/launcher.json - same keys as DEFAULTS below, only the ones to
change.
"""

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

HOME_DIR = os.path.expanduser(os.environ.get("COCOHRIP_HOME", "~/.cocohrip"))
CONFIG_FILE = os.path.join(HOME_DIR, "launcher.json")
CHOICES_FILE = os.path.join(HOME_DIR, "launcher_choices.json")
LOG_DIR = os.path.join(HOME_DIR, "logs")

DEFAULTS = {
    "robot_ip": "192.168.0.5",
    "ros_setup": "",                     # "" = /opt/ros/jazzy/setup.bash
    "autostart": True,                   # start everything when the panel opens
    "robot": "ros2 launch cocohrip_control bringup_all.launch.py",
    "robot_nodes": ["controller_manager", "move_group"],
    "control": "ros2 launch robot_control new_launch.py",
    "control_nodes": ["robot_control_node"],
    "games": {
        "chess": {
            "title": "Chess", "workspace": "~/chess_ws",
            "game": "ros2 launch chess_game chess_game_launch.py engine:={engine} level:={level}",
            "game_nodes": ["chess_node"], "engine": "stockfish", "level": 3,
        },
        "checkers": {
            "title": "Checkers", "workspace": "~/checkers_ws",
            "game": "ros2 launch checkers_game checkers_game_launch.launch.py",
            "game_nodes": ["checkers_node"],
        },
    },
}

STATUS_NS = "/io_and_status_controller/"      # UR driver: robot mode, safety, program
DASHBOARD = "/dashboard_client/"              # UR driver: pendant buttons over the network

# ur_dashboard_msgs RobotMode / SafetyMode values
POWER_OFF, POWER_ON, IDLE, RUNNING = 3, 4, 5, 7
SAFETY_NORMAL, SAFETY_REDUCED, PROTECTIVE_STOP, SAFEGUARD_STOP = 1, 2, 3, 5
EMERGENCY_STOPS = (6, 7)


# ======================================================================
# configuration
# ======================================================================
def _merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def load_config(path=None):
    cfg = json.loads(json.dumps(DEFAULTS))
    path = path or CONFIG_FILE
    try:
        with open(path) as f:
            _merge(cfg, json.load(f))
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        print(f"[panel] ignoring {path}: {e}")
    return cfg


def load_choices():
    try:
        with open(CHOICES_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_choices(choices):
    try:
        os.makedirs(HOME_DIR, exist_ok=True)
        with open(CHOICES_FILE, "w") as f:
            json.dump(choices, f)
    except OSError:
        pass


def ros_setup_file(cfg):
    if cfg.get("ros_setup"):
        return os.path.expanduser(cfg["ros_setup"])
    for distro in (os.environ.get("ROS_DISTRO"), "jazzy", "humble"):
        if distro and os.path.exists(f"/opt/ros/{distro}/setup.bash"):
            return f"/opt/ros/{distro}/setup.bash"
    return "/opt/ros/jazzy/setup.bash"


# ======================================================================
# processes
# ======================================================================
def signature(command):
    """The words that identify `command` among the running processes,
    without arguments that change (engine:=..., {level})."""
    words = [w for w in shlex.split(command) if ":=" not in w and "{" not in w]
    return words[:4]


def matches(argv, sig):
    """argv contains sig (its first word compared by file name, so that
    "ros2" matches "/opt/ros/jazzy/bin/ros2")."""
    if not sig:
        return False
    first = os.path.basename(sig[0])
    for i, a in enumerate(argv):
        if os.path.basename(a) == first and argv[i + 1:i + len(sig)] == sig[1:]:
            return True
    return False


def _cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [a.decode(errors="replace") for a in f.read().split(b"\0")[:-1]]
    except OSError:
        return []


def process_table():
    """{pid: argv} of all processes but this one."""
    me = os.getpid()
    return {int(n): _cmdline(int(n)) for n in os.listdir("/proc") if n.isdigit() and int(n) != me}


def find_processes(sig, table=None):
    table = process_table() if table is None else table
    return sorted(pid for pid, argv in table.items() if matches(argv, sig))


def process_prefix(pid):
    """AMENT_PREFIX_PATH of a process (which workspace it runs from)."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for entry in f.read().split(b"\0"):
                if entry.startswith(b"AMENT_PREFIX_PATH="):
                    return entry.split(b"=", 1)[1].decode(errors="replace")
    except OSError:
        pass
    return None


def runs_from(pid, workspace):
    prefix = process_prefix(pid)
    if prefix is None:
        return None                        # cannot tell
    return os.path.join(os.path.expanduser(workspace), "install") in prefix


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def stop_pids(pids, timeout=20.0, own=None):
    """Stop processes like Ctrl+C in their terminal: SIGINT to their
    process group (the launch and its nodes), then SIGTERM, then SIGKILL.
    `own` = Popen started by us (reaped here).  True if all are gone."""
    pids = [p for p in pids if alive(p)]
    my_group = os.getpgid(0)
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        for pid in pids:
            try:
                group = os.getpgid(pid)
                if group == pid and group != my_group:
                    os.killpg(group, sig)
                else:
                    os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        t_end = time.time() + wait
        while time.time() < t_end:
            if own is not None:
                own.poll()
            if not any(alive(p) for p in pids):
                return True
            time.sleep(0.2)
    return not any(alive(p) for p in pids)


class Part:
    """One thing the panel starts: a command run in a workspace."""

    def __init__(self, key, title, command, workspace, ros_setup, nodes=(), note="", log_name=None):
        self.key, self.title, self.command, self.note = key, title, command, note
        self.workspace = os.path.expanduser(workspace)
        self.ros_setup = ros_setup
        self.sig = signature(command)
        self.nodes = list(nodes)
        self.proc = None
        self.log_path = os.path.join(LOG_DIR, f"{log_name or key}.log")
        self.started_at = 0.0

    def own(self):
        return self.proc is not None and self.proc.poll() is None

    def pids(self):
        own = [self.proc.pid] if self.own() else []
        return own + [p for p in find_processes(self.sig) if p not in own]

    def running(self):
        return bool(self.pids())

    def external(self):
        """Running, but not started by this panel (a terminal, another panel)."""
        return not self.own() and bool(find_processes(self.sig))

    def start(self, command=None):
        cmd = command or self.command
        setup = os.path.join(self.workspace, "install", "setup.bash")
        if not os.path.exists(setup):
            raise RuntimeError(f"{setup} is missing - build the workspace first (colcon build)")
        os.makedirs(LOG_DIR, exist_ok=True)
        if os.path.exists(self.log_path):
            os.replace(self.log_path, self.log_path + ".1")      # the previous run
        script = (f"source {shlex.quote(self.ros_setup)} && source {shlex.quote(setup)} "
                  f"&& exec {cmd}")
        with open(self.log_path, "w") as log:
            log.write(f"$ {cmd}\n  in {self.workspace}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            log.flush()
            self.proc = subprocess.Popen(["bash", "-c", script], cwd=self.workspace,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        self.started_at = time.time()

    def stop(self, timeout=20.0):
        return stop_pids(self.pids(), timeout, own=self.proc)


def stop_everything(cfg, log=print):
    """Stop every game, robot_control and the robot software, from any
    workspace (the "Stop robot apps" icon).  Returns the titles stopped."""
    order = [(g["title"] + " game", g["game"]) for g in cfg["games"].values()]
    order += [("Robot control", cfg["control"]), ("Robot software", cfg["robot"])]
    stopped = []
    for title, command in order:
        pids = find_processes(signature(command))
        if pids:
            log(f"stopping {title} ...")
            stop_pids(pids)
            stopped.append(title)
    return stopped


# ======================================================================
# robot cell checks
# ======================================================================
XIMEA_VENDOR = "20f7"


def camera_usb_speed(root="/sys/bus/usb/devices"):
    """USB speed (Mbit/s) of the Ximea camera, None if it is not plugged in."""
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return None
    for d in names:
        try:
            with open(os.path.join(root, d, "idVendor")) as f:
                if f.read().strip() != XIMEA_VENDOR:
                    continue
            with open(os.path.join(root, d, "speed")) as f:
                return int(float(f.read().strip()))
        except (OSError, ValueError):
            continue
    return None


def usbfs_memory_mb(path="/sys/module/usbcore/parameters/usbfs_memory_mb"):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def ping(ip):
    try:
        return subprocess.run(["ping", "-c", "1", "-W", "1", ip], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=3).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return None


class RosMonitor:
    """The robot's state (UR driver topics) and the running ROS nodes,
    updated by a background thread.  Works without ROS too (ok=False)."""

    def __init__(self):
        self.ok = False
        self.error = None
        self.nodes = set()
        self.robot_mode = None
        self.safety_mode = None
        self.program_running = None
        self.has_modes = False
        self._stop = False
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from std_msgs.msg import Bool
        except ImportError as e:
            self.error = f"ROS 2 Python not found ({e})"
            return
        try:
            from ur_dashboard_msgs.msg import RobotMode, SafetyMode
            self.has_modes = True
        except ImportError:
            RobotMode = SafetyMode = None
        self.rclpy = rclpy
        try:                                    # keep the normal Ctrl+C / SIGTERM behaviour
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        except (ImportError, TypeError):
            rclpy.init(args=[])
        self.node = rclpy.create_node("cocohrip_panel")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        sub = self.node.create_subscription
        sub(Bool, STATUS_NS + "robot_program_running",
            lambda m: setattr(self, "program_running", bool(m.data)), latched)
        if self.has_modes:
            sub(RobotMode, STATUS_NS + "robot_mode",
                lambda m: setattr(self, "robot_mode", int(m.mode)), latched)
            sub(SafetyMode, STATUS_NS + "safety_mode",
                lambda m: setattr(self, "safety_mode", int(m.mode)), latched)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.ok = True
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        last = 0.0
        while not self._stop and self.rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
                if time.time() - last > 1.0:
                    last = time.time()
                    self.nodes = {n for n, _ in self.node.get_node_names_and_namespaces()}
                    if "io_and_status_controller" not in self.nodes:
                        # the driver is gone: its last (latched) values are stale
                        self.robot_mode = self.safety_mode = self.program_running = None
            except Exception:                   # noqa: BLE001 - shutting down
                time.sleep(0.2)

    def driver_up(self):
        return "io_and_status_controller" in self.nodes

    def trigger(self, service, timeout=6.0):
        """Call a std_srvs/Trigger service; (success, message)."""
        if not self.ok:
            return False, "ROS 2 not available"
        from std_srvs.srv import Trigger
        client = self.node.create_client(Trigger, service)
        try:
            if not client.wait_for_service(timeout_sec=2.0):
                return False, f"{service} is not available (robot software not running?)"
            future = client.call_async(Trigger.Request())
            t_end = time.time() + timeout
            while not future.done() and time.time() < t_end:
                time.sleep(0.05)
            if not future.done():
                return False, f"{service}: no answer"
            res = future.result()
            return bool(res.success), res.message.strip()
        finally:
            self.node.destroy_client(client)

    def shutdown(self):
        self._stop = True
        if self.ok:
            try:
                self.node.destroy_node()
                self.rclpy.try_shutdown()
            except Exception:                   # noqa: BLE001
                pass


class Checker:
    """Robot reachable, camera on USB 3, the parts' processes - every second."""

    def __init__(self, robot_ip, parts):
        self.robot_ip = robot_ip
        self.parts = parts
        self.reachable = None
        self.camera_speed = None
        self.usbfs = None
        self.pids = {p.key: [] for p in parts}
        self.external = {p.key: False for p in parts}
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last_ping = 0.0
        while not self._stop:
            table = process_table()
            for p in self.parts:
                found = find_processes(p.sig, table)
                own = [p.proc.pid] if p.own() else []
                self.pids[p.key] = own + [x for x in found if x not in own]
                self.external[p.key] = bool(found) and not own
            self.camera_speed = camera_usb_speed()
            self.usbfs = usbfs_memory_mb()
            if time.time() - last_ping > 3.0:
                last_ping = time.time()
                self.reachable = ping(self.robot_ip)
            time.sleep(1.0)

    def stop(self):
        self._stop = True


# ======================================================================
# the window
# ======================================================================
class Cancelled(Exception):
    pass


class Failed(Exception):
    pass


COL = {"ok": "#27ae60", "wait": "#e67e22", "bad": "#c0392b", "off": "#95a5a6",
       "info": "#2f6fb0", "action": "#d35400", "error": "#c0392b", "done": "#27ae60"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Panel:
    def __init__(self, root, cfg, game_key):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.cfg = cfg
        self.game_key = game_key
        self.game = cfg["games"][game_key]
        self.title = self.game["title"]
        ros = ros_setup_file(cfg)
        ws = self.game["workspace"]
        self.parts = [
            Part("robot", "Robot software", cfg["robot"], ws, ros, cfg["robot_nodes"],
                 "driver, MoveIt, RViz, gripper", f"{game_key}_robot"),
            Part("control", "Robot control", cfg["control"], ws, ros, cfg["control_nodes"],
                 "plans and executes the moves", f"{game_key}_control"),
            Part("game", f"{self.title} game", self.game["game"], ws, ros,
                 self.game.get("game_nodes", []), "camera, rules, the game window",
                 f"{game_key}_game"),
        ]
        self.by_key = {p.key: p for p in self.parts}
        self.ros = RosMonitor()
        self.checker = Checker(cfg["robot_ip"], self.parts)
        self.message = ("Opening ..." if cfg.get("autostart", True) else "Not started", "info",
                        "" if cfg.get("autostart", True) else
                        "▶ Start everything starts the robot software, robot control and the game.")
        self.worker = None
        self.cancel = threading.Event()
        self.skip_robot_wait = False
        self.waiting_since = None
        self.log_part = "game"
        self.game_was_ready = False
        self.choices = load_choices()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(300, self._tick)
        if cfg.get("autostart", True):
            root.after(1200, self.start_all)

    # ------------------------------------------------------------ layout
    def _build(self):
        tk, ttk, root = self.tk, self.ttk, self.root
        root.title(f"{self.title} - robot cell")
        root.minsize(720, 640)
        root.geometry("800x860")
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        from tkinter import font
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            font.nametofont(name).configure(size=11)
        style.configure("Big.TButton", font=("TkDefaultFont", 13, "bold"), padding=(14, 8))
        style.configure("Head.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Sec.TLabel", font=("TkDefaultFont", 11, "bold"), foreground="#555")
        style.configure("Dim.TLabel", foreground="#666")

        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        head = ttk.Frame(outer)
        head.pack(fill="x")
        ttk.Label(head, text=f"{self.title}", style="Head.TLabel").pack(side="left")
        ttk.Label(head, text="  robot cell control", style="Dim.TLabel").pack(side="left", pady=(6, 0))

        self.banner = tk.Label(outer, text="", anchor="w", justify="left", fg="white",
                               bg=COL["info"], font=("TkDefaultFont", 13, "bold"),
                               padx=12, pady=8, wraplength=680)
        self.banner.pack(fill="x", pady=(8, 2))
        self.hint = ttk.Label(outer, text="", style="Dim.TLabel", wraplength=680, justify="left")
        self.hint.pack(fill="x", pady=(0, 6))
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 6))
        self.start_btn = ttk.Button(bar, text="▶  Start everything", style="Big.TButton",
                                    command=self.start_all)
        self.start_btn.pack(side="left")
        ttk.Button(bar, text="■  Stop everything", style="Big.TButton",
                   command=self.stop_all).pack(side="left", padx=8)
        self.skip_btn = ttk.Button(bar, text="Continue without robot status",
                                   command=self._skip_wait)

        # robot checks
        ttk.Label(outer, text="ROBOT", style="Sec.TLabel").pack(anchor="w", pady=(6, 0))
        checks = ttk.Frame(outer)
        checks.pack(fill="x")
        self.check_rows = {}
        for key, text in (("reach", f"Robot switched on and reachable ({self.cfg['robot_ip']})"),
                          ("power", "Powered on, brakes released"),
                          ("safety", "No stop (emergency / protective)"),
                          ("program", "Pendant program running (External Control)"),
                          ("camera", "Camera on USB 3")):
            self.check_rows[key] = self._light_row(checks, text)
        self.fix_btn = ttk.Button(self.check_rows["camera"][2], text="Fix (asks for the password)",
                                  command=self._fix_usbfs)

        # software parts
        ttk.Label(outer, text="SOFTWARE", style="Sec.TLabel").pack(anchor="w", pady=(8, 0))
        sw = ttk.Frame(outer)
        sw.pack(fill="x")
        sw.columnconfigure(1, weight=1)
        self.part_rows = {}
        for i, p in enumerate(self.parts):
            dot = tk.Canvas(sw, width=16, height=16, highlightthickness=0)
            dot.grid(row=i, column=0, padx=(0, 6), pady=2)
            label = ttk.Label(sw, text=f"{p.title}  ", anchor="w")
            label.grid(row=i, column=1, sticky="w")
            state = ttk.Label(sw, text="", style="Dim.TLabel", width=26, anchor="w")
            state.grid(row=i, column=2, sticky="w")
            btn = ttk.Button(sw, text="Start", width=7, command=lambda k=p.key: self.toggle_part(k))
            btn.grid(row=i, column=3, padx=4)
            ttk.Button(sw, text="Log", width=5, command=lambda k=p.key: self._show_log(k)).grid(
                row=i, column=4)
            self.part_rows[p.key] = (dot, label, state, btn)

        # opponent (chess)
        self.engine_box = self.level_box = None
        if "engine" in self.game:
            self._build_opponent(outer)

        # robot buttons (UR dashboard)
        ttk.Label(outer, text="ROBOT BUTTONS  (work when the pendant is in Remote Control; "
                  "otherwise use the pendant)", style="Sec.TLabel").pack(anchor="w", pady=(10, 2))
        rb = ttk.Frame(outer)
        rb.pack(fill="x")
        for text, service, confirm in (("Power on", "power_on", False),
                                       ("Release brakes", "brake_release", True),
                                       ("Play program", "play", True),
                                       ("Stop program", "stop", False),
                                       ("Unlock protective stop", "unlock_protective_stop", True)):
            ttk.Button(rb, text=text, command=lambda s=service, c=confirm, t=text:
                       self.dashboard(s, t, c)).pack(side="left", padx=(0, 6))

        # log
        lh = ttk.Frame(outer)
        lh.pack(fill="x", pady=(10, 2))
        self.log_title = ttk.Label(lh, text="LOG", style="Sec.TLabel")
        self.log_title.pack(side="left")
        ttk.Button(lh, text="Open in a terminal", command=self._log_terminal).pack(side="right")
        self.log = tk.Text(outer, height=10, bg="#1e1e1e", fg="#dddddd", insertbackground="white",
                           font=("DejaVu Sans Mono", 9), wrap="word", relief="flat")
        self.log.pack(fill="both", expand=True)

    def _light_row(self, parent, text):
        tk, ttk = self.tk, self.ttk
        row = ttk.Frame(parent)
        row.pack(fill="x")
        dot = tk.Canvas(row, width=16, height=16, highlightthickness=0)
        dot.pack(side="left", padx=(0, 6), pady=1)
        ttk.Label(row, text=text).pack(side="left")
        detail = ttk.Label(row, text="", style="Dim.TLabel")
        detail.pack(side="left", padx=8)
        return dot, detail, row

    def _build_opponent(self, parent):
        ttk = self.ttk
        try:
            from chess_game.engines import APPROX_ELO, CHOICES, LEVELS, TITLES
        except ImportError:
            return
        self.engines = [n for n in CHOICES if n in APPROX_ELO]
        self.engine_titles = {n: TITLES.get(n, n) for n in self.engines}
        self.approx = APPROX_ELO
        self.levels = list(LEVELS)
        mine = self.choices.get(self.game_key, {})
        engine = mine.get("engine", self.game.get("engine", "stockfish"))
        level = int(mine.get("level", self.game.get("level", 3)))
        ttk.Label(parent, text="OPPONENT", style="Sec.TLabel").pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(parent)
        row.pack(fill="x")
        self.engine_box = ttk.Combobox(row, state="readonly", width=20,
                                       values=[self.engine_titles[n] for n in self.engines])
        self.engine_box.pack(side="left")
        self.level_box = ttk.Combobox(row, state="readonly", width=22)
        self.level_box.pack(side="left", padx=8)
        ttk.Label(row, text="in the game window: E engine, 1-5 level",
                  style="Dim.TLabel").pack(side="left")
        self.engine_box.current(self.engines.index(engine) if engine in self.engines else 0)
        self._fill_levels(level)
        self.engine_box.bind("<<ComboboxSelected>>", lambda e: (self._fill_levels(), self._changed()))
        self.level_box.bind("<<ComboboxSelected>>", lambda e: (self._save_opponent(), self._changed()))

    def _fill_levels(self, level=None):
        name = self.engines[self.engine_box.current()]
        if level is None:
            level = self.levels[max(self.level_box.current(), 0)] if self.level_box.get() else 3
        elo = self.approx.get(name, {})
        self.level_box["values"] = [f"level {lv}  (~{elo[lv]} Elo)" if lv in elo else f"level {lv}"
                                    for lv in self.levels]
        self.level_box.current(self.levels.index(level) if level in self.levels else 2)
        self._save_opponent()

    def _changed(self):
        if self.by_key["game"].running():
            self.say("The new opponent plays from the next game start", "info",
                     "Or now: E (engine) and 1-5 (level) in the game window.")

    def _opponent(self):
        if self.engine_box is None:
            return self.game.get("engine", "stockfish"), int(self.game.get("level", 3))
        return self.engines[self.engine_box.current()], self.levels[self.level_box.current()]

    def _save_opponent(self):
        engine, level = self._opponent()
        self.choices[self.game_key] = {"engine": engine, "level": level}
        save_choices(self.choices)

    def _game_command(self):
        engine, level = self._opponent()
        return self.game["game"].format(engine=shlex.quote(engine), level=int(level))

    # ------------------------------------------------------------ messages
    def say(self, text, kind="info", hint=""):
        self.message = (text, kind, hint)

    def _skip_wait(self):
        self.skip_robot_wait = True

    # ------------------------------------------------------------ actions
    def _run(self, fn, *args):
        if self.worker is not None and self.worker.is_alive():
            return False
        self.cancel.clear()
        self.worker = threading.Thread(target=self._guard, args=(fn,) + args, daemon=True)
        self.worker.start()
        return True

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except Cancelled:
            pass
        except (Failed, RuntimeError, OSError) as e:
            self.say(str(e), "error", "See the log below (Log button of that row).")
        finally:
            self.waiting_since = None

    def start_all(self):
        if self.worker is not None and self.worker.is_alive():
            return
        foreign = self._foreign()
        if foreign:
            from tkinter import messagebox
            names = ", ".join(sorted({t for t, _ in foreign}))
            if not messagebox.askyesno(
                    f"Start {self.title}",
                    f"Robot software from another workspace is running ({names}).\n\n"
                    f"Stop it and start {self.title}?", parent=self.root):
                self.say(f"Not started: {names} is running", "error",
                         "Stop it first (Stop everything, or its own panel).")
                return
        self._run(self._start_sequence, [pid for _, pid in foreign])

    def _foreign(self):
        """(title, pid) of cell processes running from another workspace."""
        out = []
        cmds = [(g["title"] + " game", g["game"]) for k, g in self.cfg["games"].items()]
        cmds += [("Robot control", self.cfg["control"]), ("Robot software", self.cfg["robot"])]
        for title, cmd in cmds:
            for pid in find_processes(signature(cmd)):
                if runs_from(pid, self.game["workspace"]) is False:
                    out.append((title, pid))
        return out

    def _up(self, part):
        """Running: its process (from here or a terminal), or its ROS nodes
        (started some other way, e.g. ros2 run) - never start it twice."""
        return part.running() or bool(self.ros.ok and part.nodes and set(part.nodes) <= self.ros.nodes)

    def _start_sequence(self, foreign_pids):
        robot, control, game = self.parts
        if foreign_pids:
            self.say("Stopping the other game's robot software ...", "info")
            stop_pids(foreign_pids)
        self.skip_robot_wait = False
        ip = self.cfg["robot_ip"]
        self._wait(lambda: self.checker.reachable is not False,
                   lambda: (f"Robot not reachable ({ip})", "action",
                            "Switch the robot on at the control box / pendant and check the "
                            "network cable. Starting as soon as it answers."), timeout=None)
        if not self._up(robot):
            self.say("Starting the robot software ...", "info",
                     "Driver, MoveIt, RViz and the gripper - about 20 s.")
            self._show_log("robot")
            robot.start()
        self._wait_ready(robot, 120)
        self._wait_robot()
        if not self._up(control):
            self.say("Starting robot control ...", "info")
            self._show_log("control")
            control.start()
        self._wait_ready(control, 60)
        if not self._up(game):
            self.say(f"Starting the {self.title.lower()} game ...", "info",
                     "Its window opens in a few seconds.")
            self._show_log("game")
            game.start(self._game_command())
        self._wait_ready(game, 90)
        self.game_was_ready = True
        self.say(f"Ready - play in the {self.title} window", "done",
                 "Keep the E-stop at hand. Stop everything when you are done.")

    def _wait(self, ok, message, timeout=60.0, part=None):
        t0 = time.time()
        self.waiting_since = t0
        while True:
            if self.cancel.is_set():
                raise Cancelled()
            if ok():
                return
            if part is not None and not self._up(part):
                raise Failed(f"{part.title} stopped while starting")
            if timeout is not None and time.time() - t0 > timeout:
                raise Failed(f"{part.title if part else 'Waiting'}: not ready after {timeout:.0f} s")
            self.say(*message())
            time.sleep(0.3)

    def _wait_ready(self, part, timeout):
        if not self.ros.ok or not part.nodes:     # cannot see nodes: give it a few seconds
            t_end = time.time() + 8.0
            self._wait(lambda: time.time() > t_end,
                       lambda: (f"Starting {part.title.lower()} ...", "info", ""), part=part)
            return
        self._wait(lambda: set(part.nodes) <= self.ros.nodes,
                   lambda: (f"Starting {part.title.lower()} ...", "info",
                            f"Waiting for {', '.join(part.nodes)}."),
                   timeout=timeout, part=part)

    def _robot_problem(self):
        """What the robot still needs, or None when the program runs."""
        r = self.ros
        if self.skip_robot_wait or not r.ok:
            return None
        if not r.driver_up():
            return ("Waiting for the robot driver ...", "info", "")
        s, m = r.safety_mode, r.robot_mode
        if s in EMERGENCY_STOPS:
            return ("EMERGENCY STOP is pressed", "action",
                    "Release the E-stop, then on the pendant: power on and release the brakes.")
        if s == PROTECTIVE_STOP:
            return ("Protective stop", "action",
                    "Make sure nothing is in the way, then unlock it on the pendant "
                    "(or 'Unlock protective stop' below).")
        if s == SAFEGUARD_STOP:
            return ("Safeguard stop", "action", "A safety input stopped the robot - check the cell.")
        if m == POWER_OFF:
            return ("Robot is powered off", "action",
                    "On the pendant: tap the red button bottom left, ON, then START "
                    "(or 'Power on' and 'Release brakes' below).")
        if m in (POWER_ON, IDLE):
            return ("Release the brakes", "action",
                    "On the pendant: START in the power dialog (or 'Release brakes' below).")
        if r.program_running:
            return None
        return ("Press ▶ (play) on the pendant", "action",
                "Run the External Control program, as for checkers. "
                "Starting robot control as soon as it runs.")

    def _wait_robot(self):
        t0 = time.time()
        self.waiting_since = t0
        while True:
            if self.cancel.is_set():
                raise Cancelled()
            if not self._up(self.parts[0]):
                raise Failed("Robot software stopped - see its log")
            problem = self._robot_problem()
            if problem is None:
                return
            self.say(*problem)
            time.sleep(0.3)

    def stop_all(self):
        self.cancel.set()
        threading.Thread(target=self._stop_sequence, daemon=True).start()

    def _stop_sequence(self):
        if self.worker is not None:
            self.worker.join(timeout=3.0)
        self.say("Stopping ...", "info", "Game, robot control, then the robot software.")
        for p in reversed(self.parts):
            if p.running():
                self.say(f"Stopping {p.title.lower()} ...", "info")
                p.stop()
        self.game_was_ready = False
        self.say("Everything stopped", "info",
                 "Robot: power it off on the pendant if nobody uses it any more.")

    def toggle_part(self, key):
        part = self.by_key[key]
        if not part.running() and self._up(part):
            self.say(f"{part.title} was started some other way", "info",
                     f"Its nodes ({', '.join(part.nodes)}) run, but not from a launch this panel "
                     "knows - stop it where it was started.")
            return
        if part.running():
            if key == "game":
                self.game_was_ready = False
            self._run(lambda: (self.say(f"Stopping {part.title.lower()} ...", "info"),
                               part.stop(), self.say(f"{part.title} stopped", "info")))
        else:
            def go():
                part.start(self._game_command() if key == "game" else None)
                self._wait_ready(part, 120)
                if key == "game":
                    self.game_was_ready = True
                self.say(f"{part.title} running", "done")
            self._run(go)
        self._show_log(key)

    def dashboard(self, service, title, confirm):
        from tkinter import messagebox
        if confirm and not messagebox.askyesno(
                title, "The robot may move.\n\nIs someone at the E-stop and the area clear?",
                parent=self.root):
            return

        def call():
            ok, msg = self.ros.trigger(DASHBOARD + service)
            self.say(f"{title}: {'done' if ok else 'failed'}", "info" if ok else "error",
                     msg + ("" if ok else "  (Local mode on the pendant? Then use the pendant.)"))
        threading.Thread(target=call, daemon=True).start()

    def _fix_usbfs(self):
        cmd = ["pkexec", "sh", "-c", "echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb"]
        threading.Thread(target=lambda: subprocess.run(cmd), daemon=True).start()

    # ------------------------------------------------------------ log
    def _show_log(self, key):
        self.log_part = key
        self._log_size = None

    def _log_terminal(self):
        path = self.by_key[self.log_part].log_path
        for cmd in (["gnome-terminal", "--title", os.path.basename(path), "--",
                     "tail", "-n", "300", "-f", path],
                    ["x-terminal-emulator", "-e", f"tail -n 300 -f {shlex.quote(path)}"],
                    ["xdg-open", path]):
            try:
                subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue

    def _update_log(self):
        part = self.by_key[self.log_part]
        self.log_title.configure(text=f"LOG - {part.title}   (~/.cocohrip/logs/{os.path.basename(part.log_path)})"
                                 if part.log_path.startswith(os.path.expanduser("~/.cocohrip/"))
                                 else f"LOG - {part.title}   ({os.path.basename(part.log_path)})")
        try:
            size = os.path.getsize(part.log_path)
        except OSError:
            size = -1
        if size == getattr(self, "_log_size", None):
            return
        self._log_size = size
        text = "(nothing yet)"
        if size >= 0:
            with open(part.log_path, "rb") as f:
                f.seek(max(0, size - 32000))
                text = ANSI.sub("", f.read().decode(errors="replace"))
                text = "\n".join(text.splitlines()[-300:])
        at_end = self.log.yview()[1] > 0.98
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", text)
        self.log.configure(state="disabled")
        if at_end:
            self.log.see("end")

    # ------------------------------------------------------------ refresh
    def _dot(self, canvas, colour):
        canvas.delete("all")
        canvas.create_oval(2, 2, 14, 14, fill=COL[colour], outline="")

    def _tick(self):
        try:
            self._refresh()
        finally:
            self.root.after(400, self._tick)

    def _refresh(self):
        c, r = self.checker, self.ros
        busy = self.worker is not None and self.worker.is_alive()
        # game closed by the player (Q) while everything else runs
        game = self.by_key["game"]
        if self.game_was_ready and not busy and not self.checker.pids["game"] and not game.own():
            self.game_was_ready = False
            self.say(f"The {self.title.lower()} game was closed", "info",
                     "Start it again with Start in its row, or Stop everything.")
        text, kind, hint = self.message
        self.banner.configure(text=text, bg=COL.get(kind, COL["info"]))
        self.hint.configure(text=hint)
        self.start_btn.state(["disabled"] if busy else ["!disabled"])
        waiting = busy and self.waiting_since and time.time() - self.waiting_since > 20
        if waiting and kind in ("action", "info") and not self.skip_robot_wait:
            self.skip_btn.pack(side="left", padx=8)
        else:
            self.skip_btn.pack_forget()

        rows = self.check_rows
        reach = {True: ("ok", ""), False: ("bad", "no answer"), None: ("off", "checking")}
        self._set(rows["reach"], *reach[c.reachable])
        if not r.ok:
            for k in ("power", "safety", "program"):
                self._set(rows[k], "off", r.error or "")
        elif not r.driver_up():
            for k in ("power", "safety", "program"):
                self._set(rows[k], "off", "robot software not running")
        else:
            m, s = r.robot_mode, r.safety_mode
            if not r.has_modes:
                self._set(rows["power"], "off", "unknown (ur_dashboard_msgs missing)")
                self._set(rows["safety"], "off", "unknown")
            else:
                self._set(rows["power"], *({RUNNING: ("ok", "")}.get(
                    m, ("bad", {POWER_OFF: "powered off", POWER_ON: "brakes locked",
                                IDLE: "brakes locked"}.get(m, f"mode {m}")))))
                self._set(rows["safety"], *(("ok", "") if s in (SAFETY_NORMAL, SAFETY_REDUCED)
                                             else ("bad", {PROTECTIVE_STOP: "protective stop",
                                                           SAFEGUARD_STOP: "safeguard stop",
                                                           6: "EMERGENCY STOP",
                                                           7: "EMERGENCY STOP"}.get(s, f"mode {s}"))))
            p = r.program_running
            self._set(rows["program"], *({True: ("ok", ""), False: ("bad", "press ▶ on the pendant"),
                                          None: ("off", "unknown")}[p]))
        speed = c.camera_speed
        if speed is None:
            self._set(rows["camera"], "bad", "not found - check the cable")
        elif speed < 5000:
            self._set(rows["camera"], "bad", f"{speed} Mbit/s - use a blue USB 3 port")
        elif c.usbfs not in (None, 0) and c.usbfs < 1000:
            self._set(rows["camera"], "wait", f"USB memory limit {c.usbfs} MB (after a reboot)")
        else:
            self._set(rows["camera"], "ok", "")
        if c.usbfs not in (None, 0) and c.usbfs < 1000:
            self.fix_btn.pack(side="left")
        else:
            self.fix_btn.pack_forget()

        for p in self.parts:
            dot, label, state, btn = self.part_rows[p.key]
            running = bool(c.pids[p.key]) or p.own()
            ready = running and (set(p.nodes) <= r.nodes if r.ok and p.nodes else True)
            if not running:
                colour, txt = "off", "stopped"
            elif ready:
                colour, txt = "ok", "running"
            else:
                colour, txt = "wait", "starting ..."
            if running and c.external[p.key]:
                txt += " (started elsewhere)"
            self._dot(dot, colour)
            state.configure(text=txt)
            btn.configure(text="Stop" if running else "Start")
            btn.state(["disabled"] if busy else ["!disabled"])
        self._update_log()

    def _set(self, row, colour, detail):
        dot, label = row[:2]
        self._dot(dot, colour)
        label.configure(text=detail)

    # ------------------------------------------------------------ close
    def on_close(self):
        from tkinter import messagebox
        if any(self.checker.pids[p.key] or p.own() for p in self.parts):
            answer = messagebox.askyesnocancel(
                "Close the panel",
                "Stop the game, robot control and the robot software too?\n\n"
                "Yes = stop everything    No = leave them running", parent=self.root)
            if answer is None:
                return
            if answer:
                self.cancel.set()
                self.say("Stopping everything ...", "info")
                self.banner.configure(text="Stopping everything ...")
                self.root.update()
                for p in reversed(self.parts):
                    p.stop()
        self.cancel.set()
        self.checker.stop()
        self.ros.shutdown()
        self.root.destroy()


# ======================================================================
def _single_instance():
    """Only one panel at a time (one game on the cell)."""
    os.makedirs(HOME_DIR, exist_ok=True)
    f = open(os.path.join(HOME_DIR, "panel.lock"), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", default="chess")
    ap.add_argument("--stop-all", action="store_true", help="stop every game and the robot software")
    ap.add_argument("--no-autostart", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.no_autostart:
        cfg["autostart"] = False

    import tkinter as tk
    from tkinter import messagebox
    if args.stop_all:
        root = tk.Tk(className="cocohrip-stop")
        root.withdraw()
        stopped = stop_everything(cfg)
        messagebox.showinfo("Robot apps", ("Stopped: " + ", ".join(stopped)) if stopped
                            else "Nothing was running.", parent=root)
        root.destroy()
        return 0
    if args.game not in cfg["games"]:
        ap.error(f"game must be one of {', '.join(cfg['games'])}")
    lock = _single_instance()
    root = tk.Tk(className=f"cocohrip-{args.game}")
    if lock is None:
        root.withdraw()
        messagebox.showinfo("Robot cell", "The robot cell panel is already open.", parent=root)
        root.destroy()
        return 1
    here = os.path.dirname(os.path.abspath(__file__))
    icon = os.path.join(here, "icons", f"{args.game}.png")
    if os.path.exists(icon):
        try:
            root.iconphoto(True, tk.PhotoImage(file=icon))
        except tk.TclError:
            pass
    Panel(root, cfg, args.game)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
