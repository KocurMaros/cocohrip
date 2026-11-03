#!/usr/bin/env python3
"""Control panel for the robot cell: one window per game that starts the
robot software, robot_control and the game in the right order, shows what
the robot still needs (power, brakes, the pendant program, a stop), and
stops everything again.

    cocohrip_panel.py chess | checkers     the desktop icons
    cocohrip_panel.py --stop-all            the "Stop robot apps" icon

The window uses CustomTkinter (installed by launcher/install.sh).
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


# dark theme
BG, CARD, CARD2, LINE = "#0f1216", "#181c22", "#232932", "#2d343e"
TEXT, MUTED, FAINT = "#e8eaed", "#8b949e", "#5b636e"
GREEN, GREEN_HI = "#238636", "#2ea043"
RED, RED_HI = "#da3633", "#f85149"
AMBER, BLUE = "#d29922", "#4493f8"
KIND = {"info": BLUE, "action": AMBER, "done": GREEN_HI, "error": RED_HI, "idle": FAINT}
SPIN = "◐◓◑◒"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SHORT = {"builtin": "Simple", "maia": "Maia", "stockfish": "Stockfish", "leela": "Leela"}
ABOUT = {"builtin": "small built-in search - for beginners",
         "maia": "plays like a human of this rating",
         "stockfish": "classic engine, calibrated strength",
         "leela": "neural network - very strong"}
FONTS = ("Ubuntu", "Inter", "Cantarell", "Noto Sans", "DejaVu Sans")
MONO = ("Ubuntu Mono", "DejaVu Sans Mono", "Noto Sans Mono")


def _family(candidates):
    import tkinter.font as tkfont
    have = set(tkfont.families())
    return next((f for f in candidates if f in have), None)


class Ui:
    """Small helpers shared by the panel and the dialogs."""

    def __init__(self, ctk, root):
        self.ctk, self.root = ctk, root
        self.family = _family(FONTS)
        self.mono = _family(MONO)
        self._cache = {}

    def font(self, size, weight="normal", mono=False):
        return self.ctk.CTkFont(family=self.mono if mono else self.family, size=size, weight=weight)

    def set(self, widget, **kw):
        """configure() only what changed (CustomTkinter redraws on every call)."""
        old = self._cache.setdefault(id(widget), {})
        new = {k: v for k, v in kw.items() if old.get(k, object()) != v}
        if new:
            widget.configure(**new)
            old.update(new)

    def card(self, parent, pady=(14, 0)):
        f = self.ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16)
        f.pack(fill="x", pady=pady)
        return f

    def label(self, parent, text="", size=14, colour=TEXT, weight="normal", **kw):
        return self.ctk.CTkLabel(parent, text=text, font=self.font(size, weight), text_color=colour, **kw)

    def button(self, parent, text, command, kind="plain", height=34, width=0, size=13, **kw):
        colours = {"go": (GREEN, GREEN_HI, "white", None), "stop": (RED, RED_HI, "white", None),
                   "plain": (CARD2, LINE, TEXT, None),
                   "outline": ("transparent", CARD2, TEXT, LINE),
                   "link": ("transparent", CARD, MUTED, None)}[kind]
        fg, hover, text_colour, border = colours
        extra = {"border_width": 2, "border_color": border} if border else {"border_width": 0}
        return self.ctk.CTkButton(parent, text=text, command=command, height=height, width=width or 140,
                                  corner_radius=10 if height < 40 else 12, fg_color=fg,
                                  hover_color=hover, text_color=text_colour,
                                  text_color_disabled=FAINT,
                                  font=self.font(size, "bold" if kind in ("go", "stop") else "normal"),
                                  **extra, **kw)

    def ask(self, title, text, options):
        """Modal question. options: [(label, value, kind)], the first is
        the default answer; closing the window = None."""
        ctk = self.ctk
        win = ctk.CTkToplevel(self.root)
        win.title(title)
        win.configure(fg_color=CARD)
        win.resizable(False, False)
        win.transient(self.root)
        self.label(win, title, 17, weight="bold").pack(anchor="w", padx=24, pady=(22, 6))
        self.label(win, text, 14, MUTED, wraplength=400, justify="left").pack(anchor="w", padx=24)
        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(20, 22))
        answer = {"value": None}

        def choose(value):
            answer["value"] = value
            win.destroy()
        for label, value, kind in reversed(options):
            self.button(row, label, lambda v=value: choose(v), kind, height=38, width=120).pack(
                side="right", padx=(8, 0))
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + 160
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")

        def grab():
            try:
                win.grab_set()
                win.focus_force()
            except Exception:                   # noqa: BLE001 - not mapped yet
                win.after(100, grab)
        win.after(80, grab)
        self.root.wait_window(win)
        return answer["value"]


class Panel:
    STEPS = (("reach", "Robot on the network"), ("robot", "Robot software"),
             ("ready", "Robot ready"), ("control", "Robot control"), ("game", None))

    def __init__(self, root, cfg, game_key, ctk):
        self.root, self.ctk = root, ctk
        self.ui = Ui(ctk, root)
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
        autostart = cfg.get("autostart", True)
        self.message = ("Opening ...", "info", "") if autostart else (
            "Ready to start", "idle", "Start runs the robot software, robot control and the game.")
        self.worker = None
        self.stopping = False
        self.cancel = threading.Event()
        self.skip_robot_wait = False
        self.waiting_since = None
        self.log_part = "game"
        self._log_size = None
        self.game_was_ready = False
        self.details_open = False
        self.frame = 0
        self.choices = load_choices()
        self.sel_engine, self.sel_level = self._initial_opponent()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(150, self._tick)
        if autostart:
            root.after(1200, self.start_all)

    # ------------------------------------------------------------ layout
    def _build(self):
        ui, ctk, root = self.ui, self.ctk, self.root
        root.title(f"{self.title} - robot cell")
        root.configure(fg_color=BG)
        root.geometry("600x880")
        root.minsize(520, 480)
        page = ctk.CTkScrollableFrame(root, fg_color=BG, scrollbar_button_color=CARD,
                                      scrollbar_button_hover_color=CARD2)
        self.page = page
        page.pack(fill="both", expand=True, padx=(16, 4), pady=(10, 10))

        # header
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.pack(fill="x", pady=(6, 0))
        icon = self._icon(44)
        if icon is not None:
            ctk.CTkLabel(head, text="", image=icon).pack(side="left", padx=(2, 14))
        names = ctk.CTkFrame(head, fg_color="transparent")
        names.pack(side="left")
        ui.label(names, self.title, 24, weight="bold").pack(anchor="w")
        ui.label(names, "Robot cell", 13, MUTED).pack(anchor="w")

        # status + the main button
        card = ui.card(page, pady=(16, 0))
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(18, 0))
        self.status_dot = ui.label(top, "●", 22, BLUE, width=30, anchor="w")
        self.status_dot.pack(side="left", anchor="n")
        texts = ctk.CTkFrame(top, fg_color="transparent")
        texts.pack(side="left", fill="x", expand=True)
        self.status_title = ui.label(texts, "", 19, weight="bold", anchor="w", justify="left",
                                     wraplength=460)
        self.status_title.pack(fill="x")
        self.status_hint = ui.label(texts, "", 14, MUTED, anchor="w", justify="left", wraplength=460)
        self.status_hint.pack(fill="x", pady=(2, 0))
        self.skip_btn = ui.button(card, "Continue without robot status", self._skip_wait, "link",
                                  height=26, width=220, size=12)
        buttons = ctk.CTkFrame(card, fg_color="transparent")
        buttons.pack(fill="x", padx=20, pady=(16, 20))
        self.primary = ui.button(buttons, "Start", self._primary, "go", height=48, size=16)
        self.primary.pack(side="left", fill="x", expand=True)
        self.secondary = ui.button(buttons, "Stop", self.stop_all, "outline", height=48, width=130,
                                   size=15)
        self.primary_action = "start"
        self.secondary_shown = False

        # steps
        card = ui.card(page)
        self.step_rows = {}
        for i, (key, name) in enumerate(self.STEPS):
            self.step_rows[key] = self._step_row(card, name or f"{self.title} game", first=i == 0)
        ctk.CTkFrame(card, height=1, fg_color=LINE).pack(fill="x", padx=20, pady=(10, 4))
        self.step_rows["camera"] = self._step_row(card, "Camera")
        self.fix_btn = ui.button(self.step_rows["camera"][2], "Fix", self._fix_usbfs, "plain",
                                 height=26, width=56, size=12)
        ctk.CTkFrame(card, height=10, fg_color="transparent").pack()

        # opponent (chess)
        self.has_opponent = "engine" in self.game and self._build_opponent(page)

        # details
        self.more_btn = ui.button(page, "Show details  ▾", self._toggle_details, "link",
                                  height=34, width=150, size=13, anchor="w")
        self.more_btn.pack(anchor="w", pady=(12, 0))
        self.details = ctk.CTkFrame(page, fg_color="transparent")
        self._build_details(self.details)

    def _icon(self, size):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", f"{self.game_key}.png")
        try:
            from PIL import Image
            img = Image.open(path)
            return self.ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        except Exception:                       # noqa: BLE001 - no Pillow / no icon: no image
            return None

    def _step_row(self, parent, name, first=False):
        ui = self.ui
        row = self.ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(16 if first else 6, 0))
        dot = ui.label(row, "●", 16, FAINT, width=22, anchor="w")
        dot.pack(side="left")
        ui.label(row, name, 15).pack(side="left", padx=(4, 0))
        detail = ui.label(row, "", 13, MUTED)
        detail.pack(side="right")
        return dot, detail, row

    def _build_opponent(self, page):
        try:
            from chess_game.engines import APPROX_ELO, CHOICES, LEVELS
        except ImportError:
            return False
        ui, ctk = self.ui, self.ctk
        self.engines = [n for n in CHOICES if n in APPROX_ELO]
        self.approx = APPROX_ELO
        self.levels = list(LEVELS)
        if self.sel_engine not in self.engines:
            self.sel_engine = "stockfish" if "stockfish" in self.engines else self.engines[0]
        card = ui.card(page)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=(16, 10))
        ui.label(head, "Opponent", 16, weight="bold").pack(side="left")
        self.opp_note = ui.label(head, "", 12, MUTED)
        self.opp_note.pack(side="right")
        seg = dict(height=38, corner_radius=10, border_width=3, fg_color=CARD2,
                   selected_color=BLUE, selected_hover_color="#2f7be0",
                   unselected_color=CARD2, unselected_hover_color=LINE,
                   text_color=TEXT, font=ui.font(14))
        self.engine_seg = ctk.CTkSegmentedButton(card, values=[SHORT.get(n, n) for n in self.engines],
                                                 command=self._engine_picked, **seg)
        self.engine_seg.pack(fill="x", padx=20)
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(14, 0))
        ui.label(row, "Level", 14, MUTED).pack(side="left", padx=(0, 12))
        self.level_seg = ctk.CTkSegmentedButton(row, values=[str(lv) for lv in self.levels],
                                                command=self._level_picked, width=230, **seg)
        self.level_seg.pack(side="left")
        self.elo_label = ui.label(row, "", 26, weight="bold")
        self.elo_label.pack(side="right")
        self.about_label = ui.label(card, "", 13, MUTED, anchor="w")
        self.about_label.pack(fill="x", padx=20, pady=(10, 0))
        ui.label(card, "In the game window: E switches the engine, 1-5 the level.", 12, FAINT,
                 anchor="w").pack(fill="x", padx=20, pady=(2, 16))
        self.engine_seg.set(SHORT.get(self.sel_engine, self.sel_engine))
        self.level_seg.set(str(self.sel_level))
        self._show_opponent()
        return True

    def _build_details(self, parent):
        ui, ctk = self.ui, self.ctk
        # parts
        card = ui.card(parent, pady=(6, 0))
        ui.label(card, "Parts", 16, weight="bold").pack(anchor="w", padx=20, pady=(16, 4))
        self.part_rows = {}
        for p in self.parts:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=4)
            names = ctk.CTkFrame(row, fg_color="transparent")
            names.pack(side="left")
            ui.label(names, p.title, 14, anchor="w").pack(anchor="w")
            state = ui.label(names, p.note, 12, MUTED, anchor="w")
            state.pack(anchor="w")
            log = ui.button(row, "Log", lambda k=p.key: self._show_log(k), "plain", height=30,
                            width=52, size=12)
            log.pack(side="right")
            stop = ui.button(row, "Stop", lambda k=p.key: self.stop_part(k), "plain", height=30,
                             width=62, size=12)
            stop.pack(side="right", padx=6)
            start = ui.button(row, "Start", lambda k=p.key: self.start_part(k), "plain", height=30,
                              width=76, size=12)
            start.pack(side="right")
            self.part_rows[p.key] = (state, start, stop)
        ctk.CTkFrame(card, height=12, fg_color="transparent").pack()

        # robot buttons (UR dashboard)
        card = ui.card(parent)
        ui.label(card, "Robot buttons", 16, weight="bold").pack(anchor="w", padx=20, pady=(16, 0))
        ui.label(card, "Only in Remote Control mode on the pendant - otherwise use the pendant.",
                 12, MUTED, anchor="w").pack(fill="x", padx=20, pady=(0, 8))
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=20, pady=(0, 18))
        for i, (text, service, confirm) in enumerate((
                ("Power on", "power_on", False), ("Release brakes", "brake_release", True),
                ("Unlock stop", "unlock_protective_stop", True), ("Play program", "play", True),
                ("Stop program", "stop", False))):
            grid.columnconfigure(i % 3, weight=1)
            ui.button(grid, text, lambda s=service, t=text, c=confirm: self.dashboard(s, t, c),
                      "plain", height=34, size=13).grid(row=i // 3, column=i % 3, sticky="ew",
                                                        padx=(0 if i % 3 == 0 else 6, 0), pady=3)

        # log
        card = ui.card(parent)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=(14, 8))
        self.log_title = ui.label(head, "Log", 16, weight="bold")
        self.log_title.pack(side="left")
        ui.button(head, "Open in a terminal", self._log_terminal, "plain", height=28, width=140,
                  size=12).pack(side="right")
        self.log = ctk.CTkTextbox(card, height=220, fg_color=BG, text_color="#c9d1d9",
                                  font=ui.font(12, mono=True), wrap="word", corner_radius=10,
                                  border_width=0)
        self.log.pack(fill="x", padx=14, pady=(0, 14))
        self.log.configure(state="disabled")

    def _toggle_details(self):
        self.details_open = not self.details_open
        if self.details_open:
            self.details.pack(fill="x")
            self.more_btn.configure(text="Hide details  ▴")
        else:
            self.details.pack_forget()
            self.more_btn.configure(text="Show details  ▾")
        self._log_size = None

    # ------------------------------------------------------------ opponent
    def _initial_opponent(self):
        mine = self.choices.get(self.game_key, {})
        return (mine.get("engine", self.game.get("engine", "stockfish")),
                int(mine.get("level", self.game.get("level", 3))))

    def _engine_picked(self, value):
        self.sel_engine = next(n for n in self.engines if SHORT.get(n, n) == value)
        self._opponent_changed()

    def _level_picked(self, value):
        self.sel_level = int(value)
        self._opponent_changed()

    def _opponent_changed(self):
        self.choices[self.game_key] = {"engine": self.sel_engine, "level": self.sel_level}
        save_choices(self.choices)
        self._show_opponent()

    def _show_opponent(self):
        elo = self.approx.get(self.sel_engine, {}).get(self.sel_level)
        self.elo_label.configure(text=f"~{elo} Elo" if elo else "")
        self.about_label.configure(text=f"{SHORT.get(self.sel_engine, self.sel_engine)} level "
                                        f"{self.sel_level} - {ABOUT.get(self.sel_engine, '')}")
        running = self.by_key["game"].running()
        self.opp_note.configure(text="from the next game start" if running else "")

    def _opponent(self):
        return self.sel_engine, self.sel_level

    def _game_command(self):
        engine, level = self._opponent()
        return self.game["game"].format(engine=shlex.quote(engine), level=int(level))

    # ------------------------------------------------------------ messages
    def say(self, text, kind="info", hint=""):
        self.message = (text, kind, hint)

    def _skip_wait(self):
        self.skip_robot_wait = True

    # ------------------------------------------------------------ actions
    def _primary(self):
        if self.primary_action == "start":
            self.start_all()
        elif self.primary_action == "stop":
            self.stop_all()

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
            self.say(str(e), "error", "Show details - Log shows that part's output.")
        finally:
            self.waiting_since = None

    def start_all(self):
        if self.worker is not None and self.worker.is_alive():
            return
        foreign = self._foreign()
        if foreign:
            names = ", ".join(sorted({t for t, _ in foreign}))
            if not self.ui.ask(f"Start {self.title}?",
                               f"{names} is running from another workspace. Both use the same "
                               f"robot. Stop it and start {self.title}?",
                               [("Stop it and start", True, "go"), ("Cancel", False, "plain")]):
                self.say(f"Not started - {names} is running", "error",
                         "Stop it first, then start again.")
                return
        self._run(self._start_sequence, [pid for _, pid in foreign])

    def _foreign(self):
        """(title, pid) of cell processes running from another workspace."""
        out = []
        cmds = [(g["title"] + " game", g["game"]) for g in self.cfg["games"].values()]
        cmds += [("Robot control", self.cfg["control"]), ("Robot software", self.cfg["robot"])]
        table = process_table()
        own = {p.proc.pid for p in self.parts if p.own()}
        for title, cmd in cmds:
            for pid in find_processes(signature(cmd), table):
                if pid not in own and runs_from(pid, self.game["workspace"]) is False:
                    out.append((title, pid))
        return out

    def _up(self, part):
        """Running: its process (from here or a terminal), or its ROS nodes
        (started some other way, e.g. ros2 run) - never start it twice."""
        return part.running() or bool(self.ros.ok and part.nodes and set(part.nodes) <= self.ros.nodes)

    def _start_sequence(self, foreign_pids):
        robot, control, game = self.parts
        if foreign_pids:
            self.say("Stopping the other game ...", "info")
            stop_pids(foreign_pids)
        self.skip_robot_wait = False
        ip = self.cfg["robot_ip"]
        self._wait(lambda: self.checker.reachable is not False or self.skip_robot_wait,
                   lambda: ("Switch the robot on", "action",
                            f"The robot ({ip}) does not answer. Switch it on at the control box "
                            "and check the network cable - starting as soon as it answers."),
                   timeout=None)
        if not self._up(robot):
            self.say("Starting the robot software", "info",
                     "Driver, MoveIt, RViz and the gripper - about 20 s.")
            self._show_log("robot")
            robot.start()
        self._wait_ready(robot, 120)
        self._wait_robot()
        if not self._up(control):
            self.say("Starting robot control", "info")
            self._show_log("control")
            control.start()
        self._wait_ready(control, 60)
        if not self._up(game):
            self.say(f"Opening the {self.title.lower()} game", "info",
                     "Its window opens in a few seconds.")
            self._show_log("game")
            game.start(self._game_command())
        self._wait_ready(game, 90)
        self.game_was_ready = True
        self.say(f"Ready - play in the {self.title} window", "done",
                 "Keep the E-stop at hand. Stop when you are done.")

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
                       lambda: (f"Starting {part.title.lower()}", "info", ""), part=part)
            return
        self._wait(lambda: set(part.nodes) <= self.ros.nodes,
                   lambda: (self.message[0], "info", self.message[2]),
                   timeout=timeout, part=part)

    def _robot_problem(self):
        """(title, kind, hint, short) of what the robot still needs, or None."""
        r = self.ros
        if self.skip_robot_wait or not r.ok:
            return None
        if not r.driver_up():
            return ("Connecting to the robot", "info", "", "connecting")
        s, m = r.safety_mode, r.robot_mode
        if s in EMERGENCY_STOPS:
            return ("Emergency stop is pressed", "error",
                    "Release the E-stop, then on the pendant: power on and release the brakes.",
                    "emergency stop")
        if s == PROTECTIVE_STOP:
            return ("Protective stop", "error",
                    "Make sure nothing is in the way, then unlock it on the pendant.",
                    "protective stop")
        if s == SAFEGUARD_STOP:
            return ("Safeguard stop", "error", "A safety input stopped the robot - check the cell.",
                    "safeguard stop")
        if m == POWER_OFF:
            return ("Power on the robot", "action",
                    "On the pendant: the red button bottom left, then ON and START.",
                    "powered off")
        if m in (POWER_ON, IDLE):
            return ("Release the brakes", "action", "On the pendant: START in the power dialog.",
                    "brakes locked")
        if r.program_running:
            return None
        return ("Press ▶ on the pendant", "action",
                "Run the External Control program, as for checkers. Robot control starts "
                "as soon as it runs.", "press ▶ on the pendant")

    def _wait_robot(self):
        self.waiting_since = time.time()
        while True:
            if self.cancel.is_set():
                raise Cancelled()
            if not self._up(self.parts[0]):
                raise Failed("Robot software stopped - see its log")
            problem = self._robot_problem()
            if problem is None:
                return
            self.say(*problem[:3])
            time.sleep(0.3)

    def stop_all(self):
        if self.stopping:
            return
        self.stopping = True
        self.cancel.set()
        threading.Thread(target=self._stop_sequence, daemon=True).start()

    def _stop_sequence(self):
        try:
            if self.worker is not None:
                self.worker.join(timeout=3.0)
            for p in reversed(self.parts):
                if p.running():
                    self.say(f"Stopping {p.title.lower()}", "info",
                             "Game, robot control, then the robot software.")
                    p.stop()
            self.game_was_ready = False
            self.say("Everything is stopped", "idle",
                     "Power the robot off on the pendant if nobody needs it.")
        finally:
            self.stopping = False

    def start_part(self, key):
        part = self.by_key[key]
        if self._up(part):
            self.say(f"{part.title} is already running", "info", "")
            return

        def go():
            self.say(f"Starting {part.title.lower()}", "info")
            part.start(self._game_command() if key == "game" else None)
            self._wait_ready(part, 120)
            if key == "game":
                self.game_was_ready = True
            self.say(f"{part.title} is running", "done")
        self._show_log(key)
        self._run(go)

    def stop_part(self, key):
        part = self.by_key[key]
        if not part.running():
            if self._up(part):
                self.say(f"{part.title} was started some other way", "info",
                         "Stop it where it was started.")
            return
        if key == "game":
            self.game_was_ready = False

        def go():
            self.say(f"Stopping {part.title.lower()}", "info")
            part.stop()
            self.say(f"{part.title} is stopped", "idle")
        self._show_log(key)
        self._run(go)

    def dashboard(self, service, title, confirm):
        if confirm and not self.ui.ask(
                title, "The robot may move. Is someone at the E-stop and the area clear?",
                [("Yes, go ahead", True, "go"), ("Cancel", False, "plain")]):
            return

        def call():
            ok, msg = self.ros.trigger(DASHBOARD + service)
            self.say(f"{title}: {'done' if ok else 'not possible'}", "info" if ok else "error",
                     msg + ("" if ok else " - in Local mode, use the pendant."))
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
        self.ui.set(self.log_title, text=f"Log - {part.title}")
        try:
            size = os.path.getsize(part.log_path)
        except OSError:
            size = -1
        if size == self._log_size:
            return
        self._log_size = size
        text = "Nothing yet."
        if size >= 0:
            with open(part.log_path, "rb") as f:
                f.seek(max(0, size - 32000))
                text = ANSI.sub("", f.read().decode(errors="replace"))
                text = "\n".join(text.splitlines()[-300:])
        at_end = self.log._textbox.yview()[1] > 0.98
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", text)
        self.log.configure(state="disabled")
        if at_end:
            self.log.see("end")

    # ------------------------------------------------------------ refresh
    def _tick(self):
        try:
            self._refresh()
        finally:
            self.root.after(150, self._tick)

    def _dot(self, key, colour, detail, spin=False):
        dot, label = self.step_rows[key][:2]
        self.ui.set(dot, text=SPIN[self.frame % 4] if spin else "●", text_color=colour)
        self.ui.set(label, text=detail)

    def _refresh(self):
        ui, c, r = self.ui, self.checker, self.ros
        self.frame += 1
        busy = (self.worker is not None and self.worker.is_alive()) or self.stopping
        running = {p.key: bool(c.pids[p.key]) or p.own() for p in self.parts}
        ready = {p.key: running[p.key] and (set(p.nodes) <= r.nodes if r.ok and p.nodes else True)
                 for p in self.parts}
        if self.game_was_ready and not busy and not running["game"]:
            self.game_was_ready = False
            self.say(f"The {self.title.lower()} game was closed", "idle",
                     "Start opens it again. Stop ends everything.")

        # status card
        text, kind, hint = self.message
        ui.set(self.status_dot, text=SPIN[self.frame % 4] if busy else "●",
               text_color=KIND.get(kind, BLUE))
        ui.set(self.status_title, text=text)
        ui.set(self.status_hint, text=hint)
        waiting = busy and not self.stopping and self.waiting_since \
            and time.time() - self.waiting_since > 20 and not self.skip_robot_wait \
            and kind in ("action", "info")
        if waiting:
            self.skip_btn.pack(anchor="w", padx=56, pady=(6, 0), before=self.primary.master)
        else:
            self.skip_btn.pack_forget()

        # main button: Start / Stop (+ Stop beside Start when only some parts run)
        any_running = any(running.values())
        if self.stopping:
            action, label, kind_btn = "none", "Stopping ...", "stop"
        elif busy or all(ready.values()):
            action, label, kind_btn = "stop", "Stop", "stop"
        else:
            action, label, kind_btn = "start", "Start", "go"
        if action != self.primary_action or self.primary.cget("text") != label:
            self.primary_action = action
            fg, hover = (RED, RED_HI) if kind_btn == "stop" else (GREEN, GREEN_HI)
            ui.set(self.primary, text=label, fg_color=fg, hover_color=hover,
                   state="disabled" if action == "none" else "normal")
        show_secondary = action == "start" and any_running
        if show_secondary != self.secondary_shown:
            self.secondary_shown = show_secondary
            if show_secondary:
                self.secondary.pack(side="left", padx=(10, 0))
            else:
                self.secondary.pack_forget()

        # steps
        ip = self.cfg["robot_ip"]
        self._dot("reach", *{True: (GREEN_HI, ip), False: (RED_HI, "no answer"),
                             None: (FAINT, "checking")}[c.reachable])
        for key in ("robot", "control", "game"):
            elsewhere = "  (terminal)" if running[key] and c.external[key] else ""
            if ready[key]:
                done = "window open" if key == "game" else "running"
                self._dot(key, GREEN_HI, done + elsewhere)
            elif running[key]:
                self._dot(key, AMBER, "starting", spin=True)
            else:
                self._dot(key, FAINT, "off")
        if not r.ok:
            self._dot("ready", FAINT, "no ROS - not checked")
        elif self.skip_robot_wait:
            self._dot("ready", FAINT, "not checked")
        elif not r.driver_up():
            self._dot("ready", FAINT, "-")
        else:
            problem = self._robot_problem()
            if problem is None:
                self._dot("ready", GREEN_HI, "ready")
            else:
                self._dot("ready", RED_HI if problem[1] == "error" else AMBER, problem[3])
        speed, usbfs = c.camera_speed, c.usbfs
        needs_fix = usbfs not in (None, 0) and usbfs < 1000
        if speed is None:
            self._dot("camera", RED_HI, "not found - check the cable")
        elif speed < 5000:
            self._dot("camera", RED_HI, "USB 2 - use a blue USB 3 port")
        elif needs_fix:
            self._dot("camera", AMBER, "memory limit after a reboot")
        else:
            self._dot("camera", GREEN_HI, "USB 3")
        if needs_fix:
            self.fix_btn.pack(side="right", padx=(10, 0), before=self.step_rows["camera"][1])
        else:
            self.fix_btn.pack_forget()

        if self.has_opponent:
            ui.set(self.opp_note, text="from the next game start" if running["game"] else "")

        # details
        if self.details_open:
            for p in self.parts:
                state, start, stop = self.part_rows[p.key]
                if ready[p.key]:
                    txt, col = "running" + ("  (terminal)" if c.external[p.key] else ""), GREEN_HI
                elif running[p.key]:
                    txt, col = "starting", AMBER
                else:
                    txt, col = "off", MUTED
                ui.set(state, text=txt, text_color=col)
                ui.set(start, state="disabled" if busy or running[p.key] else "normal")
                ui.set(stop, state="disabled" if busy or not running[p.key] else "normal")
            self._update_log()

    # ------------------------------------------------------------ close
    def on_close(self):
        if any(self.checker.pids[p.key] or p.own() for p in self.parts):
            answer = self.ui.ask(
                "Close the panel?",
                "The game, robot control and the robot software are still running.",
                [("Stop everything", "stop", "stop"), ("Leave running", "leave", "plain"),
                 ("Cancel", None, "plain")])
            if answer is None:
                return
            if answer == "stop":
                self.cancel.set()
                self.say("Stopping everything", "info", "")
                self._refresh()
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


def _note_window(ctk, title, text, work=None):
    """A small window with a message (and `work` run first, showing
    'please wait')."""
    root = ctk.CTk(className="cocohrip-stop")
    root.title(title)
    root.configure(fg_color=CARD)
    root.minsize(400, 1)
    root.resizable(False, False)
    ui = Ui(ctk, root)
    ui.label(root, title, 18, weight="bold").pack(anchor="w", padx=26, pady=(24, 6))
    msg = ui.label(root, "Please wait ..." if work else text, 14, MUTED, wraplength=380,
                   justify="left", anchor="w")
    msg.pack(fill="x", padx=26)
    btn = ui.button(root, "Close", root.destroy, "plain", height=38, width=110)
    btn.pack(anchor="e", padx=26, pady=(22, 22))
    if work:
        btn.configure(state="disabled")

        def run():
            result = work()
            root.after(0, lambda: (msg.configure(text=result), btn.configure(state="normal")))
        root.after(300, lambda: threading.Thread(target=run, daemon=True).start())
    root.mainloop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", default="chess")
    ap.add_argument("--stop-all", action="store_true", help="stop every game and the robot software")
    ap.add_argument("--no-autostart", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.no_autostart:
        cfg["autostart"] = False
    try:
        import customtkinter as ctk
    except ImportError:
        print("customtkinter is missing: run launcher/install.sh", file=sys.stderr)
        subprocess.run(["zenity", "--error", "--text",
                        "The robot cell panel needs customtkinter.\nRun: bash launcher/install.sh"],
                       stderr=subprocess.DEVNULL)
        return 1
    ctk.set_appearance_mode("dark")

    if args.stop_all:
        def work():
            stopped = stop_everything(cfg, log=lambda *a: None)
            return ("Stopped: " + ", ".join(stopped) + ".") if stopped else "Nothing was running."
        _note_window(ctk, "Stop robot apps", "", work)
        return 0
    if args.game not in cfg["games"]:
        ap.error(f"game must be one of {', '.join(cfg['games'])}")
    lock = _single_instance()
    if lock is None:
        _note_window(ctk, "Already open", "The robot cell panel is already open.")
        return 1
    root = ctk.CTk(className=f"cocohrip-{args.game}")
    icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", f"{args.game}.png")
    if os.path.exists(icon):
        try:
            import tkinter as tk
            root.iconphoto(True, tk.PhotoImage(file=icon))
        except Exception:                       # noqa: BLE001
            pass
    Panel(root, cfg, args.game, ctk)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
