# Desktop icons for the robot cell

Three icons on the robot PC's desktop:

| Icon | What it does |
|---|---|
| **Chess robot** | opens the control panel and starts everything for chess (from `~/chess_ws`) |
| **Checkers robot** | the same for checkers (from `~/checkers_ws`) |
| **Stop robot apps** | stops every game, robot_control and the robot software, wherever they were started |

## Install (once, and again after updates)

In the desktop session of the robot PC (not over ssh), from a clone of this
branch:

```bash
bash launcher/install.sh          # --dock also pins Chess and Checkers to the dock
```

It installs `python3-tk` if missing (asks for the password), copies the panel
to `~/.local/share/cocohrip/launcher/`, and puts the icons on the desktop and
in the app list. If GNOME shows an icon as untrusted: right-click it and
choose **Allow Launching**.

## Using it

1. Switch the robot on at the pendant, as before. Have the E-stop at hand.
2. Double-click **Chess robot** (or **Checkers robot**). The panel starts:
   - the **robot software**: driver, MoveIt, RViz, gripper (`bringup_all.launch.py`);
   - then it waits until the robot is ready. The banner says what is
     missing: powered off, brakes, an emergency or protective stop, or
     "press ▶ on the pendant" (the External Control program);
   - **robot control** (`robot_control new_launch.py`);
   - the **game**: its window opens. For chess, the opponent and level
     come from the panel, with the rough Elo of each level.
3. Play. **Stop everything** stops the game, robot control and the
   robot software, in that order and gently, like Ctrl+C in their
   terminals. Closing the panel asks whether to stop them too.

The rows show what runs: green = running, orange = starting, grey =
stopped. **Start**/**Stop** in a row restarts only that part, e.g. the game
after Q. **Log** shows that part's output, also kept in
`~/.cocohrip/logs/<game>_<part>.log` (the previous run is in `.log.1`).

What the panel checks:
- the robot answers on the network (ping 192.168.0.5);
- power, brakes, safety and the pendant program, read from the UR driver;
- the camera is on USB 3. After a reboot, **Fix** sets the USB memory
  limit the Ximea camera needs (asks for the password).

**Robot buttons** (power on, release brakes, play or stop the program, unlock
a protective stop) only work when the pendant is in *Remote Control*. In
*Local* mode the robot refuses them, and you use the pendant as usual.

Only one panel is open at a time. If the other game is running, the panel
offers to stop it first: both games use the same robot, and each uses its
own workspace. Software started by hand in terminals is recognised, so it is
reused rather than started twice, and can still be stopped from the panel.

## Settings

`~/.cocohrip/launcher.json` changes the defaults (see `DEFAULTS` in
`cocohrip_panel.py`). Only list what changes, for example:

```json
{"robot_ip": "192.168.0.5", "autostart": true,
 "games": {"checkers": {"workspace": "~/checkers_ws"}}}
```

## Files

- `cocohrip-panel`: the wrapper the icons run. It sets up a clean ROS
  environment (ROS + that game's workspace, plus the DDS settings from
  `~/.bashrc` such as `ROS_DOMAIN_ID`), then starts the panel.
- `cocohrip_panel.py`: the panel (Tk). `--stop-all` is the stop icon.
- `install.sh`: installs the icons. `icons/make_icons.py` redraws them.
- `test_panel.py`: `python3 -m pytest launcher/test_panel.py`, tests that
  start and stop processes and need no robot or ROS.
