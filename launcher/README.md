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

It installs what the window needs (`python3-tk`, Pillow, CustomTkinter; asks
for the password if something is missing), copies the panel to
`~/.local/share/cocohrip/launcher/`, and puts the icons on the desktop and in
the app list. If GNOME shows an icon as untrusted: right-click it and choose
**Allow Launching**.

## Using it

1. Switch the robot on at the pendant, as before. Have the E-stop at hand.
2. Double-click **Chess robot** (or **Checkers robot**). The panel (dark,
   one window) starts everything by itself and shows it as steps:
   - **Robot on the network**: it answers on 192.168.0.5.
   - **Robot software**: driver, MoveIt, RViz, gripper (`bringup_all.launch.py`).
   - **Robot ready**: power, brakes, no stop, External Control running. Until
     then the big card says what to do on the pendant, e.g. "Press ▶ on
     the pendant".
   - **Robot control** (`robot_control new_launch.py`).
   - **The game**: its window opens.
   - **Camera**: it must be on USB 3. After a reboot, **Fix** sets the USB
     memory limit the camera needs (asks for the password).
3. For chess, choose the **opponent** (Simple, Maia, Stockfish, Leela) and
   the **level 1-5**. The rough Elo is shown, and the choice is remembered.
   It applies when the game starts; inside the game, E and 1-5 change it.
4. Play. The one big button is **Start** or **Stop**. Stop ends the game,
   robot control and the robot software, in that order and gently, like
   Ctrl+C in their terminals. If you closed the game (Q), **Start** opens it
   again and **Stop** ends everything. Closing the panel asks whether to stop
   everything or leave it running.

**Show details** opens three more sections:
- each part with **Start**, **Stop** and **Log**;
- the **robot buttons** (power on, release brakes, unlock a protective stop,
  play or stop the program). These only work when the pendant is in *Remote
  Control*; in *Local* mode use the pendant as usual;
- the **log** of the selected part. All logs are kept in
  `~/.cocohrip/logs/<game>_<part>.log`, with the previous run in `.log.1`.

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
- `cocohrip_panel.py`: the panel (CustomTkinter). `--stop-all` is the stop icon.
- `install.sh`: installs the icons. `icons/make_icons.py` redraws them.
- `test_panel.py`: `python3 -m pytest launcher/test_panel.py`, tests that
  start and stop processes and need no robot or ROS.
