#!/bin/bash
# Desktop icons for the robot cell:
#   Chess robot / Checkers robot - a control panel that starts the robot
#                                  software, robot_control and the game, and stops them
#   Stop robot apps              - stops every game and the robot software
# Run it as the desktop user, in the desktop session (not over ssh), and
# again after updates:   bash launcher/install.sh [--dock]
#   --dock   also pin Chess and Checkers to the dock
set -e
SRC=$(dirname "$(readlink -f "$0")")
DEST=$HOME/.local/share/cocohrip/launcher
APPS=$HOME/.local/share/applications
DESKTOP=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")

echo "== python3-tk (the panel's window)"
if python3 -c "import tkinter" 2>/dev/null; then echo "   installed"
else sudo apt-get install -y python3-tk
fi

echo "== panel -> $DEST"
mkdir -p "$DEST/icons" "$APPS" "$DESKTOP"
cp "$SRC/cocohrip_panel.py" "$SRC/cocohrip-panel" "$DEST/"
cp "$SRC"/icons/*.png "$DEST/icons/"
chmod +x "$DEST/cocohrip-panel" "$DEST/cocohrip_panel.py"

entry() {   # file name comment arguments icon
  cat > "$APPS/$1" <<END
[Desktop Entry]
Type=Application
Version=1.0
Name=$2
Comment=$3
Exec=$DEST/cocohrip-panel $4
Icon=$DEST/icons/$5.png
Terminal=false
Categories=Education;
StartupWMClass=Cocohrip-$5
END
}
entry cocohrip-chess.desktop "Chess robot" \
  "Start the robot software and the chess game; stop them again" chess chess
entry cocohrip-checkers.desktop "Checkers robot" \
  "Start the robot software and the checkers game; stop them again" checkers checkers
entry cocohrip-stop.desktop "Stop robot apps" \
  "Stop the games, robot_control and the robot software" --stop-all stop
update-desktop-database "$APPS" 2>/dev/null || true

echo "== icons on the desktop ($DESKTOP)"
for f in chess checkers stop; do
  cp "$APPS/cocohrip-$f.desktop" "$DESKTOP/"
  chmod +x "$DESKTOP/cocohrip-$f.desktop"
  # GNOME: "Allow Launching" (otherwise right-click the icon -> Allow Launching)
  gio set "$DESKTOP/cocohrip-$f.desktop" metadata::trusted true 2>/dev/null \
    || echo "   right-click cocohrip-$f on the desktop -> Allow Launching"
done

if [ "$1" = "--dock" ]; then
  echo "== dock"
  fav=$(gsettings get org.gnome.shell favorite-apps 2>/dev/null || echo "[]")
  new=$(python3 - "$fav" <<'END'
import ast, sys
s = sys.argv[1].replace("@as ", "")
apps = ast.literal_eval(s) if s.startswith("[") else []
for a in ("cocohrip-chess.desktop", "cocohrip-checkers.desktop"):
    if a not in apps:
        apps.append(a)
print(str(apps))
END
)
  gsettings set org.gnome.shell favorite-apps "$new" && echo "   pinned"
fi
echo "done: double-click 'Chess robot' or 'Checkers robot' on the desktop"
