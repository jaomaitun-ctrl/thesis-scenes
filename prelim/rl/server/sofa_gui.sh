#!/usr/bin/env bash
# runSofa on a virtual display (Xvfb :1), shown in the browser through x11vnc + noVNC.
#   /srv/train/web/sofa_gui.sh display   # Xvfb + x11vnc (localhost:5900, password) + noVNC on :6080
#   /srv/train/web/sofa_gui.sh sofa      # runSofa (ImGui) with the patient 44 insertion scene
#   /srv/train/web/sofa_gui.sh start     # both, in tmux session sofagui
#   /srv/train/web/sofa_gui.sh stop      # stop everything (runSofa renders continuously on the CPU)
set -uo pipefail
case "${1:-}" in
display)
  Xvfb :1 -screen 0 800x600x24 -nolisten tcp &
  sleep 2
  x11vnc -display :1 -rfbauth "$HOME/.vnc/passwd" -forever -shared -localhost -rfbport 5900 -quiet &
  sleep 1
  exec websockify --web /usr/share/novnc 0.0.0.0:6080 localhost:5900 ;;
sofa)
  E=/srv/train/sofa/.pixi/envs/supported-plugins
  S=/srv/train/thesis-scenes
  export DISPLAY=:1 SOFA_ROOT=$E LIBGL_ALWAYS_SOFTWARE=1
  export LD_LIBRARY_PATH=$E/lib PYTHONHOME=$E
  export PYTHONPATH=$E/plugins/SofaPython3/lib/python3/site-packages:$S
  export ROBOT_CASE=${ROBOT_CASE:-design_patient44_right_r3_csf}
  export ROBOT_WALL=${ROBOT_WALL:-$S/walls/patient44_right_entryhole.stl}
  cd $S
  while true; do
    # default: the v2 policy drives the robot (rl/policy_scene.py); SOFA_SCENE=schedule for the fitted schedule
    if [ "${SOFA_SCENE:-policy}" = schedule ]; then SCN=$S/cosserat_robot_01_from_matlab.py; else SCN=$S/rl/policy_scene.py; fi
    $E/bin/runSofa -l SofaPython3 -g ${SOFA_GUI:-imgui} $SCN
    echo "runSofa exited ($?); restarting in 3 s"; sleep 3
  done ;;
start)
  tmux new-session -d -s sofagui -n display "$0 display"
  sleep 4
  tmux new-window -t sofagui -n sofa "$0 sofa > /srv/train/web/sofa_gui.log 2>&1" ;;
stop)
  tmux kill-session -t sofagui 2>/dev/null
  pkill -u "$USER" -f "^Xvfb :1 " ; pkill -u "$USER" -f "^x11vnc -display :1"
  pkill -u "$USER" -f "^/usr/bin/python3 /usr/bin/websockify --web /usr/share/novnc" ; pkill -u "$USER" -f "websockify --web /usr/share/novnc 0.0.0.0:6080"
  pkill -u "$USER" -f "^/srv/train/sofa/.pixi/envs/supported-plugins/bin/runSofa -l SofaPython3 -g"
  echo stopped ;;
*) echo "usage: $0 display|sofa|start|stop"; exit 1 ;;
esac
