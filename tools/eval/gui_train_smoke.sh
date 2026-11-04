#!/usr/bin/env bash
# tools/eval/gui_train_smoke.sh — does the GUI survive the first training frames? Headless, on a hidden X server.
#
#   tools/eval/gui_train_smoke.sh <checkout> <dataset> <images folder> <strategy> <ppisp 0|1> <bilateral 0|1> <hdr 0|1> [port]
#
# Starts build/LichtFeld-Studio under xvfb-run with an ISOLATED HOME (never ~/.lichtfeld), loads <dataset> through the
# GUI's own MCP server (scene_load_dataset), sets the optimisation parameters through its Python console (editor_run),
# starts training and watches ~45 s for a crash report. Prints one RESULT line: OK or CRASH. The NVIDIA Vulkan driver
# runs under Xvfb (measured 2026-11-04: the real adapter, the same crash as on the desktop).
#
# The 2026-11-04 bug it pins: composePixelState grew the OCIO LUT / dummy buffers while recording — resizeDeviceBuffer
# splits the command batch (GrowthBatchSplitGuard), the pass went on recording into an ended command buffer, and the
# driver segfaulted at the first compose of EVERY training (LDR and HDR alike). ensureOcioBuffers now grows them before
# the pass takes its command buffer. Upstream 0.5.4 passes this script; the fork before the fix crashed in every mode.
set -u
CK=$1; DS=$2; IMG=$3; STRAT=$4; PP=$5; BG=$6; HDR=$7; PORT=${8:-45999}
WORK=${GUI_SMOKE_WORK:-/tmp/lfs-gui-smoke}; H=$WORK/home_$PORT; rm -rf "$H"; mkdir -p "$H"
LOGS=$H/.lichtfeld/logs
call(){ curl -s -m 60 "http://127.0.0.1:$PORT/mcp" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d "$1"; }
py(){ printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"editor_run","arguments":{"show_console":False,"wait_for_completion":True,"timeout_ms":30000,"code":sys.stdin.read()}}}))'; }
( HOME=$H LD_LIBRARY_PATH="$CK/build/Build/lib:$CK/build/vcpkg_installed/x64-linux/lib" env -u WAYLAND_DISPLAY -u DISPLAY \
  timeout -s INT 170 xvfb-run -a -s "-screen 0 1920x1080x24" "$CK/build/LichtFeld-Studio" --mcp-port=$PORT > "$H/gui.out" 2>&1 ) &
for _ in $(seq 1 30); do sleep 1; grep -q "MCP HTTP server listening" "$H/gui.out" 2>/dev/null && break; done
sleep 4
call '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"gui_train_smoke","version":"0"}}}' > "$H/call_init.json"
call "$(python3 -c 'import json,sys; print(json.dumps({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"scene_load_dataset","arguments":{"path":sys.argv[1],"images_folder":sys.argv[2],"output_path":sys.argv[3],"max_iterations":300,"strategy":sys.argv[4]}}}))' "$DS" "$IMG" "$H/out" "$STRAT")" > "$H/call_load.json"
sleep 10
call "$(py "import lichtfeld as lf
o=lf.optimization_params()
o.use_exposure_correction=False
o.set_strategy('$STRAT'); o.ppisp=bool($PP); o.use_bilateral_grid=bool($BG)
try:
    o.hdr=bool($HDR)
except Exception as e:
    print('no hdr property:', e)
o.undistort=True; o.iterations=300
lf.start_training()
print('started', o.strategy, o.ppisp, o.use_bilateral_grid)")" > "$H/call_start.json"
res=OK
for _ in $(seq 1 9); do sleep 5; if ls "$LOGS"/lichtfeld-studio-crash-*.log >/dev/null 2>&1; then res=CRASH; break; fi; done
last=$(grep -oE "Initializing trainer with [0-9]+ iterations|Failed to start training[^\n]{0,60}|Saving final project at iteration [0-9]+" "$H/gui.out" | tail -1)
pkill -f "[L]ichtFeld-Studio --mcp-port=$PORT" 2>/dev/null; sleep 2
echo "RESULT $(basename "$CK") images=$IMG strategy=$STRAT ppisp=$PP bilateral=$BG hdr=$HDR → $res  ($last)"
[ "$res" = OK ]
