# Kinematics files

The SVA kinematics for the known UR assets `asset_catalog.py` references
(`ur3e`, `ur5e`, `ur7e`, `ur20`), packaged inside the module archive so
`GetKinematics` and `MoveToPosition` never depend on a network fetch for these
assets. `ur7e` spawns the `ur5e` mesh as a geometry stand-in (Isaac Sim 5.0
ships no distinct `ur7e` USD) but uses its own kinematics file, the real
`ur7e`'s.

Downloaded 2026-09-10 from the UR driver module. To refresh them, run this
from this directory:

```sh
UR=https://raw.githubusercontent.com/viam-modules/universal-robots/main/src/kinematics
curl -sSfO "$UR/ur3e.json"
curl -sSfO "$UR/ur5e.json"
curl -sSfO "$UR/ur7e.json"
curl -sSfO "$UR/ur20.json"
```

## The Robotiq EPick

`epick_model.json` is the real driver's own kinematics document, copied byte for byte from
`viam-labs/robotiq-epick` (`epick/epick_model.json`, commit `81aa5c75`, read 2026-09-22). The
`viam:isaac-sim-devin:vacuum` model serves it verbatim from `GetKinematics`, so the motion service
plans against exactly the collision boxes it sees on the real machine: six boxes in the gripper
frame (z = 0 at the TCP, the flange at z = -196 mm), with no collider in the last 26 mm before the
TCP so a grab approach is never refused. The same repo's `epick/geometry.go` carries the measured
constants those boxes come from, vendored into `asset_catalog.py` as `EPICK`. To refresh:

```sh
curl -sSfO https://raw.githubusercontent.com/viam-labs/robotiq-epick/main/epick/epick_model.json
```
