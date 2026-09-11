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
