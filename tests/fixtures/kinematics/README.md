# Kinematics fixtures

Real SVA kinematics files, read by `tests/test_kinematics_chain.py` and
`tests/test_arm_move_to_position.py` so the IK tests run against the same
bytes `GetKinematics` serves.

`ur5e.json` and `ur20.json` come from the UR driver module, since those are
assets this module spawns. `xarm7.json` comes from rdk's own kinematics
directory, and it is the 7-DOF chain the two UR files cannot cover.

Downloaded 2026-09-08. To refresh them, run this from this directory:

```sh
UR=https://raw.githubusercontent.com/viam-modules/universal-robots/main/src/kinematics
RDK=https://raw.githubusercontent.com/viamrobotics/rdk/main/components/arm/kinematics
curl -sSfO "$UR/ur5e.json"
curl -sSfO "$UR/ur20.json"
curl -sSfO "$RDK/xarm7.json"
```
