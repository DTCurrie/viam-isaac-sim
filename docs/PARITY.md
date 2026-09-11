# Parity ledger

Every method a real driver implements, the sim model implements with the same semantics. This
ledger is the record of that rule. One section per sim model. One row per method of the Viam
component API that model serves, plus one row per `DoCommand` verb either side implements.

Columns: `Method` is the API method or `DoCommand: <verb>`. `Real driver` is what the named real
module does, with the source file and line it was read from. `Sim` is what this module does, with
the file and line. `Gap` is empty when the semantics match, and otherwise says in one line what
differs and why it stays.

Method rows come from the viam-sdk 0.80.0 abstract method list for each component class, read via
`inspect.getmembers`. `get_geometries` and `close` are not in that list (both have a default
implementation in the SDK's `ComponentBase`), so they are not rows here even though the sim and
some real drivers override them.

## Arm

Real driver: `viam:universal-robots:*` (`viam-modules/universal-robots`, `src/viam/ur/module/`).
Reference shape for `MoveToPosition`: rdk's `fake` arm
(`viamrobotics/rdk`, `components/arm/fake/fake.go`).

SDK abstract methods: 7 (`get_end_position`, `get_joint_positions`, `get_kinematics`, `is_moving`,
`move_to_joint_positions`, `move_to_position`, `stop`).

| Method | Real driver | Sim | Gap |
| --- | --- | --- | --- |
| `get_end_position` | Returns `read_tcp_pose()`, the arm's last-reported TCP pose (`ur_arm.cpp:652`, `ur_arm_state.cpp:281`). | Returns the configured end-effector prim's pose in the arm base frame, converted from quaternion to an orientation vector (`arm.py:173`). | Not read: whether the real driver's `tcp_state` is base-frame or world-frame is set by the URCL telemetry layer, not read in this pass. |
| `get_joint_positions` | Reads joint positions from `current_state_` and converts radians to degrees (`ur_arm.cpp:436`). | Reads joint positions from the sim handle, converts to degrees, then snaps any value within `_JOINT_LIMIT_TOLERANCE_DEG` of an SVA limit onto that limit (`arm.py:397`). | Sim clamps reported values that drift micro-degrees past a declared joint limit back onto the limit. The real driver reports the raw value with no such clamp. |
| `get_kinematics` | Synthesizes an SVA JSON document from calibrated DH parameters fetched from the physical controller at configure time (`ur_arm.cpp:675`). | Fetches a static SVA or URDF kinematics file from `kinematics_url`, or a per-asset URL for known models, and caches it to disk (`arm.py:465`). | Sim serves a static, pre-published kinematics file. The real driver serves kinematics calibrated live from the connected controller. A sim arm with no `kinematics_url` and no known asset raises `NotImplementedError` (`arm.py:432`), where the real driver always has calibration data once connected. |
| `is_moving` | Delegates to `current_state_->is_moving()` (`ur_arm.cpp:658`). | Delegates to the sim handle's `is_moving()` (`arm.py:417`). | Not read: the real driver's `is_moving()` body inside `state_` was not traced past the header declaration. |
| `move_to_joint_positions` | Converts degrees to radians and calls `move_joint_space_` with a default `MoveOptions{}`, which plans a TOTG trajectory and blocks until the controller finishes it (`ur_arm.cpp:448`, `ur_arm.cpp:901`). No joint-limit check appears in this path. | Validates the joint count, clamps or raises `JointTargetOutOfLimitsError` (`INVALID_ARGUMENT`) against SVA-declared limits with a 0.01 deg tolerance, commands the move, then polls settle and raises `ArmMoveStalledError` (`ABORTED`) or `ArmMoveTimeoutError` (`DEADLINE_EXCEEDED`) (`arm.py:286`). | The sim raises three distinct, typed gRPC errors for out-of-limits, stall, and timeout. The real driver's move path throws generic `std::runtime_error` / `std::invalid_argument` for planner failures, and this file does no explicit joint-limit check at all (the firmware or trajectory planner enforces it). See the `MoveToPosition` bullet under "Rulings". |
| `move_to_position` | Forwards the target pose to `move_tool_space_`, which skips the move if already within 1 mm / ~0.06 deg of the target, otherwise hands the pose to the controller to solve and blocks until done (`ur_arm.cpp:668`, `ur_arm.cpp:844`). No IK is computed in this module. | Skips the move under the same 1 mm / 0.06 deg tolerance. Otherwise solves inverse kinematics against the served kinematics file from the arm's current joints (`ChainSVA.ik`/`ChainURDF.ik`, `kinematics.py`) and drives the joint path through `move_to_joint_positions`, so settle, stall and timeout behave as a joint move does (`move_to_position`, `arm.py`). | The real driver hands the pose to the controller to solve. The sim solves against the file it serves through `GetKinematics`, so `GetEndPosition`, the frame system and the motion service agree with the sim by construction, which the real driver has no equivalent guarantee for. Documented deviation: the arm API defines this method as a straight line in Cartesian space, and the sim drives a joint-space path between the same two endpoints. |
| `stop` | Calls the private `stop_(rlock)` helper after confirming the arm is configured (`ur_arm.cpp:723`). | Calls the sim handle's `stop()` (`arm.py:414`). | Not read: `stop_()`'s internal behavior was not traced. |
| `DoCommand: set_vel` | Sets `speed_degs_per_sec` joint velocity limits from a value in deg/s (`ur_arm.cpp:740`, `ur_arm.cpp:776`). | Not implemented. | real-only |
| `DoCommand: set_vel_degs_per_sec` | Same handler as `set_vel` (`ur_arm.cpp:742`, `ur_arm.cpp:776`). | Not implemented. | real-only |
| `DoCommand: set_acc` | Sets `acceleration_degs_per_sec2` joint acceleration limits (`ur_arm.cpp:741`, `ur_arm.cpp:780`). | Not implemented. | real-only |
| `DoCommand: set_accel_degs_per_sec2` | Same handler as `set_acc` (`ur_arm.cpp:743`, `ur_arm.cpp:780`). | Not implemented. | real-only |
| `DoCommand: get_tcp_forces_base` | Returns the last TCP force/torque reading in the robot base frame (`ur_arm.cpp:744`, `ur_arm.cpp:784`). | Not implemented. | real-only |
| `DoCommand: get_tcp_forces_tool` | Returns the last TCP force/torque reading rotated into the tool frame (`ur_arm.cpp:745`, `ur_arm.cpp:797`). | Not implemented. | real-only |
| `DoCommand: clear_pstop` | Clears an active protective stop, or errors if none is active or the arm is disconnected (`ur_arm.cpp:746`, `ur_arm.cpp:810`). | Not implemented. | real-only |
| `DoCommand: zero_ftsensor` | Zeroes the force-torque sensor bias (`ur_arm.cpp:747`, `ur_arm.cpp:813`). | Not implemented. | real-only |
| `DoCommand: is_controllable_state` | Reports whether the arm's current connection state accepts motion commands (`ur_arm.cpp:748`, `ur_arm.cpp:816`). | Not implemented. | real-only |
| `DoCommand: get_state_description` | Returns a human-readable description of the current connection state (`ur_arm.cpp:749`, `ur_arm.cpp:822`). | Not implemented. | real-only |
| `DoCommand: get_calibrated_dh_params` | Returns the calibrated DH parameters (a, d, alpha, theta) fetched from the controller (`ur_arm.cpp:750`, `ur_arm.cpp:828`). | Not implemented. | real-only |

The arm's sim-only introspection verbs now live on the world's `DoCommand`, keyed by the arm's
component name (`joint_state`, `dof_names` with an optional `all`, and `prim_pose`, formerly
`joint_state`, `dof_names`, `all_dof_names` and `prim_world_pose` on the arm), so the arm component
itself carries no verb a real UR driver could not answer. The former `get_joint_positions_radians`
verb was dropped, since `joint_state` carries positions.

## Gripper

Real driver: `viam:robotiq:2f-grippers` (`viam-modules/robotiq`, `robotiq/gripper.go`).

SDK abstract methods: 8 (`get_current_inputs`, `get_kinematics`, `go_to_inputs`, `grab`,
`is_holding_something`, `is_moving`, `open`, `stop`).

| Method | Real driver | Sim | Gap |
| --- | --- | --- | --- |
| `open` | Calls `SetPos` to the configured open limit, which writes the target over serial and polls the `POS` register until it matches or five consecutive reads come back unchanged, i.e. it blocks until reached or stalled (`robotiq_gripper.go:216`, `robotiq_gripper.go:181`). | Commands the sim handle's jaw target and blocks, polling `is_moving`, until it settles or the grab timeout passes, the same shape as `grab()` (`open`, `gripper.py`). | No gap found in this pass. |
| `grab` | Calls `SetPos` to the close limit. If the jaw fully closes, treats that as "closed on nothing" and returns `false` without checking further. If it stalled short of fully closed, reads the `OBJ` register and returns whether it reports an object (`robotiq_gripper.go:234`). | Closes the jaw (non-blocking command), polls `is_moving` until it settles or `grab_timeout_sec` elapses, then polls `is_holding()` or a closed-jaw check, and returns `is_holding()` (`gripper.py:197`). | Same close-then-check-holding shape on both sides, over different substrates (a firmware register vs. a physics stall predicate). No behavioral gap found in this pass. |
| `is_holding_something` | Reads the `OBJ` register and returns `true` iff it equals `"OBJ 2"`, with no metadata (`robotiq_gripper.go:319`). | Computes holding from the sim handle and returns it with a `meta` payload of jaw, open, and closed angles in degrees plus the normalized `input` value (`gripper.py:216`). | Sim always returns metadata the real driver never returns. |
| `is_moving` | Returns whether the SDK operation manager currently has an operation running for this resource, not a direct read of physical motion (`robotiq_gripper.go:294`). | Returns the sim handle's physical jaw-velocity-based `is_moving()` (`gripper.py:236`, `sim_manager.py:3659`). | The real driver's signal is "a client-issued operation is in flight." The sim's is "the jaw is physically still travelling." They usually agree because `Open`/`Close`/`Grab` block for the real driver, but the two are not the same predicate. |
| `get_kinematics` | Returns the error `"Kinematics not supported"` (`robotiq_gripper.go:304`). | Synthesizes a one-link, zero-joint SVA document describing the gripper's bounding box, centered on the tool axis relative to the TCP (`gripper.py:239`). | The real driver has no gripper kinematics. The sim invents a kinematics document so the frame system can place the gripper's geometry. |
| `get_current_inputs` | Always returns an empty slice, this method is an unimplemented stub (`robotiq_gripper.go:309`). | Returns the live normalized jaw position in `[0, 1]` (`gripper.py:248`). | Real driver never reports a current input. Sim always does. |
| `go_to_inputs` | Returns the error `"GoToInputs not supported"` (`robotiq_gripper.go:314`). | Validates exactly one value in `[0, 1]`, raising `INVALID_ARGUMENT` otherwise, commands the corresponding jaw angle, and polls until it settles or `grab_timeout_sec` elapses (`gripper.py:256`). | The real driver does not support `GoToInputs` at all. The sim fully implements it. |
| `stop` | Sets `GTO` (go-to) to `0`, halting motion (`robotiq_gripper.go:284`). | Calls the sim handle's `stop()`, which holds the current jaw angle (`gripper.py:233`, `sim_manager.py:3655`). | No gap found in this pass. |
| `DoCommand: get` | Reads the raw `POS` register (0 = open, 255 = closed) and returns it as `{"pos": <int>}` (`robotiq_gripper.go:338`). | Not implemented. | real-only |
| `DoCommand: set` | Writes a raw position value and returns `{"position": <int>}` once commanded (`robotiq_gripper.go:349`). | Not implemented. | real-only |

The gripper's sim-only introspection verbs (`dof_names`, `jaw_deg`, `tcp_pose`) now live on the
world's `DoCommand`, keyed by the gripper's component name, so the gripper component itself
carries no verb a real robotiq gripper could not answer.

## Camera

Real driver: `viam:camera:realsense` (`viam-modules/viam-camera-realsense`, `src/module/realsense.hpp`).

SDK abstract methods: 3 (`get_images`, `get_point_cloud`, `get_properties`).

| Method | Real driver | Sim | Gap |
| --- | --- | --- | --- |
| `get_images` | Grabs the latest synchronized frameset from the librealsense pipeline, optionally aligns depth to color, encodes the sensors named in `filter_source_names` (or all configured sensors), and raises if the camera is in DFU/recovery mode or no frameset has arrived yet (`realsense.hpp:551`). | Grabs the latest rendered frame from the sim handle, encodes color to PNG or JPEG and, when `depth` is enabled, encodes depth to `VIAM_RAW_DEPTH`, filtered by `filter_source_names` (`camera.py:146`). | The real driver has hardware-fault paths (DFU mode, stale USB connection) with no sim equivalent, since the sim always has a rendered frame once the camera is attached. |
| `get_point_cloud` | Requires both a fresh color and depth frame (raises on staleness via `throwIfTooOld`), deprojects through the device's point-cloud filter, and raises if the encoded data exceeds the gRPC message size limit (`realsense.hpp:719`). | Raises `MethodNotImplementedError` immediately if the camera's `depth` attribute is false. Otherwise deprojects the rendered depth frame using cached intrinsics into a PCD, and raises `ValueError` if the encoded cloud exceeds `MAX_POINT_CLOUD_BYTES` (32 MiB), the same guard the RealSense module applies (`get_point_cloud`, `camera.py`). | No gap found in this pass. |
| `get_properties` | Reports `supports_pcd = true` unconditionally once a device is streaming, regardless of whether depth is part of the active profile, and fills intrinsics and distortion from the live color or depth stream (`realsense.hpp:800`). | Reports `supports_pcd = handle.depth_enabled`, sets `mime_types` to the color mime plus depth and PCD mimes when depth is on, and computes `frame_rate` from the configured `frequency` or the render step (`camera.py:199`). | The real driver's `supports_pcd` does not depend on whether depth is configured. The sim's does. Not read: the C++ SDK's `Camera::properties` struct definition was not fetched, so whether the real driver populates `mime_types` or `frame_rate` through a different path is not read. |
| `DoCommand: update_firmware` | Applies a firmware update image to the connected device (`realsense.hpp:492`). | Not implemented. | real-only |
| `DoCommand: set_laser_power` | Sets the depth sensor's laser power option (`realsense.hpp:399`). | Not implemented. | real-only |
| `DoCommand: set_depth_emitter` | Enables or disables the depth emitter (`realsense.hpp:406`). | Not implemented. | real-only |
| `DoCommand: set_depth_visual_preset` | Sets a depth visual preset (`realsense.hpp:413`). | Not implemented. | real-only |
| `DoCommand: set_depth_exposure_us` | Sets depth sensor exposure in microseconds (`realsense.hpp:427`). | Not implemented. | real-only |
| `DoCommand: set_depth_auto_exposure` | Enables or disables depth auto-exposure (`realsense.hpp:434`). | Not implemented. | real-only |
| `DoCommand: set_depth_gain` | Sets the depth sensor's gain option (`realsense.hpp:442`). | Not implemented. | real-only |
| `DoCommand: get_depth_options` | Returns the depth sensor's current option values (`realsense.hpp:448`). | Not implemented. | real-only |
| `DoCommand: sample_color` | Not implemented. | Returns the mean RGB and hex color over a pixel region of the most recent frame (`camera.py:230`, `camera.py:236`). | sim-only |

## Base

Real driver: `rdk:builtin:wheeled` (`viamrobotics/rdk` `components/base/wheeled/wheeled_base.go`).

SDK abstract methods: 7 (`get_properties`, `is_moving`, `move_straight`, `set_power`,
`set_velocity`, `spin`, `stop`).

Neither side implements any `DoCommand` verb for the base, so this section has no `DoCommand` rows.

| Method | Real driver | Sim | Gap |
| --- | --- | --- | --- |
| `move_straight` | Converts distance and speed to per-wheel RPM and rotation count, then runs `motor.GoFor` on both sides in parallel, which blocks until the commanded rotations complete. A zero speed or zero distance calls `Stop` and returns with no error (`wheeled_base.go:263`). | A velocity near `NEAR_ZERO_THRESHOLD` calls `stop()` and returns with no error. Otherwise sets wheel velocity, sleeps for the computed duration, then stops, also blocking for the whole move (`move_straight`, `base.py`). | No gap found in this pass. |
| `spin` | Blocks via `runAllGoFor` until the spin's rotations complete. Raises an error for an angle within `0.0001` deg of zero. A near-zero speed calls `Stop` and returns with no error (`wheeled_base.go:238`). | An angle near `NEAR_ZERO_THRESHOLD` raises `ValueError`. A velocity near `NEAR_ZERO_THRESHOLD` calls `stop()` and returns with no error. Otherwise sleeps for the computed duration then stops, blocking for the whole move (`spin`, `base.py`). | No gap found in this pass. |
| `set_power` | A zero linear/angular vector calls `Stop`. Otherwise computes differential-drive motor powers and issues `motor.SetPower` in parallel, which does not block for the motors to reach that power (`wheeled_base.go:431`). | Clamps components to `[-1, 1]`, scales by `max_linear_mps`/`max_angular_rps`, and issues one non-blocking `set_velocity` call to the sim handle (`base.py:110`). | No gap found in this pass. Both are fire-and-forget commands, and a zero vector produces the same net stop either way. |
| `set_velocity` | A zero linear/angular vector calls `Stop`. Otherwise converts mm/s and deg/s to per-wheel RPM and issues non-blocking `motor.SetRPM` calls in parallel (`wheeled_base.go:409`). | Converts mm/s to m/s and deg/s to rad/s and issues one non-blocking `set_velocity` call (`base.py:115`). | No gap found in this pass. |
| `stop` | Stops both left and right motor groups in parallel (`wheeled_base.go:515`). | Calls the sim handle's `stop()` (`base.py:119`). | No gap found in this pass. |
| `is_moving` | Returns `true` if any underlying motor's `IsPowered` reports powered (`wheeled_base.go:539`). | Delegates to the sim handle's physical `is_moving()` (`base.py:122`). | Not read: the sim handle's `is_moving()` body was not traced beyond `base.py`. |
| `get_properties` | Returns `TurningRadiusMeters: 0`, and width and wheel circumference converted from the configured millimeter attributes (`wheeled_base.go:557`). | Returns `turning_radius_meters=0.0`, and width and wheel circumference computed from the sim handle's wheel base and wheel radius (`base.py:130`). | No gap found in this pass. |

## Documented deviations

Where the sim departs from a real driver on purpose, and why.

- Sim-only `DoCommand` verbs live on the world's `DoCommand`, keyed by component name, never on
  the component that stands in for hardware. For the arm: `joint_state`, `dof_names` with an
  optional `all`, and `prim_pose`. For the gripper: `dof_names`, `jaw_deg` and `tcp_pose`. The
  camera's `sample_color` stays on the camera, since a real camera module could carry it.
- `MoveOptions` acceleration fields and `max_tcp_speed` are logged and not honored. The UR module
  honors them through its trajectory planner, which the sim does not have.
- The sim arm takes a config-level `max_vel_degs_per_sec`, carried from a real UR's
  `speed_degs_per_sec`, as the velocity cap for a move that carries no `MoveOptions` cap of its
  own. `acceleration_degs_per_sec2` has no sim counterpart: the sim arm honors no config-level
  acceleration limit.
- `MoveToPosition` solves IK against the served kinematics and drives a joint path where the API
  defines a Cartesian straight line. An unreachable target raises `INVALID_ARGUMENT`, and a
  solution outside joint limits raises `JointTargetOutOfLimitsError`.
- Gripper `open()` blocks until the jaw settles or stalls, like `grab()` and like the real robotiq
  driver. `is_holding_something` metadata, `get_current_inputs`, `go_to_inputs` and the sim's
  gripper kinematics are the gaps the table above names.
- Base `move_straight` and `spin` follow the wheeled base's zero-velocity and near-zero-angle
  semantics. Camera `get_point_cloud` carries the real driver's message-size guard.
  `get_properties.supports_pcd` follows the depth config.
- `get_kinematics` serves a static file. The sim has no controller to calibrate against, so IK and
  collision checks reproduce the nominal model, not a particular arm's calibration.
