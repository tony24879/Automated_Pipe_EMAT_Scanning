Lite6 mesh backend input folder

Put per-link STL files in this folder with the following names:
- base.stl
- link1.stl
- link2.stl
- link3.stl
- link4.stl
- link5.stl
- link6.stl

Current behavior:
- Mesh backend can run without STL files (it will still render joint line/spheres, TCP, target, cylinder, and trace).
- If STL files exist, they will be loaded and displayed.
- Dynamic per-link mesh articulation is the next step and requires per-link frame alignment metadata.

Recommended mesh conventions:
- Units: millimeters (or set --robot-mesh-scale accordingly).
- Coordinate system: right-handed.
- Link-local origin: at each joint frame if possible.
- Zero pose: all joints at controller zero angles.

Run examples:
- Auto backend (mesh first, fallback to matplotlib):
  python -m scans.cylindrical_scan_horizontal --calibration-file cylinder_calibration_horizontal.json --view-3d-backend auto

- Force mesh backend:
  python -m scans.cylindrical_scan_horizontal --calibration-file cylinder_calibration_horizontal.json --view-3d-backend mesh --robot-mesh-dir 3Dview/meshes/lite6

- If meshes are in meters, convert to mm with scale 1000:
  python -m scans.cylindrical_scan_horizontal --calibration-file cylinder_calibration_horizontal.json --view-3d-backend mesh --robot-mesh-scale 1000
