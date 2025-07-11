# Robo-Game: deception-based path planning

This project implements a deceptive path planning algorithm on a mobile robotics platform. The platform consists of an overhead camera for observing the game, a Raspberry Pi for running the code, and one or more differential drive mobile robots. AprilTags are used with the camera for SLAM, replacing costly hardware such as LiDAR units and IMU's. The mobile robots consist of an ESP32 Feather board and two continuous servos, and communicate with the Raspberry Pi via BLE.

---

## ⚙️ Requirements

- Python 3.8 or higher
- Install required pip modules by running 'pip install -r requirements.txt"
   - OpenCV (cv2)
   - NumPy
   - pupil_apriltags
   - bleak
   - tripy
   - rdp
   - matplotlib
   - scipy
   - numba
   - networkx
- A wide-angle camera, with parameters saved as `fisheye_params.npz` if there is fisheye distortion
- AprilTags printed and placed in the physical arena

---

## 🧠 System Architecture

1. **AprilTag Detection**  
   Uses `AprilTag_sensor.py` to detect tags in real-time using a fisheye camera with distortion correction.

2. **Global Planner**  
   Uses Voronoi diagrams and deceptive path planning (from `DPP.py`) to compute paths that mislead observers. Implemented in `global_planner.py`.

3. **Local Planner**  
   Executes a Dynamic Window Approach (DWA) to follow the global path while dynamically avoiding obstacles. Found in `local_planner.py`.

4. **BLE Communication**  
   The `BLE.py` module sends velocity commands over Bluetooth to an ESP32 robot. Supports reconnection on drop.

5. **Execution Control**  
   The system is started through `main.py`, which coordinates the detector, planner, and BLE bridge. A GUI shows live detection, paths, and tag overlays.

---

## 🎮 Controls (in GUI window)

- Press **q** to quit
- Press **p** to run the global planner
- Press **t** to toggle tag annotations
- Press **s** to save the current frame (debug mode only)

---

## 🏁 Tag ID Mapping

- Ground Left: ID 0  
- Ground Right: ID 1  
- Real Goal: ID 2  
- Fake Goal: ID 3  
- Obstacle(s): ID 4  
- Robot(s): ID 5, 6, 7, ...  

These tag IDs must match the physical AprilTags placed in the arena.

---

## 🚀 How to Run

1. Place the AprilTags in the arena according to the ID mapping.
2. Ensure the camera is calibrated and `fisheye_params.npz` is available.
3. Power on the ESP32 robot and ensure it is advertising over BLE.
4. Run the program by executing this command in your terminal:

   python3 main.py

5. A window will open showing the live camera feed with tag overlays and planned paths.

---

## ✅ Tips for Success

- Use even, glare-free lighting for reliable tag detection.
- Hold AprilTags flat and clearly within view of the camera.
- Use tags from the `tag25h9` family, printed at high quality.
- Verify the camera calibration file is accurate for your camera setup.

---

## 📝 Notes

- The system is multithreaded: one thread for AprilTag detection, one for BLE communication, and the main thread for GUI.
- Path planning uses game-theoretic deception via PBNE (Perfect Bayesian Nash Equilibrium).
- BLE sends velocity updates every 200 milliseconds.
- The Voronoi diagram is dynamically built and optimized to respect arena boundaries and obstacles.
- If planning fails (e.g., missing tags or obstacles), the robot will stop or fallback safely.

---

## 📜 License

MIT License or similar — add your license text here if applicable.
