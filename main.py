import asyncio
import threading
import struct
from bleak import BleakClient
from local_planner import LocalPlanner
from AprilTag_sensor import AprilTagDetector
import BLE

# === USER PARAMETERS ===
robot_MAC_address = "14:2b:2f:cc:ed:72"
# =======================

if __name__ == "__main__":

    # ── ask user how to configure the observer ──────────────────────────
    resp = input("Use CPU observer? (y/n): ").strip().lower()
    AprilTag_sensor.USE_CPU_OBSERVER = resp.startswith('y')

    # get priors p1,p2 that sum to 1.0
    while True:
        pri = input("Enter priors for goal1 and goal2 (e.g. 0.6,0.4): ").strip()
        try:
            p1, p2 = map(float, pri.split(','))
            if abs((p1+p2) - 1.0) > 1e-6:
                print("Priors must sum to 1. Try again.")
                continue
            AprilTag_sensor.PRIOR_BELIEF = (p1, p2)
            break
        except:
            print("Invalid format. Please enter two numbers separated by a comma.")
    # ────────────────────────────────────────────────────────────────────

    
    # Create your detector (threaded)
    tag_detector = AprilTagDetector()  # Your existing threaded class
    
    # Create local planner
    lp = LocalPlanner(tag_detector)
    
    # Create BLE bridge
    ble_bridge = BLE.BLEBridge_FAMP(
        lp, 
        ble_address=robot_MAC_address
    )
    
    # Start BLE in a separate thread
    ble_thread = threading.Thread(
        target=BLE.start_asyncio_ble,
        args=(ble_bridge,),
        daemon=True
    )
    ble_thread.start()
    

    try: # runs in main thread
        tag_detector.run_gui()  # detects global plan key ('p')
        # after this line will run after detector is closed
    finally:
        # Cleanup
        tag_detector.close()
        ble_bridge.running = False
        ble_thread.join()
