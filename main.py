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
    
    # Create your detector (threaded)
    tag_detector = AprilTagDetector()  # Your existing threaded class
    
    # Create local planner
    lp = LocalPlanner(tag_detector)
    
    # Create BLE bridge
    ble_bridge = BLE.BLEBridge(
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
