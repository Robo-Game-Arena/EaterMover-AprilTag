import cv2
import numpy as np
import time
from pupil_apriltags import Detector
import threading
import os
import atexit
import signal

from global_planner import Planner
from observer import CPUObserver, ManualObserver

# ── OBSERVER CONFIG ─────────────────────────────────────────────────────────
USE_CPU_OBSERVER = True              # True = automatic CPU, False = manual key control
PRIOR_BELIEF    = (0.6, 0.4)         # your prior over goal-tag IDs 2 vs 3
# ─────────────────────────────────────────────────────────────────────────────

# === USER PARAMETERS ===
TAG_SIZE_M     = 0.04              # side length of robot AprilTag in meters
CALIB_FILE     = "fisheye_params.npz"
SRC            = "/dev/video0"
RESOLUTION     = (2560, 1440)
# =======================

class AprilTagDetector:
    def __init__(self):
        # Register cleanup handlers
        atexit.register(self.close)
        signal.signal(signal.SIGINT, lambda sig, frame: self.close())
        
        self.init_camera() # initialize camera

        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            raise Exception("Failed to open camera")
        
        # Get actual camera parameters
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"Camera initialized with resolution: {width}x{height}")
        
        # Approximate camera parameters (replace with actual calibration if available)
        # Note: These are critical for detection accuracy
        focal_length = width
        self.camera_params = [focal_length, focal_length, width/2, height/2]  # [fx, fy, cx, cy]
        
        # Store detector parameters (since we can't read them back from the detector)
        self.quad_decimate = 2.0  # Reduced for better performance
        self.quad_sigma = 0.0
        self.decode_sharpening = 0.25
        
        # Initialize detector
        self.detector = Detector(
            families='tag25h9',  # Using tag25h9 family
            nthreads=2,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=1,
            decode_sharpening=self.decode_sharpening,
            debug=0
        )
        
        # Tag data storage
        self.tags = []
        
        # Debug mode - disabled by default for better performance
        self.debug_mode = False
        self.frame_count = 0
        self.last_detection_time = time.time()
        
        # Create debug directory only if debug mode is enabled
        self.debug_dir = "debug_frames"
        if self.debug_mode and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # Thread control
        self.running = True
        self.lock = threading.Lock()
        
        # Start detection thread
        self.detect_thread = threading.Thread(target=self.detection_loop)
        self.detect_thread.daemon = True
        self.detect_thread.start()
        self.show_tags = True

        # ── OBSERVER PLACEHOLDERS ───────────────────────────────────────────
        self.observer    = None        # will init on first frame
        self.cost1       = 0.0
        self.cost2       = 0.0
        self._obs_ready  = False
        self._cost_active = False
        # ─────────────────────────────────────────────────────────────────────

        # added code for planning
        self.frame_width = width
        self.frame_height = height
        self.path = None
        self.curr_planner = None
        self.has_new_plan = False
        self.dwa_path = None
        self.curr_goal = None

        # ── FRAME‐GATING FOR OBSERVER ───────────────────────────────────────
        # only update observer once per new frame
        self._last_update_frame = -1
        self._last_sigma1       = 0.5
        self._last_sigma2       = 0.5
        self._last_cost1        = 0.0
        self._last_cost2        = 0.0
        # ────
    
    def init_camera(self):
        # Initialize camera

        # 1) Load fisheye calibration
        calib = np.load(CALIB_FILE)
        K, D  = calib["K"], calib["D"]

        # 2) Open camera & set parameters
        cap = cv2.VideoCapture(SRC)
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE,      100)
        cap.set(cv2.CAP_PROP_GAIN,          4)
        cap.set(cv2.CAP_PROP_AUTO_WB,       1)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE,4500)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # fisheye rectification maps
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )

        self.cap = cap
        return


    def detection_loop(self):
        """Run continuous detection in a separate thread"""
        while self.running:
            start_time = time.time()
            
            # flush stale frames
            self.cap.grab()
            self.cap.grab()
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image")
                time.sleep(0.05)
                continue
            
            # undistort
            img = cv2.remap(
                frame, self.map2, self.map1,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

            # Convert to grayscale (only use one method for speed)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            self.frame_count += 1
            
            # Detect AprilTags
            detected_tags = self.detector.detect(
                gray, 
                estimate_tag_pose=True, 
                camera_params=self.camera_params,
                tag_size=TAG_SIZE_M  # Meters (adjust based on your tag size)
            )
            
            # Update tags with the lock to prevent race conditions
            with self.lock:
                self.frame = img.copy()
                self.tags = detected_tags
            
            # Sleep to maintain approximately 0.1s intervals
            elapsed = time.time() - start_time
            sleep_time = max(0, 1/40 - elapsed)
            time.sleep(sleep_time)
    
    def print_tag_info(self):
        """Print tag information to the terminal (reduced frequency for performance)"""
        # Calculate time since last detection
        time_since_detection = time.time() - self.last_detection_time
        
        if not self.tags:
            # Print less frequently when no tags detected
            print(f"Time: {time.strftime('%H:%M:%S')} - No tags detected (%.1f seconds since last detection)" % time_since_detection)
            return
        
        print(f"\nTime: {time.strftime('%H:%M:%S')} - Detected {len(self.tags)} tags:")
        
        for i, tag in enumerate(self.tags):
            print(f"  Tag {i+1} (ID: {tag.tag_id}):")
            print(f"    Center: ({tag.center[0]:.1f}, {tag.center[1]:.1f})")
            print(f"    Decision margin: {tag.decision_margin:.4f}")  # How confident the detector is
        print()
    
    def run_gui(self):
        """Run the GUI visualization loop"""
        cv2.namedWindow('AprilTag Detector', cv2.WINDOW_NORMAL)

        # Precompute some GUI layout constants
        BAR_HEIGHT    = 40           # height of allocation slider
        BOX_WIDTH     = 120          # width of each cost box
        BOX_HEIGHT    = 60           # height of each cost box
        BOX_Y_OFFSET  = BAR_HEIGHT + 10
        FONT          = cv2.FONT_HERSHEY_SIMPLEX
        SLIDER_COLORS = [(0,0,255), (255,0,0)]  # BGR: red, blue

        while self.running:
            # 1) Grab frame & detections
            with self.lock:
                if not hasattr(self, 'frame'):
                    time.sleep(0.01)
                    continue
                frame = self.frame.copy()
                tags  = self.tags.copy() if self.tags else []

            # 2) Dynamic init of observer (as before)
            if not self._obs_ready:
                goals = {t.tag_id: t.center for t in tags if t.tag_id in (2,3)}
                if len(goals)==2:
                    obs_polys = [t.corners.tolist() for t in tags if t.tag_id==4]
                    goal_positions = [np.array(goals[2]), np.array(goals[3])]
                    if USE_CPU_OBSERVER:
                        self.observer = CPUObserver(goal_positions,
                                                    PRIOR_BELIEF,
                                                    obstacle_polygons=obs_polys)
                    else:
                        self.observer = ManualObserver(goal_positions,
                                                       PRIOR_BELIEF)
                    self._obs_ready = True
                else:
                    cv2.imshow('AprilTag Detector', frame)
                    if cv2.waitKey(1)&0xFF==ord('q'):
                        self.running=False
                    continue

            # 3) Draw the arena/path/robot (as before)
            if self.show_tags:
                self.draw_tags(frame, tags)
            if self.curr_planner is not None:
                self.draw_arena(frame)

            # 5) Observer‐update gated on new frame (as we added earlier)
            if self.frame_count != self._last_update_frame:
                self._last_update_frame = self.frame_count
                robot_tags = [t for t in tags if t.tag_id >= 5]
                if robot_tags:
                    cp = np.array(robot_tags[0].center)

                    sigma1, sigma2, dt = self.observer.update(current_pos, time.time())
                    # if the observer just hit the goal, deactivate cost tracking
                    if self.observer._finished:
                        self._cost_active = False

                    # otherwise accumulate cost while active
                    if self._cost_active:
                        self.cost1 += sigma1 * dt
                        self.cost2 += sigma2 * dt
                    self._last_sigma1 = sigma1
                    self._last_sigma2 = sigma2
                    self._last_cost1  = self.cost1
                    self._last_cost2  = self.cost2
            else:
                σ1, σ2 = self._last_sigma1, self._last_sigma2
                self.cost1, self.cost2 = self._last_cost1, self._last_cost2

            # 6) Draw allocation slider at top
            w = frame.shape[1]
            w1 = int(w * self._last_sigma1)
            # red portion
            cv2.rectangle(frame, (0,0), (w1, BAR_HEIGHT), SLIDER_COLORS[0], -1)
            # blue portion
            cv2.rectangle(frame, (w1,0), (w, BAR_HEIGHT), SLIDER_COLORS[1], -1)
            # text labels
            pct1 = f"{self._last_sigma1*100:.1f}%"
            pct2 = f"{self._last_sigma2*100:.1f}%"
            for idx, (pct, start_x) in enumerate(((pct1, 0), (pct2, w1))):
                (tw, th), _ = cv2.getTextSize(pct, FONT, 0.8, 2)
                txt_x = start_x + ((w1 if idx==0 else (w - w1)) - tw)//2 + (0 if idx==0 else w1)
                txt_y = BAR_HEIGHT//2 + th//2
                cv2.putText(frame, pct, (txt_x, txt_y), FONT, 0.8, (0,0,0), 2)

            # 7) Draw cost boxes
            # red box for goal1
            x1 = 50
            y1 = BOX_Y_OFFSET
            cv2.rectangle(frame, (x1,y1), (x1+BOX_WIDTH,y1+BOX_HEIGHT),
                          SLIDER_COLORS[0], -1)
            # blue box for goal2
            x2 = frame.shape[1] - BOX_WIDTH - 50
            y2 = BOX_Y_OFFSET
            cv2.rectangle(frame, (x2,y2), (x2+BOX_WIDTH,y2+BOX_HEIGHT),
                          SLIDER_COLORS[1], -1)
            # overlay cost text
            cost1_txt = f"{self._last_cost1:.1f}"
            cost2_txt = f"{self._last_cost2:.1f}"
            for txt, (bx, by) in ((cost1_txt,(x1,y1)), (cost2_txt,(x2,y2))):
                (tw, th), _ = cv2.getTextSize(txt, FONT, 1.0, 3)
                txt_x = bx + (BOX_WIDTH - tw)//2
                txt_y = by + (BOX_HEIGHT + th)//2
                cv2.putText(frame, txt, (txt_x, txt_y), FONT, 1.0, (0,0,0), 3)

            # 8) Final show + key handling (unchanged)
            cv2.imshow('AprilTag Detector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('p'):
                self.run_planner()
                self.adjustDrawings()
            elif key == ord('t'):
                self.show_tags = not self.show_tags
            # manual mode
            if not USE_CPU_OBSERVER and self._obs_ready:
                if key==ord('a'): self.observer.adjust(+0.05)
                elif key==ord('d'): self.observer.adjust(-0.05)

    
    def adjustDrawings(self):
        if self.curr_planner == None:
            return
        p = self.curr_planner
        # adjust path
        path = self.path # dont need to copy because only adjusted values will be used
        path[:, 1] = self.frame_height - (path[:, 1]*p.arena_height + p.oy)
        path[:, 0] = path[:, 0]*p.arena_width + p.ox
        self.raw_path = path.astype(int)

        # adjust boundaries
        self.raw_bounds = np.array([[0, 0], [p.arena_width, 0], [p.arena_width, p.arena_height], 
                                    [0, p.arena_height]]) + [self.curr_planner.ox, self.curr_planner.oy]
        self.raw_bounds[:, 1] = self.curr_planner.frame_height - self.raw_bounds[:, 1]
        self.raw_bounds = self.raw_bounds.astype(int)

        # adjust obstacles
        self.raw_obstacles = np.array(self.curr_planner.obstacle_corners).astype(int)
        self.obstacle_bounds = [[ob[0][0], ob[1][0], ob[3][1], ob[1][1]] for ob in self.raw_obstacles] # list of [xmin xmax ymin ymax]
        return

        

    def draw_arena(self, frame):
        cv2.polylines(frame, [self.raw_bounds], True, (0, 255, 0), 2)
        cv2.polylines(frame, [self.raw_path], False, (0, 255, 0), 2)
        cv2.circle(frame, self.curr_goal, 5, (255, 255, 0), -1)
        for obstacle in self.raw_obstacles:
            cv2.polylines(frame, [obstacle], True, (0, 255, 0), 2)
        if self.dwa_path is not None:
            cv2.polylines(frame, [self.dwa_path.astype(int)], False, (255, 100, 100), 4)
        if len(self.points_traveled) > 1 :
            cv2.polylines(frame, np.int32([self.points_traveled]), False, (0, 0, 255), 3)

    def draw_tags(self, frame, tags):
        """Draw detected tags on the frame"""
        for tag in tags:
            # choose BGR color
            if   tag.tag_id == 2: clr = (0, 0, 255)   # red for goal1
            elif tag.tag_id == 3: clr = (255, 0, 0)   # blue for goal2
            else:                 clr = (0, 255, 0)   # green for everything else

            corners = tag.corners.astype(int)
            cv2.polylines(frame, [corners], True, clr, 2)
            
            # Draw tag center
            center = tuple(int(x) for x in tag.center)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # Draw tag ID
            text_pos = (int(tag.center[0]), int(tag.center[1]) - 10)
            cv2.putText(frame, f"ID: {tag.tag_id}", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw coordinate information
            info_text = f"({center[0]}, {center[1]})"
            info_pos = (int(tag.center[0]), int(tag.center[1]) + 20)
            cv2.putText(frame, info_text, info_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw confidence score
            conf_text = f"Conf: {tag.decision_margin:.2f}"
            conf_pos = (int(tag.center[0]), int(tag.center[1]) + 40)
            cv2.putText(frame, conf_text, conf_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def run_planner(self):
        self.curr_planner = None
        self.curr_planner = Planner(self.tags, self.frame_width, self.frame_height)
        if self.curr_planner.deceptivepath is None:
            print('Planning failed')
            self.curr_planner = None
            return
        self.path = self.curr_planner.deceptivepath
        self.has_new_plan = True
        # NOW START COUNTING COSTS
        self._cost_active = True
        self.cost1 = 0.0
        self.cost2 = 0.0
        # also reset the observer’s history so timing/progress restarts cleanly
        if self.observer:
            self.observer.prev_pos  = None
            self.observer.prev_dist = None
            # last_time will be set on the next update call
               
        

    def close(self):
        """Clean up resources"""
        print("Releasing resources...")
        self.running = False
        
        # Wait for thread to finish
        if hasattr(self, 'detect_thread') and self.detect_thread.is_alive():
            try:
                self.detect_thread.join(timeout=1.0)  # Wait up to 1 second
            except:
                pass
        
        # Release the camera
        if hasattr(self, 'cap'):
            try:
                self.cap.release()
                print("Camera released successfully")
            except Exception as e:
                print(f"Error releasing camera: {e}")
        
        # Destroy any remaining windows
        try:
            cv2.destroyAllWindows()
            # Sometimes destroyAllWindows() doesn't work on first try
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass


def main():
    print("AprilTag Detector - Optimized Version")
    print("------------------------------------")
    print("Controls:")
    print("- Press 'q' in the display window to exit")
    print("- Camera will show detected tags with green outlines and red centers")
    print("- Tag IDs, coordinates, and confidence scores are displayed")
    print("\nFor better detection:")
    print("1. Ensure good lighting without glare")
    print("2. Hold tags 20-60cm from camera")
    print("3. Keep tags flat and clearly visible")
    
    detector = None
    try:
        # Create and run detector
        detector = AprilTagDetector()
        
        # Run the GUI in the main thread
        detector.run_gui()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Make sure to release the camera even if an error occurs
        if detector:
            detector.close()
                
        # Force destroy all windows
        cv2.destroyAllWindows()
        
        print("Detector stopped")
        print("\nIf tags aren't being detected:")
        print("1. Check lighting conditions (avoid glare)")
        print("2. Ensure you're using tag25h9 family tags")
        print("3. Try different distances from the camera")
        print("4. Make sure tags are clearly printed with good contrast")

if __name__ == "__main__":
    main()
