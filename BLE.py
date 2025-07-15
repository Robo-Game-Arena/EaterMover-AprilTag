import asyncio
import threading
import struct
import numpy as np
from bleak import BleakClient
from bleak.exc import BleakDeviceNotFoundError, BleakError # Import specific exceptions
import logging # Import logging for better error messages
import time # For delays

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from local_planner import LocalPlanner

# === USER PARAMETERS ===
ESP32_SERVICE_UUID = "0000S001-0000-1000-8000-00805f9b34fb" # find using phone app: nRF Connect 
ESP32_CHAR_UUID    = "0000C001-0000-1000-8000-00805f9b34fb" # find using phone app: nRF Connect 
SEND_INTERVAL      = 0.2 # interval between updating velocities in seconds
# =======================

class BLEBridge:
    def __init__(self, planner: LocalPlanner, ble_address): 
        self.planner = planner
        self.ble_address = ble_address
        self.ble_client = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5 # Max attempts for initial connection
        self.initial_reconnect_delay = 2 # seconds
        self.send_interval = SEND_INTERVAL

    async def connect_ble(self):
        """Attempts to connect to BLE device. Returns True on success, False on failure."""
        if self.ble_client and self.ble_client.is_connected:
            logging.info("Already connected to BLE.")
            return True

        # Ensure a new client is created for each connection attempt
        self.ble_client = BleakClient(self.ble_address)

        try:
            # BleakClient.connect() internally scans for a default timeout (e.g., 20s).
            # If you want a shorter *scan* timeout before connecting, you'd use BleakScanner first.
            # But for simple retry, just catching the error here is sufficient.
            await self.ble_client.connect()
            logging.info(f"Successfully connected to BLE device: {self.ble_address}")
            self.reconnect_attempts = 0 # Reset attempts on successful connection
            return True
        except BleakDeviceNotFoundError:
            logging.warning(f"Device with address {self.ble_address} not found. Retrying...")
            return False
        except BleakError as e: # Catch other Bleak-related errors
            logging.error(f"BLE connection error: {e}")
            return False
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"An unexpected error occurred during BLE connection: {e}")
            return False

    async def ensure_connected(self):
        """Continuously tries to connect until successful or max attempts reached."""
        while self.running:
            if await self.connect_ble():
                return True # Successfully connected
            
            self.reconnect_attempts += 1
            if self.reconnect_attempts > self.max_reconnect_attempts:
                logging.error(f"Maximum initial connection attempts ({self.max_reconnect_attempts}) reached. Aborting.")
                return False # Failed to connect after max attempts

            delay = self.initial_reconnect_delay * (1.5 ** (self.reconnect_attempts - 1)) # Exponential backoff
            delay = min(delay, 30) # Cap the delay
            logging.info(f"Retrying connection in {delay:.1f} seconds (Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})...")
            await asyncio.sleep(delay)
        return False # Not running anymore

    async def send_velocities(self):
        """take velocity command from local planner and send via BLE"""
        while self.running:
            if not self.ble_client or not self.ble_client.is_connected:
                logging.warning("BLE client not connected. Attempting to re-establish connection.")
                if not await self.ensure_connected(): # Try to reconnect if disconnected
                    logging.error("Failed to re-establish BLE connection. Stopping send_velocities.")
                    break # Exit loop if re-connection fails

            # Planner logic
            if self.planner.sensor.has_new_plan:
                self.planner.add_plan()
                self.planner.sensor.has_new_plan = False

            if self.planner.gpath is not None and not self.planner.reached_goal:
                self.planner.updatePosition()
                if self.planner.reached_goal:
                    continue
                omega, _ = self.planner.replan()
                if omega is None:
                    await asyncio.sleep(self.send_interval) # Still wait if no valid omega
                    continue

                data = struct.pack("<ii", int(self.planner.parms.v_mm), -int(omega*180/np.pi))
                print('sent v: ', self.planner.parms.v_mm, " w: ", -int(omega*180/np.pi))
                try:
                    await self.ble_client.write_gatt_char(ESP32_CHAR_UUID, data)
                except Exception as e:
                    logging.error(f"BLE send error: {e}. Attempting to handle...")
                    # handle_ble_error now part of ensure_connected
                    # disconnect any existing client cleanly before ensure_connected tries to make new one
                    try:
                        if self.ble_client and self.ble_client.is_connected:
                            await self.ble_client.disconnect()
                    except Exception as disconnect_e:
                        logging.error(f"Error during client disconnect: {disconnect_e}")
                    
                    if not await self.ensure_connected(): # Try to reconnect
                        logging.error("Failed to re-establish BLE connection during send. Stopping send_velocities.")
                        break # Exit loop if re-connection fails
            else: # either no global path or the goal is reached
                data = struct.pack("<ii", 0, 0)
                try:
                    await self.ble_client.write_gatt_char(ESP32_CHAR_UUID, data)
                except Exception as e:
                    logging.error(f"BLE send error: {e}. Attempting to handle...")
                    # handle_ble_error now part of ensure_connected
                    # disconnect any existing client cleanly before ensure_connected tries to make new one
                    try:
                        if self.ble_client and self.ble_client.is_connected:
                            await self.ble_client.disconnect()
                    except Exception as disconnect_e:
                        logging.error(f"Error during client disconnect: {disconnect_e}")
                    
                    if not await self.ensure_connected(): # Try to reconnect
                        logging.error("Failed to re-establish BLE connection during send. Stopping send_velocities.")
                        break # Exit loop if re-connection fails
            
            await asyncio.sleep(self.send_interval)

    async def run(self):
        """Main async entry point"""
        self.running = True
        
        logging.info(f"Starting BLE bridge for address: {self.ble_address}")
        if not await self.ensure_connected(): # Ensure initial connection before starting send loop
            logging.error("Initial BLE connection failed. Bridge will not start.")
            self.running = False
            return

        try:
            await self.send_velocities()
        except asyncio.CancelledError:
            logging.info("BLE bridge tasks cancelled.")
        except Exception as e:
            logging.exception("An unhandled error occurred in run():") # Print full traceback
        finally:
            logging.info("BLE bridge shutting down.")
            self.running = False
            if self.ble_client and self.ble_client.is_connected:
                try:
                    await self.ble_client.disconnect()
                    logging.info("BLE client disconnected.")
                except Exception as e:
                    logging.error(f"Error during final BLE disconnect: {e}")

def start_asyncio_ble(bridge):
    """Run the asyncio BLE bridge in a separate thread"""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bridge.run())
    except KeyboardInterrupt:
        logging.info("BLE thread interrupted.")
    except Exception as e:
        logging.exception("Error in BLE thread:")
    finally:
        loop.close()
