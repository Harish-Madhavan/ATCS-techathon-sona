try:
    import serial
except ImportError:
    serial = None
import time
import traceback

class ESP32SerialController:
    def __init__(self, port, baudrate, approach_mapping):
        self.port = port
        self.baudrate = baudrate
        self.approach_mapping = dict(approach_mapping or {})
        self.serial_connection = None
        self.is_connected = False
        self._connect()

    @property
    def is_mock(self):
        return str(self.port).upper() == "MOCK"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _connect(self):
        if str(self.port).upper() == "MOCK":
            self.is_connected = True
            print("[ESP32 Mock] Bypassing hardware connection and loading virtual device mode.")
            return

        if serial is None:
            print("[ESP32 Error] pyserial is not installed. Use port='MOCK' for virtual simulation.")
            self.is_connected = False
            return

        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2) 
            self.is_connected = True
            print(f"[ESP32] Successfully connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"[ESP32 Error] Failed to connect to {self.port}: {e}")
            self.serial_connection = None 
            self.is_connected = False
        except Exception as e_generic: 
            print(f"[ESP32 Error] A generic error occurred during connection to {self.port}: {e_generic}")
            traceback.print_exc()
            self.serial_connection = None
            self.is_connected = False


    def update_lights(self, approach_statuses):
        if not self.is_connected: 
            return

        if str(self.port).upper() != "MOCK" and not self.serial_connection:
            return

        command_parts = []
        for approach_name in sorted(approach_statuses):
            status_info = approach_statuses[approach_name] or {}
            short_code = self.approach_mapping.get(approach_name)
            if short_code:
                raw_state = str(status_info.get('state', 'RED')).upper()
                light_state_char = raw_state[0] if raw_state else 'R'
                if light_state_char not in ('R', 'Y', 'G'):
                    light_state_char = 'R'
                command_parts.append(f"{short_code}:{light_state_char}")

        if not command_parts:
            return

        command_string = ",".join(command_parts) + "\n"
        self.send_command(command_string)

    def build_command(self, approach_statuses):
        """Pure helper for tests: deterministic command string without I/O."""
        parts = []
        for approach_name in sorted(approach_statuses or {}):
            code = self.approach_mapping.get(approach_name)
            if not code:
                continue
            raw = str((approach_statuses[approach_name] or {}).get('state', 'RED')).upper()
            ch = raw[0] if raw else 'R'
            parts.append(f"{code}:{ch if ch in ('R', 'Y', 'G') else 'R'}")
        return ",".join(parts) + "\n" if parts else ""

    def send_command(self, command_string):
        """ Sends a formatted command string to the ESP32. """
        if not self.is_connected:
            print("[ESP32 Error] Attempted to send command while not connected.")
            return

        if str(self.port).upper() == "MOCK":
            print(f"[ESP32 Mock Writing] > {command_string.strip()}")
            return

        if not self.serial_connection:
            print("[ESP32 Error] Attempted to send command while not connected.")
            return

        try:
            print(f"[ESP32 Sending] > {command_string.strip()}")
            bytes_written = self.serial_connection.write(command_string.encode('ascii'))
            self.serial_connection.flush() 
            
        except serial.SerialTimeoutException:
            print(f"[ESP32 Error] Serial write timeout on {self.port}.")
            
        except serial.SerialException as e:
            print(f"[ESP32 Error] Failed to write to serial port {self.port}: {e}")
            self.is_connected = False 
            self.close() 
        except Exception as e_gen: 
            print(f"[ESP32 Error] Unexpected error during send to {self.port}: {e_gen}")
            traceback.print_exc()
            self.is_connected = False
            self.close()


    def close(self):
        if str(self.port).upper() == "MOCK":
            self.is_connected = False
            return

        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
                print(f"[ESP32] Serial connection to {self.port} closed.")
            except Exception as e:
                print(f"[ESP32 Error] Error closing serial port {self.port}: {e}")
        self.is_connected = False
        self.serial_connection = None 

    def reconnect(self): 
        if not self.is_connected:
            print(f"[ESP32] Attempting to reconnect to {self.port}...")
            self.close() 
            self._connect()
        return self.is_connected
