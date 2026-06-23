import asyncio
import legoeducation as le
from pyscript.js_modules import ble
from pyscript import window

class Element:
    def __init__(self):
        self.myble = ble.BLEDevice.new()
        self.hub = None

    async def connect(self):
        print("connecting")
        try:
            await self.myble.scan()
        except Exception as e:
            print('Scan error: ',e)
            return 
        if not self.myble.device:
            window.console.log(f'failed {self.myble.device}')
            return
        try:
            await self.myble.connect(self.my_callback)
            name = self.myble.device.name 
            if "Double Motor" in name:
                try:                    
                    import legoeducation.basic_device as bd
                    registry = getattr(bd.my_worker, '_myble_registry', None)
                    if registry is not None:
                        registry[id(self.device)] = self.myble
                    print(f"Connected: {self.myble.device.name}")
                    self.hub = le.DoubleMotor()
                    self.hub.connected = True
                    print("trying to run")
                    
                except Exception as e:
                    print(f"Connection failed: {e}")

            elif "Single Motor" in name:
                self.hub = le.SingleMotor()
            elif "Color Sensor" in name:
                self.hub = le.ColorSensor()
            elif "Controller" in name:
                self.hub = le.Controller()
            return
        except Exception as e:
            print(f"Interrupted {e}")

    def my_callback(self, characteristic, data):
        window.console.log("calling back")
              
    
        
        