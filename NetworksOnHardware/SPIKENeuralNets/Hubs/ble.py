# Chris's js ble code converted to python with claud 
import asyncio
from bleak import BleakClient, BleakScanner

SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb'
WRITE_UUID   = '0000fd02-0001-1000-8000-00805f9b34fb'
NOTIFY_UUID  = '0000fd02-0002-1000-8000-00805f9b34fb'

class BLEDevice:
    def __init__(self):
        self.device   = None
        self.client   = None
        self.callback = None

    async def scan(self):
        print("Scanning for SPIKE Prime hub...")

        def device_filter(device, adv):
            name = adv.local_name == "Hub 1" or adv.local_name == "Hub 2"
            uuid = SERVICE_UUID.lower() in [str(u).lower() for u in (adv.service_uuids or [])]
            return name and uuid
        
        self.device = await BleakScanner.find_device_by_filter(
            filterfunc=device_filter,
            timeout=10.0,
        )
        if self.device is None:
            raise RuntimeError("No SPIKE Prime hub found.")
        print(f"Found: {self.device.name}  ({self.device.address})")

    async def connect(self, callback):
        self.callback = callback
        if self.device:
            try:
                self.client = BleakClient(self.device)
                await self.client.connect(timeout=15.0, use_cached=False)
                await self.client.start_notify(NOTIFY_UUID, self._handle_notification)
                print("Connected.")
            except Exception as e:
                print(f"Error connecting: {e}")

    async def _handle_notification(self, sender, data: bytearray):
        if self.callback:
            asyncio.ensure_future(self.callback(NOTIFY_UUID, bytes(data)))

    async def send(self, data: bytes):
        if not self.client or not self.client.is_connected:
            print("Error: not connected to device")
            return
        try:
            await self.client.write_gatt_char(WRITE_UUID, bytearray(data), response=False)
            print(f"Sent: {bytes(data).hex()}")
        except Exception as e:
            print(f"Error writing: {e}")

    async def disconnect(self):
        if self.client and self.client.is_connected:
            asyncio.ensure_future(self.client.disconnect())