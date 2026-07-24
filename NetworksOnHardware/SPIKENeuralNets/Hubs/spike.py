"""
spike.py — BLE communication layer for the LEGO SPIKE Prime hub.
"""

import asyncio
import struct
from bleak import BleakClient, BleakScanner

# ── BLE UUIDs ─────────────────────────────────────────────────────────────────
SERVICE_UUID = "0000fd02-0000-1000-8000-00805f9b34fb"
RX_UUID      = "0000fd02-0001-1000-8000-00805f9b34fb"  # we WRITE here
TX_UUID      = "0000fd02-0002-1000-8000-00805f9b34fb"  # we READ notifications here

# ── COBS constants ─────────────────────────────────────────────────────────────
DELIMITER        = 0x02
XOR              = 0x03
MAX_BLOCK_SIZE   = 84
COBS_CODE_OFFSET = 3
NO_DELIMITER     = 0xFF


# ══════════════════════════════════════════════════════════════════════════════
# COBS encode / decode  (matches official SPIKE Prime protocol docs exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _cobs_encode(data: bytes) -> bytearray:
    buffer = bytearray()
    code_index = block = 0

    def begin_block():
        nonlocal code_index, block
        code_index = len(buffer)
        buffer.append(NO_DELIMITER)
        block = 1

    begin_block()
    for byte in data:
        if byte > DELIMITER:
            buffer.append(byte)
            block += 1
        if byte <= DELIMITER or block > MAX_BLOCK_SIZE:
            if byte <= DELIMITER:
                buffer[code_index] = byte * MAX_BLOCK_SIZE + block + COBS_CODE_OFFSET
            begin_block()

    buffer[code_index] = block + COBS_CODE_OFFSET
    return buffer


def _cobs_decode(data: bytes) -> bytearray:
    buffer = bytearray()

    def unescape(code: int):
        if code == 0xFF:
            return None, MAX_BLOCK_SIZE + 1
        value, block = divmod(code - COBS_CODE_OFFSET, MAX_BLOCK_SIZE)
        if block == 0:
            block  = MAX_BLOCK_SIZE
            value -= 1
        return value, block

    value, block = unescape(data[0])
    for byte in data[1:]:
        block -= 1
        if block > 0:
            buffer.append(byte)
            continue
        if value is not None:
            buffer.append(value)
        value, block = unescape(byte)

    return buffer


def pack(data: bytes) -> bytes:
    """COBS-encode, XOR, append delimiter — ready to write to RX."""
    buf = _cobs_encode(data)
    for i in range(len(buf)):
        buf[i] ^= XOR
    buf.append(DELIMITER)
    return bytes(buf)


def unpack(frame: bytes) -> bytes:
    start = 1 if frame[0] == 0x01 else 0
    # Strip delimiter, reverse XOR
    unframed = bytes(b ^ XOR for b in frame[start:-1])
    # Skip the first COBS code word byte — real message starts at index 1
    return unframed[1:]


# ══════════════════════════════════════════════════════════════════════════════
# CRC32
# ══════════════════════════════════════════════════════════════════════════════

def _crc32(data: bytes, crc: int = 0) -> int:
    import binascii
    return binascii.crc32(data, crc) & 0xFFFFFFFF


# ══════════════════════════════════════════════════════════════════════════════
# Message builders
# ══════════════════════════════════════════════════════════════════════════════

def msg_info_request() -> bytes:
    return bytes([0x00])

def msg_start_file_upload(filename: str, slot: int, crc: int) -> bytes:
    name_bytes = (filename.encode() + b'\x00')[:32].ljust(32, b'\x00')
    return bytes([0x0C]) + name_bytes + struct.pack('<BI', slot, crc)

def msg_transfer_chunk(running_crc: int, chunk: bytes) -> bytes:
    return bytes([0x10]) + struct.pack('<IH', running_crc, len(chunk)) + chunk

def msg_program_flow(action: int, slot: int) -> bytes:
    return bytes([0x1E, action, slot])


# ══════════════════════════════════════════════════════════════════════════════
# SpikeHub
# ══════════════════════════════════════════════════════════════════════════════

class SpikeHub:

    def __init__(self, device_name: str = "Hub 2"):
        self._device_name = device_name
        self._client      = None
        self._rx_queue    = asyncio.Queue()
        self._info        = None
        self._rx_buf      = bytearray()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.disconnect()

    # ── connection ────────────────────────────────────────────────────────────

    async def connect(self):
        print("Scanning for SPIKE Prime hub...")
        device = await BleakScanner.find_device_by_filter(
            lambda d, adv: (
                (self._device_name and d.name == self._device_name)
                or
                (not self._device_name and SERVICE_UUID.lower() in
                    [str(u).lower() for u in (adv.service_uuids or [])])
            ),
            timeout=10.0,
        )
        if device is None:
            raise RuntimeError("No SPIKE Prime hub found.")

        print(f"Found hub: {device.name}  ({device.address})")
        await asyncio.sleep(1.0)
        self._client = BleakClient(device)
        await self._client.connect(timeout=15.0, use_cached=False)
        print("Connected.")

        # Enable notifications BEFORE sending anything
        await self._client.start_notify(TX_UUID, self._on_notify)
        await asyncio.sleep(2)

        # Skip handshake — hub only responds when a program with print() is running
        # Use safe conservative defaults
        self._info = {
            "fw_major": 0, "fw_minor": 0,
            "max_packet_size": 32,
            "max_chunk_size":  200,
        }
        print("Using default hub settings.")

    async def disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
            print("Disconnected.")

    # ── file upload ───────────────────────────────────────────────────────────

    async def upload_program(self, source_code: str, slot: int = 0,
                             filename: str = "program.py"):
        data      = source_code.encode("utf-8")
        total_crc = _crc32(data)
        print(f"Uploading {len(data)} bytes to slot {slot}...")

        await self._send_raw(msg_start_file_upload(filename, slot, total_crc))
        resp = await self._wait_for_message(0x0D, timeout=10.0)
        if resp[1] != 0x00:
            raise RuntimeError(f"StartFileUpload rejected (status={resp[1]:#x})")

        chunk_size  = self._info["max_chunk_size"]
        running_crc = 0
        offset      = 0
        chunk_num   = 0

        while offset < len(data):
            chunk        = data[offset : offset + chunk_size]
            running_crc  = _crc32(chunk, running_crc)
            await self._send_raw(msg_transfer_chunk(running_crc, chunk))
            resp = await self._wait_for_message(0x11, timeout=10.0)
            if resp[1] != 0x00:
                raise RuntimeError(f"Chunk {chunk_num} rejected (status={resp[1]:#x})")
            offset    += len(chunk)
            chunk_num += 1
            print(f"  chunk {chunk_num} ({offset}/{len(data)} bytes)")

        print("Upload complete.")

    # ── program flow ──────────────────────────────────────────────────────────

    async def start_program(self, slot: int = 0):
        print(f"Starting program in slot {slot}...")
        await self._send_raw(msg_program_flow(0x00, slot))
        await self._wait_for_message(0x1F, timeout=5.0)
        print("Program started.")

    async def stop_program(self, slot: int = 0):
        await self._send_raw(msg_program_flow(0x01, slot))
        await self._wait_for_message(0x1F, timeout=5.0)
        print("Program stopped.")

    async def stream_console(self, duration: float = 60.0,
                             line_callback=None):
        """
        Stream console output from the hub.
        If line_callback is provided, call it with each line of text
        instead of printing — useful for data collection.
        """
        print("─── Hub console ──────────────────────────────────────")
        loop     = asyncio.get_running_loop()
        deadline = loop.time() + duration
        while loop.time() < deadline:
            try:
                msg = await asyncio.wait_for(self._rx_queue.get(), timeout=1.0)
                if msg[0] == 0x22:   # ConsoleNotification
                    text = msg[1:].rstrip(b'\x00').decode("utf-8", errors="replace")
                    if line_callback:
                        line_callback(text)
                    else:
                        print(f"  HUB > {text}", end="")
            except asyncio.TimeoutError:
                continue
        print("\n─── End console ──────────────────────────────────────")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _on_notify(self, _sender, data: bytearray):
        print(f"  RAW NOTIFY: {bytes(data).hex()}  ({len(data)} bytes)")
        self._rx_buf.extend(data)
        print(f"  Buffer now: {bytes(self._rx_buf).hex()}")
        print(f"  0x02 in buffer: {0x02 in self._rx_buf}")
        
        while True:
            if 0x02 not in self._rx_buf:
                break
            end   = self._rx_buf.index(0x02)
            frame = bytes(self._rx_buf[:end + 1])
            self._rx_buf = self._rx_buf[end + 1:]
            print(f"  RAW frame: {frame.hex()}")
            try:
                message = unpack(frame)
                print(f"  Decoded: {message.hex()}  type={message[0]:#x}")
                if len(message) > 0:
                    self._rx_queue.put_nowait(message)
            except Exception as e:
                print(f"  Decode error: {e}")
                
    async def _send_raw(self, message: bytes):
        print("in send raw")
        frame       = pack(message)
        packet_size = self._info["max_packet_size"] if self._info else len(frame)
        for i in range(0, len(frame), packet_size):
            await self._client.write_gatt_char(
                RX_UUID, frame[i : i + packet_size], response=False
            )
        print(f"  Sent: {frame.hex()}")

    async def _wait_for_message(self, msg_type: int,
                                timeout: float = 10.0) -> bytes:
        loop     = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for message type {msg_type:#x}"
                )
            try:
                msg = await asyncio.wait_for(
                    self._rx_queue.get(), timeout=min(remaining, 1.0)
                )
                print(f"  _wait_for_message got: {msg.hex()}  type={msg[0]:#x}  waiting for={msg_type:#x}  match={msg[0] == msg_type}")
                if msg[0] == msg_type:
                    return msg
                await self._rx_queue.put(msg)
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                print("got timeout error, continuing")
                continue

    @staticmethod
    def _parse_info_response(data: bytes) -> dict:
        (_, rpc_maj, rpc_min, rpc_build,
         fw_maj, fw_min, fw_build,
         max_packet, max_message, max_chunk,
         device_type) = struct.unpack_from('<BBBHBBHHHH H', data)
        return {
            "rpc_major":        rpc_maj,
            "rpc_minor":        rpc_min,
            "fw_major":         fw_maj,
            "fw_minor":         fw_min,
            "max_packet_size":  max_packet,
            "max_message_size": max_message,
            "max_chunk_size":   max_chunk,
        }