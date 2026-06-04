"""
spike.py — BLE communication layer for the LEGO SPIKE Prime hub.

Handles:
  - COBS encoding/decoding (the wire format the hub requires)
  - Connecting via Bluetooth
  - Uploading a Python file to a program slot
  - Starting / stopping programs
  - Printing console output from the hub
"""

import asyncio
import struct
from bleak import BleakClient, BleakScanner

# ── BLE UUIDs (from the SPIKE Prime protocol docs) ────────────────────────────
SERVICE_UUID = "0000fd02-0000-1000-8000-00805f9b34fb"
RX_UUID      = "0000fd02-0001-1000-8000-00805f9b34fb"  # we WRITE here
TX_UUID      = "0000fd02-0002-1000-8000-00805f9b34fb"  # we READ notifications here

# ── COBS constants ─────────────────────────────────────────────────────────────
DELIMITER       = 0x02   # end-of-message marker
XOR             = 0x03   # all bytes are XOR'd with this before sending
MAX_BLOCK_SIZE  = 84
COBS_CODE_OFFSET = 3
NO_DELIMITER    = 0xFF


# ══════════════════════════════════════════════════════════════════════════════
# COBS encode / decode
# (Directly from the SPIKE Prime protocol documentation examples)
# ══════════════════════════════════════════════════════════════════════════════

def _cobs_encode(data: bytes) -> bytearray:
    """Escape 0x00, 0x01, 0x02 using the SPIKE-flavoured COBS algorithm."""
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
                delimiter_base = byte * MAX_BLOCK_SIZE
                block_offset   = block + COBS_CODE_OFFSET
                buffer[code_index] = delimiter_base + block_offset
            begin_block()

    buffer[code_index] = block + COBS_CODE_OFFSET
    return buffer


def _cobs_decode(data: bytes) -> bytearray:
    """Reverse COBS encoding."""
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
    """Encode + XOR + append delimiter — ready to write to RX characteristic."""
    buf = _cobs_encode(data)
    for i in range(len(buf)):
        buf[i] ^= XOR
    buf.append(DELIMITER)
    return bytes(buf)


def unpack(frame: bytes) -> bytes:
    """Strip delimiter, reverse XOR, COBS-decode — recovers original message."""
    start = 1 if frame[0] == 0x01 else 0
    unframed = bytes(b ^ XOR for b in frame[start:-1])
    return bytes(_cobs_decode(unframed))


# ══════════════════════════════════════════════════════════════════════════════
# CRC32 helper  (required by the file-upload protocol)
# ══════════════════════════════════════════════════════════════════════════════

def _crc32(data: bytes, crc: int = 0) -> int:
    """Standard CRC-32 (same as binascii.crc32 but dependency-free)."""
    import binascii
    return binascii.crc32(data, crc) & 0xFFFFFFFF


# ══════════════════════════════════════════════════════════════════════════════
# Message builders  (serialize raw bytes per the protocol spec)
# ══════════════════════════════════════════════════════════════════════════════

def msg_info_request() -> bytes:
    return bytes([0x00])


def msg_start_file_upload(filename: str, slot: int, crc: int) -> bytes:
    # 0x0C | name[32] null-terminated | slot uint8 | crc uint32 LE
    name_bytes = filename.encode() + b'\x00'
    name_bytes = name_bytes[:32].ljust(32, b'\x00')
    return bytes([0x0C]) + name_bytes + struct.pack('<BI', slot, crc)


def msg_transfer_chunk(running_crc: int, chunk: bytes) -> bytes:
    # 0x10 | running_crc uint32 LE | chunk_size uint16 LE | chunk bytes
    return bytes([0x10]) + struct.pack('<IH', running_crc, len(chunk)) + chunk


def msg_program_flow(action: int, slot: int) -> bytes:
    # action: 0x00 = start, 0x01 = stop
    return bytes([0x1E, action, slot])


# ══════════════════════════════════════════════════════════════════════════════
# SpikeHub — high-level async interface
# ══════════════════════════════════════════════════════════════════════════════

class SpikeHub:
    """
    Async context manager that connects to a SPIKE Prime hub over BLE.

    Usage:
        async with SpikeHub() as hub:
            await hub.upload_program("my_program.py", slot=0)
            await hub.start_program(slot=0)
    """

    def __init__(self, device_name: str = None):
        self._device_name   = device_name   # if None, connects to first hub found
        self._client        = None
        self._rx_queue      = asyncio.Queue()
        self._info          = None           # parsed InfoResponse
        self._rx_buf        = bytearray()    # accumulates partial BLE packets

    # ── context manager ───────────────────────────────────────────────────────

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
            raise RuntimeError(
                "No SPIKE Prime hub found. Make sure it is powered on and in range."
            )

        print(f"Found hub: {device.name}  ({device.address})")
        await asyncio.sleep(1.0)  # brief pause so Windows BLE stack is ready
        self._client = BleakClient(device)
        await self._client.connect(timeout=15.0)
        print("Connected.")

        # Enable TX notifications so we receive messages from the hub
        await self._client.start_notify(TX_UUID, self._on_notify)
        await asyncio.sleep(0.5)  # give the hub a moment to settle after notify setup

        # Handshake — send InfoRequest, wait for InfoResponse
        await self._send_raw(msg_info_request())
        info_bytes = await self._wait_for_message(0x01, timeout=10.0)
        self._info = self._parse_info_response(info_bytes)
        print(f"Hub firmware {self._info['fw_major']}.{self._info['fw_minor']}  "
              f"| max chunk: {self._info['max_chunk_size']} bytes")

    async def disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
            print("Disconnected.")

    # ── file upload ───────────────────────────────────────────────────────────

    async def upload_program(self, source_code: str, slot: int = 0,
                             filename: str = "program.py"):
        """
        Upload a Python source string to the hub as a program in `slot`.
        """
        data      = source_code.encode("utf-8")
        total_crc = _crc32(data)

        print(f"Uploading {len(data)} bytes to slot {slot}...")

        # 1. Tell the hub we want to upload a file
        await self._send_raw(msg_start_file_upload(filename, slot, total_crc))
        resp = await self._wait_for_message(0x0D, timeout=5.0)
        if resp[1] != 0x00:
            raise RuntimeError(f"StartFileUpload rejected (status={resp[1]:#x})")

        # 2. Send data in chunks
        chunk_size  = self._info["max_chunk_size"]
        running_crc = 0
        offset      = 0
        chunk_num   = 0

        while offset < len(data):
            chunk        = data[offset : offset + chunk_size]
            running_crc  = _crc32(chunk, running_crc)
            await self._send_raw(msg_transfer_chunk(running_crc, chunk))
            resp = await self._wait_for_message(0x11, timeout=5.0)
            if resp[1] != 0x00:
                raise RuntimeError(
                    f"TransferChunk {chunk_num} rejected (status={resp[1]:#x})"
                )
            offset    += len(chunk)
            chunk_num += 1
            print(f"  chunk {chunk_num} sent ({offset}/{len(data)} bytes)")

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

    async def stream_console(self, duration: float = 60.0):
        """Print console output from the hub for `duration` seconds."""
        print("─── Hub console output ───────────────────────────────")
        deadline = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(self._rx_queue.get(), timeout=1.0)
                if msg[0] == 0x21:          # ConsoleNotification
                    text = msg[1:].rstrip(b'\x00').decode("utf-8", errors="replace")
                    print(f"  HUB > {text}", end="")
            except asyncio.TimeoutError:
                continue
        print("\n─── End console output ───────────────────────────────")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _on_notify(self, _sender, data: bytearray):
        """Called by bleak whenever the hub sends a BLE notification packet."""
        # Accumulate bytes until we see the end-of-message delimiter (0x02)
        self._rx_buf.extend(data)
        while DELIMITER in self._rx_buf:
            end = self._rx_buf.index(DELIMITER)
            frame = bytes(self._rx_buf[: end + 1])
            self._rx_buf = self._rx_buf[end + 1 :]
            try:
                message = unpack(frame)
                self._rx_queue.put_nowait(message)
            except Exception:
                pass  # malformed frame — skip

    async def _send_raw(self, message: bytes):
        """Pack and send a message to the hub, respecting max_packet_size."""
        frame       = pack(message)
        packet_size = self._info["max_packet_size"] if self._info else len(frame)
        for i in range(0, len(frame), packet_size):
            await self._client.write_gatt_char(
                RX_UUID, frame[i : i + packet_size], response=False
            )

    async def _wait_for_message(self, msg_type: int, timeout: float = 5.0) -> bytes:
        """Block until a message of the given type arrives, then return it."""
        loop     = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for message type {msg_type:#x}"
                )
            try:
                msg = await asyncio.wait_for(self._rx_queue.get(),
                                             timeout=min(remaining, 1.0))
                if msg[0] == msg_type:
                    return msg
                # Not the message we want — put it back and keep waiting
                await self._rx_queue.put(msg)
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                continue

    @staticmethod
    def _parse_info_response(data: bytes) -> dict:
        # uint8 type | uint8 rpc_maj | uint8 rpc_min | uint16 rpc_build
        # | uint8 fw_maj | uint8 fw_min | uint16 fw_build
        # | uint16 max_packet | uint16 max_message | uint16 max_chunk | uint16 device
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