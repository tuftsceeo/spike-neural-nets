# Chris's Hub code

import asyncio 
import struct
import json
from Hubs import ble

from Hubs import SpikePrime as hub0
hubs = [hub0] 

class Hub_PS:
    def __init__(self, hub = 2):
        self.defined_hub = hub
        self.hubInfo = hubs[hub]
        self.myble = ble.BLEDevice()
        self.connected = False
        
        self.info = None
        self.reply = None
        self.list_update = False
        self.final_callback = None
        self.info_callback = None

        self.collect = False
        
        self.name = "spike"
        # self.sync = document.getElementById('sync'+suffix)
        # self.sync.onclick = self.ask
        # self.dropdown = document.getElementById('dropdown'+suffix)
        # self.latest = document.getElementById('value'+suffix)
        # self.activity = document.getElementById('activity'+suffix)

    async def ask(self): 
        if not self.connected:
            await self.myble.scan()
            if not self.myble.device: 
                print(f'failed {self.myble.device}')
                return
            self.connected = True
            try:
                await self.myble.connect(self.my_callback)
                # Send request for info and then feed rate (in msec)
                print('starting to test')
                fmt, ID, val = self.hubInfo.commands.get('info')
                print(fmt, ID)
                await self.send(fmt, ID)
                for i in range(50):
                    if self.info:  # waits for the info_response to arrive and get parsed
                        if self.info['Firmware']['major'] < 1 and self.defined_hub == 2:
                            print('wrong version')
                            self.hubInfo = hubs[1]
                        await self.feed_rate(1000)
                        return
                    #print('waiting')
                    await asyncio.sleep(0.1)
                self.hubInfo = hubs[2] if self.defined_hub == 0 else hubs[0] #maybe the wrong hub
                print('trying again ',fmt, ID)
                await self.send(fmt, ID)
                for i in range(50):
                    if self.info:  # waits for the info_response to arrive and get parsed
                        await self.feed_rate(1000)
                        return
                    print('waiting')
                    await asyncio.sleep(0.1)
                print('cannot find element')
                    
            except Exception as e:
                print(f"Interrupted {e}")
        else:
            await self.myble.disconnect()
            self.connected = False
            self.info = None
            self.list_update = False

            
    def device_message(self, data, verbose = False):
        messages = {}
        while data:
            ID = data[0]
            if verbose: print([i for i in data])
            if ID in self.hubInfo.DEVICE_MESSAGE_MAP:
                name, fmt, keys = self.hubInfo.DEVICE_MESSAGE_MAP[ID]
                if verbose: print(name, fmt, keys)
                size = struct.calcsize(fmt)
                if size > len(data):
                    if verbose: print('Remaining characters ',data)
                    break
                content = struct.unpack(fmt, data[:size])[1:]  #get rid of id
                if keys:
                    if keys[0] == 'port':
                        name = name + '_' + self.hubInfo.port_lut[content[0]]
                    messages[name] = {k:v for k,v in zip(keys,content)}
                else:
                    messages[name] = content[0] if size == 2 else content
                data = data[size:]
            else:
                print(f"Unknown device ID: {ID}")
                break
        return messages
    
    def info_response(self, data):
        messages = {}
        for LINE in self.hubInfo.INFO_MESSAGE:
            name, fmt, keys = LINE
            size = struct.calcsize(fmt)
            content = struct.unpack(fmt, data[:size])
            if keys:
                messages[name] = {k:v for k,v in zip(keys,content)}
            else:
                messages[name] = content[0] if size == 2 else content
            data = data[size:]
        return messages
    
    def makeList(self, reply):
        self.myList = list(reply.keys())
        print('my list is ',self.myList)
        self.list_update = True

        options = []
        for f in self.myList:
            if f in self.hubInfo.TO_HIDE:
                continue
            try:
                new = list(reply[f].keys())
                for n in new:
                    if n in self.hubInfo.TO_HIDE:
                        continue
                    options.append(f+': '+n)
            except:
                pass

        # for i,attribute in enumerate(options):
        #     option = document.createElement("option")
        #     option.value = attribute
        #     option.text = attribute
        #     self.dropdown.appendChild(option)
        return
    
    async def send(self, fmt, ID, val = None):
        print(fmt, ID, val)
        payload = [ID]
        if val:
            payload.extend(val['values'].values())
        print('Values: ',payload)
        message = self.hubInfo.pack(struct.pack(fmt, *payload))
        #packet_size = info['MaxSize']['packet'] if info else len(message) - issue here with TechElements
        packet_size = len(message)  # send the frame in packets of packet_size
        for i in range(0, len(message), packet_size):
            packet = message[i : i + packet_size]
            print(f"Sending: {packet}")
            await self.myble.send([p for p in packet])

    async def feed_rate(self, update = 1000):
        fmt, ID, val = self.hubInfo.commands.get('feed')
        val['values']['updateTime'] = update
        await self.send(fmt, ID, val)
    
    async def my_callback(self, characteristic, data):
        if self.hubInfo.hubType == 'SPIKEPrime':
            if data[-1] != 0x02:  # for simplicity, this example does not implement buffering
                print(f"Received incomplete message:\n {[d for d in data]}")
                return
        data = [d for d in data]
        reply = self.hubInfo.unpack(data)
        #print(f'Received: {[r for r in reply]}')
        ID = reply[0]
    
        if ID == 1:
            data = bytes(reply[1:])
            info = self.info_response(data)
            print(info)
            #self.name.innerHTML = f"{self.hubInfo.hubType} ({info['Firmware']['major']}.{info['Firmware']['minor']}b{info['Firmware']['build']})"
            self.info = info
            #print("info: " + json.dumps(info))
            if self.info_callback: 
                await self.info_callback(self.info)
            
        if ID == 60:
            if not self.info:
                return
            length = struct.unpack('<H',reply[1:3])[0]
            data = bytes(reply[3:])
            if length > len(data):
                print(f'error - {length} > {len(data)}')
                return
            self.reply = self.device_message(data, False)
            # if not self.list_update:
            #     self.makeList(self.reply)
            # if ':' in self.dropdown.value:
            #     a = self.dropdown.value.split(': ')
            #     self.value = self.reply[a[0]][a[1]]
            # else:
            #     self.value = self.reply[self.dropdown.value]
            #self.latest.innerHTML = json.dumps(self.value)
            #print("reply: " + json.dumps(self.reply))
        if self.final_callback:
            if self.collect:
                await self.final_callback(self.reply)