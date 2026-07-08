#!/usr/bin/env python3
"""BH1750环境光传感器驱动"""
import time
from smbus2 import SMBus

class BH1750:
    def __init__(self, bus=4, addr=0x23):
        self.bus = SMBus(bus)
        self.addr = addr
        self._init()

    def _init(self):
        self.bus.write_byte(self.addr, 0x01)   # Power On
        time.sleep(0.01)
        self.bus.write_byte(self.addr, 0x07)   # Reset
        time.sleep(0.01)

    def read_lux(self):
        self.bus.write_byte(self.addr, 0x20)   # One-time high-res mode
        time.sleep(0.2)
        data = self.bus.read_i2c_block_data(self.addr, 0x00, 2)
        raw = (data[0] << 8) | data[1]
        return round(raw / 1.2, 1)

    def close(self):
        self.bus.close()