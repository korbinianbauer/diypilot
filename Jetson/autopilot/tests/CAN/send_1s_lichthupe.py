import os
import can
import time
import binascii

os.system("sudo ip link set can0 up type can bitrate 33300")

bus = can.interface.Bus('can0', bustype='socketcan')

lichthupe_aus = can.Message(arbitration_id=0x10ace060, data=[0x1e, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00], is_extended_id=True)
lichthupe_an = can.Message(arbitration_id=0x10ace060, data=[0x1e, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00], is_extended_id=True)
print(hex(lichthupe_an.arbitration_id) + ": " + str(binascii.hexlify(lichthupe_an.data)))
bus.send(lichthupe_an)
time.sleep(1)
print(hex(lichthupe_aus.arbitration_id) + ": " + str(binascii.hexlify(lichthupe_aus.data)))
bus.send(lichthupe_aus)