import os
import can
import time
import binascii

os.system("sudo ip link set can0 up type can bitrate 33300")

bus = can.interface.Bus('can0', bustype='socketcan')

cc_off_msg = can.Message(arbitration_id=0x10758040, data=[0x00, 0x00], is_extended_id=True)
cc_on_msg = can.Message(arbitration_id=0x10758040, data=[0x00, 0x40], is_extended_id=True)
print(hex(cc_on_msg.arbitration_id) + ": " + str(binascii.hexlify(cc_on_msg.data)))
bus.send(cc_on_msg)
time.sleep(10)
print(hex(cc_off_msg.arbitration_id) + ": " + str(binascii.hexlify(cc_off_msg.data)))
bus.send(cc_off_msg)
