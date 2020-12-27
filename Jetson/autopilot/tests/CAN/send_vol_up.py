import os
import can
import time
import binascii

os.system("sudo ip link set can0 up type can bitrate 33300")

bus = can.interface.Bus('can0', bustype='socketcan')

no_btn_msg = can.Message(arbitration_id=0x10438040, data=[0], is_extended_id=True)
vol_up_msg = can.Message(arbitration_id=0x10438040, data=[1], is_extended_id=True)
print(hex(vol_up_msg.arbitration_id) + ": " + str(binascii.hexlify(vol_up_msg.data)))
bus.send(vol_up_msg)
time.sleep(0.1)
bus.send(no_btn_msg)
time.sleep(0.1)
bus.send(vol_up_msg)
time.sleep(0.1)
bus.send(no_btn_msg)
time.sleep(0.1)
bus.send(vol_up_msg)
time.sleep(0.1)
bus.send(no_btn_msg)
time.sleep(0.1)
bus.send(vol_up_msg)
time.sleep(0.1)
bus.send(no_btn_msg)
time.sleep(0.1)
bus.send(vol_up_msg)
time.sleep(0.1)
bus.send(no_btn_msg)
time.sleep(0.1)
bus.send(vol_up_msg)
time.sleep(0.1)
bus.send(no_btn_msg)
time.sleep(0.1)