import os
import can
import time
import binascii

os.system("sudo ip link set can0 up type can bitrate 33300")

bus = can.interface.Bus('can0', bustype='socketcan')

can_dict = {}

last_out_time = 0;

def access_bit(data, num):
    base = int(num // 8)
    shift = int(num%8)
    return (data[base] & (1<<shift)) >> shift
    
def get_bitfield(data):
    return [access_bit(data, i) for i in range(len(data)*8)]
    
def get_blinker(can_dict):
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x1020c040'):
            data = message.data
            blinker_links = get_bitfield(data)[5]
            blinker_rechts = get_bitfield(data)[6]
            
            return [blinker_links, blinker_rechts]

def get_speed(can_dict):
    
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x1020c040'):
            data = message.data
            print(get_bitfield(data))
            blinker_links = get_bitfield(data)[5]
            blinker_rechts = get_bitfield(data)[6]
            print("Blinker links: " + str(blinker_links))
            print("Blinker rechts: " + str(blinker_rechts))
            print("Blinker:" + hex(arbitration_id) + ": " + str(binascii.hexlify(data)))
            speed = int.from_bytes(data[4:6], byteorder='big', signed=True) * 0.031 - 1.5
            print(speed)
            return
    
    

while True:
    message = bus.recv()
    
    can_dict[message.arbitration_id] = message
    
    
    
    #if "5b" in hex(message.arbitration_id):
    
    if (time.time() - last_out_time > 0.2): # seconds
        os.system('clear')
        get_speed(can_dict)
        for arbitration_id, message in can_dict.items():
            pass
            
        
            #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
            #if (hex(arbitration_id) == '0x10240040'):
            #    print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
                
            
            
        
        last_out_time = time.time()
