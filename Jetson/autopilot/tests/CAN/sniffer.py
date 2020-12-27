import os
import can
import time
import binascii

os.system("sudo ip link set can0 up type can bitrate 33300")

bus = can.interface.Bus('can0', bustype='socketcan')

can_dict = {}

last_out_time = 0;

def get_steering_wheel_angle(can_dict):
    
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x10240040'):
            data = message.data
            print("SWA:" + hex(arbitration_id) + ": " + str(binascii.hexlify(data)))
            angle = -1 * int.from_bytes(data[4:6], byteorder='big', signed=True)
            print(angle)
            return
    
    

while True:
    message = bus.recv()
    
    can_dict[message.arbitration_id] = message
    
    
    
    #if "5b" in hex(message.arbitration_id):
    
    if (time.time() - last_out_time > 0.2): # seconds
        os.system('clear')
        os.system('clear')
        #get_steering_wheel_angle(can_dict)
        
        out = []
        for arbitration_id, message in can_dict.items():
            pass
            
        
            outstring = (hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
            #print(outstring)
            out.append(outstring)
            #if (hex(arbitration_id) == '0x10240040'):
            #    print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        out.append("") # make sure all msgs are printed, even if odd count     
        i=0
        print("Entries: " + str(len(out)))
        for idx in range(len(out)-1):
           if (idx%2==0):
               print(out[idx].ljust(40) + out[idx+1])
        
        last_out_time = time.time()
