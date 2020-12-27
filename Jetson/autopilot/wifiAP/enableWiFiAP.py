import subprocess
import os
import time

while True:
    time.sleep(1)
    result = subprocess.run(['lsusb'], stdout=subprocess.PIPE)
    print(result.stdout)
    
    if ("Realtek RTL8188CUS" in str(result.stdout)):
        time.sleep(3)
        os.system("sudo create_ap wlan0 eth0 'Jetson-AP' 'qweasdyxc' --freq-band 2.4 --no-virt -w 2")
        
        break