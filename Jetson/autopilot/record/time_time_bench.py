import time
import shutil 

start = time.time()
now = None
n = 1000

dummy = 0
for i in range(n):
    stat = shutil.disk_usage("/home/jetson")[-1]/1024/1024/1024

now = time.time()
    
    
print("call took {} us on avg. ({} calls)".format((now-start)/n*1000000, n))
print(stat)
time.sleep(5)
