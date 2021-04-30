import pyrealsense2 as rs
import numpy as np
import time
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
print(profile)


ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    print(dev)
    dev.hardware_reset()

