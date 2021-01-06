import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import datetime
import time
from PIL import Image
from threading import Thread
import os

class AutopilotGUI():
    
    def __init__(self):
        
        self.resolution = None
        self.frame = None

        
        self.show_overlay = False
        self.timestring = "No timestring"
        self.fullscreen = True
        self.gui_scale = 1.0
        
        self.engaged = False
        
        self.blink_left = False
        self.blink_right = False
        self.cruise_control = False
        self.cc_setpoint = 0
        self.velocity = 0
        self.actual_swa = 0
        self.predicted_swa = 0
        self.show_predicted_swa = True
        self.time_string = ""
        self.fps = 0
        self.freespace = 0
        self.recording = False
        
        self.frame_out = None
        self.window_rendering_stopped = True
        self.updated = True
        self.last_update = 0 # timestamp
        self.last_render_timestamp = 0
        
        self.gui_thread = None
        
        package_directory = os.path.dirname(os.path.abspath(__file__))
        
        engaged_border_file = font_file = os.path.join(package_directory, 'icons', 'engaged_border.png')
        engaged_border = cv2.imread(engaged_border_file, cv2.IMREAD_UNCHANGED)
        self.engaged_border = cv2.cvtColor(engaged_border, cv2.COLOR_BGRA2RGBA)
        
        engaged_border_file = font_file = os.path.join(package_directory, 'icons', 'indicator_arrow_right.png')
        indicator_icon = cv2.imread(engaged_border_file, cv2.IMREAD_UNCHANGED)
        self.right_indicator_icon = cv2.cvtColor(indicator_icon, cv2.COLOR_BGRA2RGBA)
    
        self.left_indicator_icon = cv2.flip(self.right_indicator_icon, 1)
        
        engaged_border_file = font_file = os.path.join(package_directory, 'icons', 'steering_wheel.png')
        steering_wheel_icon = cv2.imread(engaged_border_file, cv2.IMREAD_UNCHANGED)
        self.steering_wheel_icon = cv2.cvtColor(steering_wheel_icon, cv2.COLOR_BGRA2RGBA)
        
        engaged_border_file = font_file = os.path.join(package_directory, 'icons', 'steering_wheel_predict_indicator.png')
        steering_wheel_predict_icon = cv2.imread(engaged_border_file, cv2.IMREAD_UNCHANGED)
        self.steering_wheel_predict_icon = cv2.cvtColor(steering_wheel_predict_icon, cv2.COLOR_BGRA2RGBA)
        
        engaged_border_file = font_file = os.path.join(package_directory, 'icons', 'cruise_control_icon.png')
        cruise_control_icon = cv2.imread(engaged_border_file, cv2.IMREAD_UNCHANGED)
        self.cruise_control_icon = cv2.cvtColor(cruise_control_icon, cv2.COLOR_BGRA2RGBA)

        
        
    def set_frame(self, frame):
        self.frame = frame
        height, width, channels = frame.shape
        self.resolution = [width, height]
        self.set_updated(True)
        
    def set_updated(self, updated):
        self.updated = updated
        if updated:
            self.last_update = time.time()
    
    def get_updated(self):
        return self.updated
    
    def get_last_update(self):
        return self.last_update
        
    def set_show_overlay(self, show_overlay):
        self.show_overlay = show_overlay
        self.set_updated(True)
    
    def get_show_overlay(self):
        return self.show_overlay
    
    def set_timestring(self, timestring):
        self.timestring = timestring
        self.set_updated(True)
        
    def get_timestring(self):
        return self.timestring
        
    def get_frame(self):
        return self.frame
    
    def get_resolution(self):
        return self.resolution
    
    def set_fps(self, fps):
        self.fps = fps
    
    def get_fps(self):
        return self.fps
    
    def set_engaged(self, engaged):
        self.engaged = engaged
        self.set_updated(True)
    
    def get_engaged(self):
        return self.engaged
    
    def set_recording(self, recording):
        self.recording = recording
        self.set_updated(True)
    
    def get_recording(self):
        return self.recording
    
    def set_indicator_left(self, indicator_left):
        self.blink_left = indicator_left
        self.set_updated(True)
    
    def get_indicator_left(self):
        return self.blink_left
    
    def set_indicator_right(self, indicator_right):
        self.blink_right = indicator_right
        self.set_updated(True)
    
    def get_indicator_right(self):
        return self.blink_right
    
    def set_cruise_control(self, cruise_control):
        self.cruise_control = cruise_control
        self.set_updated(True)
    
    def get_cruise_control(self):
        return self.cruise_control
    
    def set_cruise_control_setpoint(self, cruise_control_setpoint):
        self.cruise_control_setpoint = cruise_control_setpoint
        self.set_updated(True)
    
    def get_cruise_control_setpoint(self):
        return self.cruise_control_setpoint
    
    def set_actual_swa(self, actual_swa):
        self.actual_swa = actual_swa
        self.set_updated(True)
    
    def get_actual_swa(self):
        return self.actual_swa
    
    def set_show_predicted_swa(self, show_predicted_swa):
        self.show_predicted_swa = show_predicted_swa
        self.set_updated(True)
        
    def get_show_predicted_swa(self):
        return self.show_predicted_swa
    
    def set_predicted_swa(self, predicted_swa):
        self.predicted_swa = predicted_swa
        self.set_updated(True)
    
    def get_predicted_swa(self):
        return self.predicted_swa
    
    def set_velocity(self, velocity):
        self.velocity = velocity
        self.set_updated(True)
    
    def get_velocity(self):
        return self.velocity
    
    def set_freespace(self, freespace):
        self.freespace = freespace
        self.set_updated(True)
    
    def get_freespace(self):
        return self.freespace
    
    def get_dummy_frame(self):
        dummy_frame = np.zeros((480,848,3), np.uint8)
        self.render_text(dummy_frame, "No Input", rel_pos=(0.5, 0.5), size=3, text_color = (255,255,255),
                    shadow_color=(0,0,0), hor_align = 0)
        return dummy_frame
        
    def render(self):
        
        if self.get_frame() is None:
            self.set_frame(self.get_dummy_frame())
            
        frame_out = self.get_frame().copy()
        
        if self.get_engaged():
            self.render_engaged_border(frame_out)
        
        if not self.get_show_overlay():
            self.frame_out = frame_out
            return self.frame_out
        
        # Render overlay items
        self.render_time(frame_out)
        if self.get_recording():
            self.render_rec(frame_out)
        self.render_fps(frame_out)
        self.render_velocity(frame_out)
        self.render_freespace(frame_out)
        if self.get_indicator_left():
            self.render_left_indicator(frame_out)
        if self.get_indicator_right():
            self.render_right_indicator(frame_out)
        if self.get_cruise_control():
            self.render_cruise_control(frame_out)
            
        if self.get_show_predicted_swa():
            self.render_predicted_swa_indicator(frame_out)
            self.render_predicted_path(frame_out)
            
        self.render_steering_wheel(frame_out)
        self.render_actual_swa(frame_out)
        self.render_actual_path(frame_out)
            
        
        
        
        
        self.frame_out = frame_out
        return self.frame_out
    
    def render_text(self, frame, text, rel_pos, size, text_color = (255,255,255), shadow_color=(0,0,0), hor_align = 0):
        # setup text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        height, width, channels = frame.shape
        size = width * size/1000
        
        shadow_thickness = int(size*8)
        text_thickness = int(size*3)

        # get boundaries of text
        shadow_bounds = cv2.getTextSize(text, font, size, shadow_thickness)[0]
        text_bounds = cv2.getTextSize(text, font, size, text_thickness)[0]

        # get coords based on boundary
        if hor_align == -1: # left
            shadow_x = int(frame.shape[1] * rel_pos[0])
        elif hor_align == 0: # center
            shadow_x = int((frame.shape[1] * rel_pos[0]) - shadow_bounds[0]/2)
        else: # right
            shadow_x = int((frame.shape[1]  * rel_pos[0]) - shadow_bounds[0])
            
        shadow_y = int((frame.shape[0] * rel_pos[1]) + shadow_bounds[1]/2)

        
        text_x = shadow_x #+ (shadow_bounds[0] - text_bounds[0])//2
        text_y = shadow_y #+ (shadow_bounds[1] - text_bounds[1])//2

        # add text centered on image
        cv2.putText(frame, text, (shadow_x, shadow_y ), font, size, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y ), font, size, text_color, text_thickness, cv2.LINE_AA)
        
    
    def render_time(self, frame):
        text = self.get_timestring()
        size = 0.8
        rel_pos = (0.5, 0.04)
        self.render_text(frame, text, rel_pos, size)
        
    def render_rec(self, frame):
        text = "REC"
        text_color = (255,0,0)
        shadow_color = (0,0,0)
        size = 1.2
        rel_pos = (0.05, 0.05)
        self.render_text(frame, text, rel_pos, size, text_color, shadow_color)
        
    def render_fps(self, frame):
        text = str(round(self.get_fps(), 1)) + " fps (GUI)"
        size = 0.8
        rel_pos = (1.0, 0.1)
        self.render_text(frame, text, rel_pos, size, hor_align = 1)
        
    def render_actual_swa(self, frame):
        text = str(round(self.get_actual_swa(), 1)) + " deg"
        size = 0.8
        rel_pos = (0.9, 0.95)
        self.render_text(frame, text, rel_pos, size, hor_align = 0)
        
    
        
    def render_velocity(self, frame):
        text = str(round(self.get_velocity(), 1)) + " km/h"
        size = 0.8
        rel_pos = (0.17, 0.95)
        self.render_text(frame, text, rel_pos, size, hor_align = 1)
        
    def render_freespace(self, frame):
        text = str(round(self.get_freespace(), 2)) + " GB free"
        size = 0.8
        rel_pos = (1.0, 0.04)
        self.render_text(frame, text, rel_pos, size, hor_align = 1)
    
    def render_engaged_border(self, frame):
        position = (0.5, 0.05)
        size = (1,0.1)
        overlay = self.engaged_border
        self.overlay_transparent(frame, overlay, position, size)
        
    def render_left_indicator(self, frame):
        position = (0.25, 0.07)
        size = (0.05,0.1)
        overlay = self.left_indicator_icon
        self.overlay_transparent(frame, overlay, position, size)
        
    def render_right_indicator(self, frame):
        position = (0.75, 0.07)
        size = (0.05,0.1)
        overlay = self.right_indicator_icon
        self.overlay_transparent(frame, overlay, position, size)
        
    def render_steering_wheel(self, frame):
        position = (0.9,0.75)
        size = (0.2,None)
        overlay = self.steering_wheel_icon.copy()
        overlay = self.rotate_image(overlay, self.get_actual_swa())
        self.overlay_transparent(frame, overlay, position, size)
        
    def render_predicted_swa_indicator(self, frame):
        position = (0.9,0.75)
        size = (0.2,None)
        overlay = self.steering_wheel_predict_icon.copy()
        overlay = self.rotate_image(overlay, self.get_predicted_swa())
        self.overlay_transparent(frame, overlay, position, size)
        
    def render_cruise_control(self, frame):
        icon_position = (0.06,0.8)
        icon_size = (0.1,None)
        overlay = self.cruise_control_icon.copy()
        self.overlay_transparent(frame, overlay, icon_position, icon_size)
        
        #text = str(round(self.get_cc_setpoint()))
        text = str(round(self.get_cruise_control_setpoint()))
        size = 1.5
        rel_pos = (0.22, 0.82)
        text_color = (254,156,0)
        shadow_color = (0,0,0)
        self.render_text(frame, text, rel_pos, size, text_color, shadow_color, hor_align = 1)
        
    def render_path(self, frame, swa, color):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
         
        hor_axis = int(abs(frame_width*0.002*swa)) #0.002
        vert_axis = max(0, int(frame_height*0.5 - 0.2*hor_axis)) # - 0.5*...
        
        x_center_offset = -16 # pixels, due to off center mounting of camera
        
        center_coordinates = (int(frame_width*0.5 + x_center_offset - hor_axis*np.sign(swa)), int(frame_height*1.0))
        axesLength = (hor_axis, vert_axis) 
        angle = 0
        
        if (swa<0):
            startAngle = 180
        else:
            startAngle = 360
        endAngle = 270


        # Line thickness of 5 px 
        thickness = frame_width//300

        # Using cv2.ellipse() method 
        # Draw a ellipse with red line borders of thickness of 5 px 
        image = cv2.ellipse(frame, center_coordinates, axesLength, 
                   angle, startAngle, endAngle, color, thickness, cv2.LINE_AA) 
        
        
    def render_actual_path(self, frame):
        swa = self.get_actual_swa()
        color = (255, 0, 0) 
        self.render_path(frame, swa, color)
        
    def render_predicted_path(self, frame):
        swa = self.get_predicted_swa()
        color = (0, 0, 255) 
        self.render_path(frame, swa, color)
        
        
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def overlay_transparent(self, background, overlay, rel_pos, rel_size):

        background_width = background.shape[1]
        background_height = background.shape[0]
        overlay_width = overlay.shape[1]
        overlay_height = overlay.shape[0]
        
        x_size = rel_size[0] * background_width
        if rel_size[1] == None: # equal scale
            y_size = overlay_height / overlay_width * x_size
        else:
            y_size = rel_size[1] * background_height
            
        size = (int(x_size), int(y_size))
            
        overlay = cv2.resize(overlay, size, interpolation = cv2.INTER_AREA)
        overlay_width = overlay.shape[1]
        overlay_height = overlay.shape[0]
        
        
        # get coords based on boundary
        x = int((background_width * rel_pos[0]) - overlay_width/2)
        y = int((background_height * rel_pos[1]) - overlay_height/2)

        if x >= background_width or y >= background_height:
            return background

        h, w = overlay.shape[0], overlay.shape[1]

        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]

        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
                ],
                axis = 2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

        return background


    
    def get_rendered_gui(self):
        return self.frame_out
    
    def show_window(self):
        if self.window_rendering_stopped == False:
            return
        else:
            self.window_rendering_stopped = False
            #cv2.namedWindow("Autopilot GUI", cv2.WINDOW_FREERATIO)
            #cv2.setWindowProperty("Autopilot GUI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.gui_thread = Thread(target=self.render_window, args=())
            self.gui_thread.start()
        return self

    def render_window(self):
        while not self.window_rendering_stopped:
            last_last_render_timestamp = self.last_render_timestamp
            self.last_render_timestamp = time.time()
            self.set_fps(1/(self.last_render_timestamp - last_last_render_timestamp))
            start_time = time.time()
            self.render()
            end_time = time.time()
            #print("Render duration: " + str(round((end_time - start_time)*1000, 2)) + "ms")
            gui_frame = self.get_rendered_gui()
            gui_frame_bgr = cv2.cvtColor(gui_frame, cv2.COLOR_RGB2BGR)
            self.set_updated(False)
            
            cv2.imshow("Autopilot GUI", gui_frame_bgr)
            
            while not self.get_updated():
                if cv2.waitKey(1) == ord("q"):
                    self.window_rendering_stopped = True
                    
                if (time.time() - self.get_last_update()) > 1:
                    # detect loss of video
                    self.set_frame(self.get_dummy_frame())
                    
                
    def stop_window(self):
        self.window_rendering_stopped = True
        self.gui_thread.join()
        cv2.destroyAllWindows()
    

    def show_jupyter(self):
        
        mpl.rcParams['figure.figsize'] = [12, 12]
        mpl.rcParams['figure.dpi'] = 72
        start_time = time.time()
        self.render()
        end_time = time.time()
        print("Render duration: " + str(end_time - start_time))
        plt.imshow(self.get_rendered_gui()/255)
        fig = plt.figure
        display(fig)
