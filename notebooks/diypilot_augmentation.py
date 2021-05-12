from math import sin, cos, sqrt
import sys
import time
import random
import numpy as np
import cv2

def horizontal_flip_augmentation(frame, orig_swa):
    frame = cv2.flip(frame, 1)
    new_swa = -orig_swa
    
    return frame, new_swa


def horizontal_rotation_augmentation(frame, orig_swa, frame_hor_fov_deg, v_vehicle, time_to_recover, turn_radius, max_swa, hor_rotate_degrees, verbose=False):
    
    def horizontal_rotate_camera(frame, horizontal_rotate_pixels):
        
        frame_height, frame_width, frame_channels = frame.shape
        M = np.float32([[1,0,horizontal_rotate_pixels], [0,1,0]])
        frame = cv2.warpAffine(frame,M,(frame_width, frame_height))
        
        return frame
        
    def pixels_for_hor_rotate(frame, frame_hor_fov_deg, hor_rotate_degrees):
        
        frame_height, frame_width, frame_channels = frame.shape
        px = frame_width * hor_rotate_degrees/frame_hor_fov_deg
        return px
    
    def swa_for_hor_rotate(orig_swa, v_vehicle, time_to_recover, turn_radius, max_swa, hor_rotate_degrees, verbose):
        
        if abs(orig_swa) > 0.1:
            r0 = -turn_radius * max_swa/orig_swa
            w0 = v_vehicle / r0
        else:
            #verbose = True
            r0 = float("inf")
            w0 = 0
        
        hor_rotate_rad = hor_rotate_degrees/180*3.141592
        w_aug = hor_rotate_rad / time_to_recover
        
        if v_vehicle < 0.1: # avoid div by r1=zero error
            v_vehicle = 0.1
        
        w_new = w0 + w_aug
        r1 = v_vehicle / w_new
        swa_new = max_swa * -turn_radius / r1
        
        if verbose:
            print()
            print("horizontal_rotation_augmentation:")
            print("V= " + str(v_vehicle) + " km/h")
            print("Center SWA: " + str(orig_swa) + " deg")
            print()
            print("orig curve radius: " + str(r0) + " m")
            print("orig curve center: " + str((r0, 0)))
            print()
            print("hor_rotate: " + str(hor_rotate_degrees) + " deg")
            print()
            print("new curve radius: " + str(r1) + " m")
            print()
            print("New SWA: " + str(swa_new) + " deg")
        
        return swa_new

    new_swa = swa_for_hor_rotate(orig_swa, v_vehicle, time_to_recover, turn_radius, max_swa, hor_rotate_degrees, verbose)
    
    horizontal_rotate_pixels = pixels_for_hor_rotate(frame, frame_hor_fov_deg, hor_rotate_degrees)
    frame = horizontal_rotate_camera(frame, horizontal_rotate_pixels)

    return frame, new_swa



def lateral_shift_augmentation(frame, orig_swa, frame_physical_width, v_vehicle, time_to_center, turn_radius, max_swa, lateral_shift, horizon_height, verbose=False):
    
    '''
    frame: Input frame
    frame_physical_width: Breite eines Objektes auf Bodenhöhe dass das Frame am unteren Rand ausfüllt in Meter
    lateral_shift: Seitliche Abweichung zur Trajectorie in Meter
    v_vehicle: Fahrzeuggeschwindigkeit in km/h
    orig_swa: Benötigter/Aufgezeichneter Lenkradwinkel bei 0 lateral shift in Grad, positive Werte -> Linkskurve
    lateral_shift: Seitliche Abweichung zur Trajectorie in Meter
    time_to_center: Zeit bis zur Rückkehr zur Trajectorie in Sekunden
    turn_radius: Wenderadius in Metern bei maximalem Lenkradeinschlag max_swa in Grad
    max_swa: Maximaler Lenkradwinkel in Grad
    verbose: Debug-prints() (Boolean)
    '''

    def lateral_shift_frame(frame, lateral_shift_pixels, horizon_height):
        
        '''
        horizon_height: the height of horizon within the frame, in pixel, from top
        '''
   
        result = frame.copy()
        
        # Locate points of the documents or object which you want to transform
        # source:
        frame_height, frame_width, frame_channels = frame.shape
        src_pt1 = [0, horizon_height]
        src_pt2 = [frame_width, horizon_height]
        src_pt3 = [0 + lateral_shift_pixels, frame_height]
        src_pt4 = [frame_width + lateral_shift_pixels, frame_height]
        
        pts1 = np.float32([src_pt1, src_pt2, src_pt3, src_pt4]) # source
        
        pts2 = np.float32([[0, horizon_height], [frame_width, horizon_height], [0, frame_height], [frame_width, frame_height]]) # target

        # Apply Perspective Transform Algorithm 
        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        warped = cv2.warpPerspective(frame, matrix, (frame_width, frame_height), borderMode=cv2.BORDER_WRAP)
        
        # combine unwarped top part (above horizon) and warped bottom part (below horizon)
        
        result[horizon_height:, :] = warped[horizon_height:, :]
        
        return result

    def pixels_for_lateral_shift(frame_width, frame_physical_width, lateral_shift):

        '''
        Ausgangssituation: Fahrzeug folgt einer Trajectorie
        Fahrzeug ist nun um lateral_shift Meter von der Trajectorie abgekommen.
        Fragestellung: Um wieviele Pixel verschiebt sich der untere Rand des ROI durch die laterale Verschiebung

        frame_width: Breite des Frames in Pixeln
        frame_physical_width: Breite eines Objektes auf Bodenhöhe dass die ROI am unteren Rand ausfüllt
        lateral_shift: Seitliche Abweichung zur Trajectorie in Meter
        '''

        pixels = frame_width * lateral_shift / frame_physical_width
        return pixels

    def swa_for_lateral_shift(v_vehicle, orig_swa, lateral_shift, time_to_center, turn_radius, max_swa, verbose = False):

        '''
        Ausgangssituation: Fahrzeug folgt mit v_vehicle km/h einer Trajectorie durch orig_swa Grad Lenkradwinkel
        Fahrzeug ist nun um lateral_shift Meter von der Trajectorie abgekommen.
        Fragestellung: Welcher Lenkradwinkel new_swa Grad muss nun stattdessen aufgebracht werden,
        um innerhalb von time_to_center Sekunden  wieder auf die Trajectorie zurückzukommen?

        v_vehicle: Fahrzeuggeschwindigkeit in km/h
        orig_swa: Benötigter Lenkradwinkel bei 0 lateral shift in Grad, positive Werte -> Linkskurve
        lateral_shift: Seitliche Abweichung zur Trajectorie in Meter
        time_to_center: Zeit bis zur Rückkehr zur Trajekctorie in Sekunden
        turn_radius: Wenderadius in Metern bei maximalem Lenkradeinschlag max_swa in Grad
        max_swa: Maximaler Lenkradwinkel in Grad

        verbose: Debug-prints() (Boolean)

        '''

        starttime = time.time()

        v = v_vehicle / 3.6 # m/s

        if abs(orig_swa) > 0.1:
            r0 = max_swa/orig_swa * -turn_radius # m
            w0 = v/r0 # rad/s
            alpha = w0 * time_to_center # rad

            #Ziel
            Zx = r0 * (1-cos(alpha)) # m
            Zy = r0 * sin(alpha) # m


        else:
            #verbose = True
            r0 = float("inf")
            w0 = 0

            #Ziel
            Zx = 0
            Zy = v * time_to_center

        Z = (Zx, Zy)

        # Start
        Sx = lateral_shift # m
        Sy = 0 # m
        S = (Sx, Sy)

        def dist(x1, y1, x2, y2):
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            return sqrt(dx**2 + dy**2)

        x_min = -99999999.0
        x_max = 99999999.0
        x = 0

        iters = 0
        if Zx > Sx:
            while (x_max - x_min) > 0.001:
                x = (x_min + x_max)/2
                dist_start = dist(Sx, Sy, x, 0)
                dist_ziel = dist(Zx, Zy, x, 0)

                if dist_start < dist_ziel:
                    x_min = x
                else:
                    x_max = x
                iters += 1

                if (iters > 1000):
                    print("Max iterations exceeded")
                    break;

        else:
            while (x_max - x_min) > 0.001:
                x = (x_min + x_max)/2
                dist_start = dist(Sx, Sy, x, 0)
                dist_ziel = dist(Zx, Zy, x, 0)

                if dist_start < dist_ziel:
                    x_max = x
                else:
                    x_min = x
                iters += 1

                if (iters > 1000):
                    print("Max iterations exceeded")
                    break;

        M = (x, 0)
        r1 = x - Sx
        a1 = v**2/r1 # Zentrpetalbeschleunigung
        new_swa = max_swa * (-turn_radius / r1)

        endtime = time.time()

        if verbose:
            print()
            print("lateral_shift_augmentation:")
            print("V= " + str(v_vehicle) + " km/h")
            print("Center SWA: " + str(orig_swa) + " deg")
            print()
            print("orig curve radius: " + str(r0) + " m")
            print("orig curve center: " + str((r0, 0)))
            print()
            print("Ziel: " + str(Z))
            print()
            print("Lateral shift: " + str(lateral_shift) + " m")
            print("Start: " + str(S))
            print()
            print("new curve radius: " + str(r1) + " m")
            print("new curve center: " + str(M))
            print()
            print("New SWA: " + str(new_swa) + " deg")
            print()
            print("Iterations: " + str(iters))
            print("Took: " + str(1000.0*(endtime-starttime)) + " ms")

        return new_swa
    
    new_swa = swa_for_lateral_shift(v_vehicle, orig_swa, lateral_shift, time_to_center, turn_radius, max_swa, verbose)
    
    frame_height, frame_width, frame_channels = frame.shape
    px_shift = pixels_for_lateral_shift(frame_width, frame_physical_width, lateral_shift)
    frame = lateral_shift_frame(frame, px_shift, horizon_height)
    
    return frame, new_swa

if __name__ == '__main__()':
    
    d = 12.8 # m
    b = 1.8 # m
    turn_radius = (d-b)/2
    max_swa = 500 # deg

    time_to_center = 2 # s
    orig_swa = 10 # deg
    v_vehicle = 20 # kmh

    lateral_shift = -0.5 # m

    swa_for_lateral_shift(v_vehicle, orig_swa, lateral_shift, time_to_center, turn_radius, max_swa)


    speeds = []
    swas = []
    diff_swas = []
    accels = []
    for i in range(10000):
        v_vehicle = random.uniform(25, 220)
        orig_swa = random.uniform(0, 0)
        lateral_shift = random.uniform(-1, 1)
        time_to_center = 2
        turn_radius = 5
        max_swa = 500

        new_swa, accel = swa_for_lateral_shift(v_vehicle, orig_swa, lateral_shift, time_to_center, turn_radius, max_swa)

        speeds.append(v_vehicle)
        diff_swas.append(orig_swa-new_swa)
        swas.append(new_swa)
        accels.append(accel)


    import matplotlib.pyplot as plt



    # Plot
    plt.scatter(speeds, diff_swas, s=0.1, color = 'blue')
    #plt.scatter(speeds, swas, s=0.1, color = 'red')
    #plt.scatter(speeds, accels, s=0.1, color = 'red')
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('km/h')
    plt.ylabel('SWA in deg')
    plt.grid(True)
    plt.show()
