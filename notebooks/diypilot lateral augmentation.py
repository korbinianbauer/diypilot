from math import sin, cos, sqrt
import sys
import time
import random

d = 12.8 # m
b = 1.8 # m
turn_radius = (d-b)/2
max_swa = 500 # deg

time_to_center = 2 # s
orig_swa = 10 # deg
v_vehicle = 20 # kmh

lateral_shift = -0.5 # m


def swa_for_lateral_shift(v_vehicle, orig_swa, lateral_shift, time_to_center, turn_radius, max_swa, verbose = False):

    '''
    Ausgangssituation: Fahrzeug folgt mit v_vehicle km/h einer Trajectorie durch orig_swa Grad Lenkradwinkel
    Fahrzeug ist nun um lateral_shift Meter von der Trajectorie abgekommen.
    Fragestellung: Welcher Lenkradwinkel new_swa Grad muss nun stattdessen aufgebracht werden,
    um innerhalb von time_to_center Sekunden  wieder auf die Trajectorie zurückzukommen?

    v_vehicle: Fahrzeuggeschwindigkeit in km/h
    orig_swa: Benötigter Lenkradwinkel bei 0 lateral shift in Grad
    lateral_shift: Seitliche Abweichung zur Trajectorie in Meter
    time_to_center: Zeit bis zur Rückkehr zur Trajekctorie in Sekunden
    turn_radius: Wenderadius in Metern bei maximalem Lenkradeinschlag max_swa in Grad
    max_swa: Maximaler Lenkradwinkel in Grad, positive Werte -> Rechtkurve

    verbose: Debug-prints() (Boolean)

    '''

    starttime = time.time()

    v = v_vehicle / 3.6 # m/s
    
    if abs(orig_swa) > 0.001:
        r0 = max_swa/orig_swa * turn_radius # m
        w0 = v/r0 # rad/s
        alpha = w0 * time_to_center # rad

        #Ziel
        Zx = r0 * (1-cos(alpha)) # m
        Zy = r0 * sin(alpha) # m
        

    else:
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
    new_swa = max_swa * (turn_radius / r1)

    endtime = time.time()

    if verbose:
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

    return new_swa, a1

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
