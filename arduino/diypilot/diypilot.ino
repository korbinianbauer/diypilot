#include <TinyGPS++.h>
//#include <SoftwareSerial.h>
//#include <AltSoftSerial.h>
#include <NeoSWSerial.h> 

#define MOT_DIR_PIN 5
#define MOT_STEP_PIN 2

#define GPS_RX_PIN 11
#define GPS_TX_PIN 10
#define GPS_BAUD 9600

TinyGPSPlus gps;
//SoftwareSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);
//AltSoftSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);
NeoSWSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);

String serial_cmds = "";
unsigned long lastValidCmd = 0;
unsigned long safetyTimeout = 1000;


void setup() {
  Serial.begin(115200);
  gpsSerial.begin(GPS_BAUD);
}

void loop() {
  if ((millis() - lastValidCmd) > safetyTimeout){
    setMotSpeed(0);
    Serial.println("Timeout: Motor stopped.");
  }
  Serial.print("{");
  Serial.print("gps_valid:");
  Serial.print(getGpsValid());
  Serial.print(",");

  Serial.print("gps_date:");
  Serial.print(getGpsYear());
  Serial.print("/");
  Serial.print(getGpsMonth());
  Serial.print("/");
  Serial.print(getGpsDay());
  Serial.print(",");

  Serial.print("gps_time:");
  Serial.print(getGpsHour());
  Serial.print(":");
  Serial.print(getGpsMinute());
  Serial.print(":");
  Serial.print(getGpsSecond());
  Serial.print(",");

  Serial.print("gps_lat:");
  Serial.print(getGpsLat(), 6);
  Serial.print(",");
  
  Serial.print("gps_long:");
  Serial.print(getGpsLong(), 6);
  Serial.print(",");

  Serial.print("gps_vel:");
  Serial.println(getGpsSpeed(), 1);
  smartDelay(1000);
}

void setMotSpeed(int mot_speed) {
  digitalWrite(MOT_DIR_PIN, (mot_speed > 0));

  if (abs(mot_speed) < 100){
    noTone(MOT_STEP_PIN);
  }
  tone(MOT_STEP_PIN, abs(mot_speed));
}

boolean getGpsValid() {
  return (gps.location.isValid() && gps.speed.isValid());
}

float getGpsLong() {
  return gps.location.lng();
}
float getGpsLat() {
  return gps.location.lat();
}

float getGpsSpeed() {
  return gps.speed.kmph();
}

int getGpsYear() {
  return gps.date.year();
}
int getGpsMonth() {
  return gps.date.month();
}
int getGpsDay() {
  return gps.date.day();
}
int getGpsHour() {
  return gps.time.hour();
}
int getGpsMinute() {
  return gps.time.minute();
}
float getGpsSecond() {
  return gps.time.second();
}



// This custom version of delay() ensures that the gps object
// is being "fed".
static void smartDelay(unsigned long ms)
{
  unsigned long start = millis();
  do
  {
    while (gpsSerial.available()) {
      gps.encode(gpsSerial.read());
    }


    if (Serial.available() > 0) {
      serial_cmds = Serial.readStringUntil('\n');

      // check if valid data request String received
      if ((String("MOT") == serial_cmds.substring(0, 3)) && (String("END") == serial_cmds.substring(11, 14))) {
        int mot_speed = serial_cmds.substring(4, 10).toInt();
        setMotSpeed(mot_speed);
        lastValidCmd = millis();
        Serial.print("New mot_speed: ");
        Serial.println(mot_speed);
      }
    }

    serial_cmds = "";
  } while (millis() - start < ms);
}
