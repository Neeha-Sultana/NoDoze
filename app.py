from flask import Flask, render_template, request, Response
import cv2
import dlib
import time
#import pygame
import requests
from scipy.spatial import distance
from twilio.rest import Client
import os
import requests
from dotenv import load_dotenv

app = Flask(__name__)


FILE_ID="1_WaKEZwuFDbJ7Hjh4J1wmiuR1CpVHih-"
DEST_PATH = "shape_predictor_68_face_landmarks.dat"

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"✅ {destination} downloaded successfully.")
    else:
        print("❌ Failed to download the file.")

# Download if not already present
if not os.path.exists(DEST_PATH):
    print("Downloading shape predictor model...")
    download_file_from_google_drive(FILE_ID, DEST_PATH)

load_dotenv()
contact_info = {}  # Store emergency contact info
camera = None  # Global variable for the camera

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")


client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Google Geolocation API for accurate location tracking
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"

'''
# Initialize pygame for sound alerts
pygame.mixer.init()
ALERT_SOUND = "song.mp3"  # Ensure this file exists in the project directory
alert_playing = False
alert_sent = False
'''
# Load the face detector and landmark predictor
# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DEST_PATH)


CREATORS = ["Neeha", "Nithya", "Sneha", "Venila"]

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) to detect drowsiness."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_live_location():
    """Fetch real-time location using IP address (IP-based geolocation)."""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()

        if "loc" in data:
            loc = data["loc"].split(",")
            lat, lon = loc[0], loc[1]
            city = data.get("city", "Unknown City")
            state = data.get("region", "Unknown State")
            print(f"📍 Live Location: {city}, {state} - Lat: {lat}, Lon: {lon}")
            return lat, lon, city, state
        else:
            print("❌ No location found!")
            return "Unknown", "Unknown", "Unknown City", "Unknown State"
    except Exception as e:
        print(f"❌ Location error: {e}")
        return "Unknown", "Unknown", "Unknown City", "Unknown State"

def send_alert():
    """Sends an SMS alert with the driver's live location (IP-based)."""
    global alert_sent
    if not alert_sent and "phone" in contact_info:
        recipient_phone = contact_info["phone"]
        lat, lon, city, state = get_live_location()

        if lat != "Unknown":
            message_body = f"🚨 Drowsiness detected! 📍 Location: {city}, {state} (Lat: {lat}, Lon: {lon})"
        else:
            message_body = "🚨 Drowsiness detected! Unable to retrieve location."

        try:
            client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE,
                to=recipient_phone
            )
            print("✅ SMS alert sent successfully!")
            alert_sent = True
        except Exception as e:
            print(f"❌ Error sending SMS: {e}")


def make_voice_call():
    """Makes an automated call to the emergency contact."""
    if contact_info.get("phone"):
        recipient_phone = contact_info["phone"]

        twiml_message = """<Response><Say>Alert! The driver is drowsy. Please check on them.</Say></Response>"""

        try:
            client.calls.create(
                twiml=twiml_message,
                from_=TWILIO_PHONE,
                to=recipient_phone
            )
            print("📞 Voice call placed successfully!")
        except Exception as e:
            print(f"❌ Error making call: {e}")
'''
def play_alert_sound():
    """Play an alert sound."""
    global alert_playing
    if not alert_playing:
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play(-1)
        alert_playing = True


def stop_alert_sound():
    """Stops the alert sound immediately."""
    global alert_playing
    if alert_playing:
        pygame.mixer.music.stop()  # Stop sound
        alert_playing = False  # Reset flag
'''

# Global flag to control video streaming
camera_running = True  
camera_running = False  # Initially, camera is off

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Starts the camera for drowsiness detection."""
    global camera_running
    camera_running = True
    return render_template('drowsy.html', camera_running=camera_running)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stops the camera, detection process, and alert sound."""
    global camera, camera_running, alert_playing
    camera_running = False  # Stop video feed

    # Release the camera
    if camera:
        camera.release()
        camera = None

    # Stop the alert sound if it's playing
    if alert_playing:
        stop_alert_sound()

    return render_template('drowsy.html', camera_running=camera_running)

def generate_frames():
    """Live video feed and drowsiness detection."""
    global camera, alert_sent, camera_running
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase width
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increase height
    drowsy_start_time = None  # Track when drowsiness starts

    while camera_running:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green box around face

            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            if avg_EAR < 0.25:  # Eyes closed
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()  # Start timer when eyes close

                # Check if the person has been drowsy for more than 7 seconds
                if time.time() - drowsy_start_time >= 7:
                    play_alert_sound()
                    send_alert()
                    make_voice_call()
                    cv2.putText(frame, "DROWSY!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:  # Eyes open
                drowsy_start_time = None  # Reset timer
                stop_alert_sound()
                alert_sent = False  # Reset alert flag

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # When stopped, release the camera
    if camera:
        camera.release()
        camera = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html", creators=CREATORS)

@app.route('/drowsy')
def drowsy():
    return render_template("drowsy.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_contact', methods=['POST'])
def save_contact():
    """Stores emergency contact information from the form."""
    global contact_info
    contact_info = {
        "name": request.form['name'],
        "phone": request.form['phone']
    }
    return render_template('index.html', contact=contact_info)

if __name__ == "__main__":
    
    app.run(debug=False)
