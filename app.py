from flask import Flask, render_template,Response
import cv2
from deepface import DeepFace
import csv
from datetime import datetime
from collections import deque, Counter
import pandas as pd 
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(face_cascade)


emotion_window = deque(maxlen=7)  # last 7 predictions
last_logged_emotion = None        # avoid duplicates
def gen_frames():
    global last_logged_emotion
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
                for face in results:
                    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                    emotion = face['dominant_emotion']
            
                    # Append to sliding window
                    emotion_window.append(emotion)
            
                    # Count most common emotion in last 7 frames
                    if len(emotion_window) == emotion_window.maxlen:
                        most_common_emotion, count = Counter(emotion_window).most_common(1)[0]
            
                        # Log only if emotion occurs in >= 5 of last 7 frames and it's not the same as last logged
                        if count >= 5 and most_common_emotion != last_logged_emotion:
                            with open("emotion_log.csv", mode="a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), most_common_emotion])
                                print("Logged:", most_common_emotion)
            
                            last_logged_emotion = most_common_emotion
            
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Draw emotion with voting info
                    label_text = emotion

                    # Show voting status if window is full
                    if len(emotion_window) == emotion_window.maxlen:
                        most_common_emotion, count = Counter(emotion_window).most_common(1)[0]
                        label_text = f"{most_common_emotion} ({count}/7)"

                    cv2.putText(frame, label_text, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    
                    # Draw a horizontal progress bar below the face box
                    if len(emotion_window) == emotion_window.maxlen:
                        bar_x1, bar_y1 = x, y + h + 10
                        bar_x2 = x + w
                        bar_height = 10

                        # Calculate bar fill based on vote count (out of 7)
                        bar_fill = int((count / emotion_window.maxlen) * (bar_x2 - bar_x1))

                        # Draw background bar (gray)
                        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + bar_height), (200, 200, 200), -1)

                        # Draw filled part (green if confident, red if low)
                        color = (0, 255, 0) if count >= 5 else (0, 0, 255)
                        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_fill, bar_y1 + bar_height), color, -1)
            except Exception as e:
                print("Emotion detection error:", e)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/dashboard')
def dashboard():
    try:
        df = pd.read_csv("emotion_log.csv", header=None, names=["timestamp", "emotion"])

        # Pie chart data
        emotion_counts = df['emotion'].value_counts()

        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio for circle

        # Save chart to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return render_template('dashboard.html', chart_data=encoded)

    except Exception as e:
        return f"Dashboard error: {e}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)