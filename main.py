from ultralytics import YOLO
from flask import request, Response, Flask, render_template
from PIL import Image
import json

app = Flask(__name__)

@app.route("/")
def root():
    return render_template('index.html')


@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
      json.dumps(boxes),
      mimetype='application/json'
    )


def detect_objects_on_image(buf):
    model = YOLO("track/yolov8m.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

if __name__=='__main__':
    app.run()