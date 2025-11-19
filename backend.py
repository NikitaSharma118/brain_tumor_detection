from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import cv2
import io
import base64
import time

# Initialize Flask app
app = Flask(__name__, static_folder="brain_tumor_site/dist", static_url_path="/")
CORS(app)

# Load model
MODEL_PATH = "brain_tumor_resnet50_final.h5"
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Class labels (modify according to your dataset)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary_tumor']

def get_explanation(pred_class):
    explanations = {
        "glioma": "The highlighted red regions represent areas of irregular, infiltrative growth, commonly seen in gliomas.",
        "meningioma": "The red-orange zone shows dense, well-circumscribed growth near the meninges — characteristic of meningiomas.",
        "pituitary_tumor": "The highlighted central region corresponds to the pituitary gland area, indicating a possible pituitary tumor.",
        "no_tumor": "No abnormal activation was detected. The MRI shows healthy brain tissue with no tumor presence."
    }
    return explanations.get(pred_class.lower(), "Unable to generate explanation.")

# Grad-CAM helper
def generate_gradcam(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = predictions[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html") 

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()

        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image provided."}), 400

        file = request.files["image"]
        img_bytes = file.read()

        # Preprocess
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[0][pred_idx])

        # Grad-CAM
        heatmap = generate_gradcam(img_array, model, last_conv_layer_name="conv5_block3_out")

        # Superimpose heatmap
        img_orig = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_orig = cv2.resize(img_orig, (224, 224))
        heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)
        _, buffer = cv2.imencode('.jpg', overlay)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        # Response
        return jsonify({
            "success": True,
            "predicted_class": pred_class,
            "confidence": f"{confidence*100:.2f}%",
            "gradcam_image": heatmap_base64,
            "analysis_time_seconds": round(time.time() - start_time, 2),
            "explanation" : get_explanation(pred_class)
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
