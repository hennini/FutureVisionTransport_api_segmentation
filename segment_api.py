from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Dummy segmentation model that returns a red image
def dummy_segmentation_model(img):
    red_image = np.zeros_like(img)
    if len(img.shape) == 3 and img.shape[2] == 3:  # vérifier si l'image a bien 3 canaux (RGB)
        red_image[:, :, 0] = 255  # Canal rouge à 255
        red_image[:, :, 1] = 0    # Canal vert à 0
        red_image[:, :, 2] = 0    # Canal bleu à 0
    return red_image

@app.route('/segment', methods=['POST'])  # Autoriser uniquement la méthode POST
def segment_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Lire l'image envoyée via la requête POST
    img = Image.open(file.stream)
    img = np.array(img)

    # Appliquer la fonction de segmentation (ici, on retourne une image rouge)
    segmented_img = dummy_segmentation_model(img)

    # Convertir l'image segmentée en un fichier compatible avec une réponse Flask
    segmented_pil_img = Image.fromarray(segmented_img.astype('uint8'))
    img_io = BytesIO()
    segmented_pil_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
