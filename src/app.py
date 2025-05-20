from pathlib import Path
from flask import Flask, request, jsonify

from src.classifier.filename_classifier import FilenameClassifier
from src.types.classifier_input import ClassifierInput
from src.types.classifier_output import ClassifierOutput
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify_file', methods=['POST'])
def classify_file_route():

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed"}), 400
    
    # Instantiate classifier
    classifier = FilenameClassifier()

    # Invoke classifier with input filepath
    input = ClassifierInput(files=Path(file.filename))
    output: ClassifierOutput = classifier.classify(input)

    return jsonify({"file_class": output.output_class.value}), 200


if __name__ == '__main__':
    app.run(debug=True)
