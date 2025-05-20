from pathlib import Path
from tempfile import TemporaryDirectory
from flask import Flask, request, jsonify

from src.classifier.zero_shot_classifier import ZeroShotClassifier
from src.types.classifier_input import ClassifierInput
from src.types.classifier_output import ClassifierOutput
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg'}

DEFAULT_CLASSIFIER = ZeroShotClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify_file', methods=['POST'])
def classify_file_route():

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    with TemporaryDirectory() as temp_dir:
        files = request.files.getlist('file')
        for file in files:
            if not file.filename:
                return jsonify({"error": "No selected file"}), 400

            if not allowed_file(file.filename):
                return jsonify({"error": f"File type not allowed"}), 400
            
            # In order to pass filepaths through the system instead of file objs,
            # we save the file to a local tmp dir for processing
            file.save(Path(temp_dir) / file.filename)

        # Invoke classifier with input filepath
        input = ClassifierInput(files=None, dir_path=Path(temp_dir))
        output: ClassifierOutput = DEFAULT_CLASSIFIER.classify(input)

    return jsonify({"file_classes": {str(filename): result_class.value for filename, result_class in output.output_per_file.items()}}), 200


if __name__ == '__main__':
    app.run(debug=True)
