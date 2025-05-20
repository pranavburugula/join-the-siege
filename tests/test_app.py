from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.app import app, allowed_file
from src.classifier.filename_classifier import FilenameClassifier
from src.types.document_type import DocumentType
from src.types.classifier_output import ClassifierOutput

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize("filename, expected", [
    ("file.pdf", True),
    ("file.png", True),
    ("file.jpg", True),
    ("file.txt", False),
    ("file", False),
])
def test_allowed_file(filename, expected):
    assert allowed_file(filename) == expected

def test_no_file_in_request(client):
    response = client.post('/classify_file')
    assert response.status_code == 400

def test_no_selected_file(client):
    data = {'file': (BytesIO(b""), '')}  # Empty filename
    response = client.post('/classify_file', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

def test_success(client, mocker):
    mocker.patch(
        'src.app.DEFAULT_CLASSIFIER.classify',
        return_value=ClassifierOutput(
            output_per_file={Path('file.pdf'): DocumentType.DRIVERS_LICENSE}
        )
    )


    data = {'file': (BytesIO(b"dummy content"), 'file.pdf')}
    response = client.post('/classify_file', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.get_json() == {"file_classes": {'file.pdf': 'drivers_license'}}

def test_multiple_files(client, mocker):
    mocker.patch(
        'src.app.DEFAULT_CLASSIFIER.classify',
        return_value=ClassifierOutput(
            output_per_file={
                Path('file1.pdf'): DocumentType.DRIVERS_LICENSE,
                Path('file2.pdf'): DocumentType.BANK_STATEMENT,
            }
        )
    )


    data = {'file': (BytesIO(b"dummy content"), 'file1.pdf'), 'file': (BytesIO(b"dummy content"), 'file2.pdf')}
    response = client.post('/classify_file', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.get_json() == {"file_classes": {'file1.pdf': 'drivers_license', 'file2.pdf': 'bank_statement'}}