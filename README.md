# Heron Coding Challenge - File Classifier

## Usage

1. Install dependencies:
    ```shell
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Run the Flask app:
    ```shell
    python -m src.app
    ```

3. Test the classifier using a tool like curl:
    ```shell
    curl -X POST -F 'file=@path_to_pdf.pdf' -F 'file=@path_to_img.jpg' ... http://127.0.0.1:5000/classify_file
    ```

4. Run tests:
   ```shell
    pytest
    ```

5. Run local eval:
    ```shell
    python -m src.local_eval
    ```

## Starting State

The initial classifier had several issues that would make it difficult to scale across use cases:
* It naively classified by filename keywords, which could be inconsistent or manipulated
* It was not written in an object-oriented way that would increase speed of model iteration
* It could only process 1 file at a time, whereas production use cases could often involve batch workflows of classifying entire directories of files
* Classifier test coverage was extremely low, meaning there was high chance that a change could break a (theoretical) production deployment

To address these issues within the ~3hr time constraint, I broke down my work into 4 phases:
1. Refactor code base to make it easier to experiment with classifier implementations
2. Implement each step in the inference pipeline, with high unit test coverage throughout
3. Use a set of open source datasets to collect evaluation accuracy and tune classifier
4. Improve inference endpoint to handle multiple files and account for edge cases during E2E testing

## Process
### Research & Design

The solution builds an inference pipeline to classify the doc's text contents into an invoice, bank statement, or driver's license. I initially considered a Naive Bayes or SVM classifier, which would need 3 stages: a) a feature generation stage to extract the contents of a doc, b) a text2vec embedding model, and c) the classifier itself. However, this would not be very scalable to new doc types as we would need to retrain the model for each supported class. Another approach would be to treat this as an image classification task (since the differences between doc types were more often in layout and structure than text contents), but this would require a training/fine tuning pass unless we used a larger multi-modal LLM that would not fit on my available compute.

Instead, I decided on a zero-shot approach which would let me build a solution within my time/local compute resource constraints. This would shorten the inference pipeline to: a) feature generation, and b) the classifier itself. Later on, given some compute and data, we could fine tune the model to be highly performant while letting us scale faster.

#### Feature Generation (OCR)

The feature generation step uses OCR to extract text from images and PDFs. Since the differences between some of our docs (e.g. driver's licenses) involved doc layout as much as text content, I wanted to ensure that whatever model we used could replicate structure to give the model contextual information. I did research and narrowed down to `pytesseract`, `dOCtR`, and `PyMuPDF`. Experimenting with each on the given examples, I found that while `dOCtR` had the highest OCR accuracy and supported both images and PDFs, it could take up to 1.5s to process each document. Compared to this, `pytesseract` was <1s and had similar results while `PyMuPDF` hovered around 1s. In addition, `PyMuPDF` had an extension, [`PyMuPDF4LLM`](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html), that could return OCR content in markdown format. This would preserve tables and headings, making it easier for a language model to extract spatial information.

I integrated both `pytesseract` and `PyMuPDF4LLM` into the OCR extractor, switching between the two depending on file type. While `PyMuPDF` has an integration with `pytesseract` to handle image OCR it required an intermediate conversion from image to PDF, adding to inference latency.

#### Classifier

For the classifier, I decided on a BERT model as it is an ideal lightweight but performant language model I could use for zero-shot classification. I experimented with a couple models (e.g. `facebook/bart-large-mnli`) and ended up using [`DeBERTav3`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) as it balanced accuracy with latency and model size (only 435M parameters). I also experimented with using Llama3.1 with [Guidance](https://github.com/guidance-ai/guidance?tab=readme-ov-file) to constrain model output to our doc type classes, but even the smallest 8B parameter model ran into compute constraints while running the model locally.

### Implementation

First, I worked to refactor the code base to make it extensible. I defined abstractions for the classifier implementation and created interfaces for I/O. I also reimplemented the `FilenameClassifier` as part of this to ensure the existing functionality was uninterrupted. Next, I created the OCR utils. I kept extensibility in mind to make it easy to switch OCR models under the hood later on, and made the decision to bake in support for batch OCR processing from the beginning. This would prove useful once I extended the Flask app to accept multiple file inputs down the line.

After implementing feature generation, I added the zero-shot classification module. I used `transformers.pipelines` to initialize the model as a zero-shot pipeline, and used the document types given in the examples to constrain model output to invoices, bank statements, driver's licenses, or other. Initially, I defined the "other" class as `UNKNOWN`. However, I found that this led to poor false positive performance during E2E testing so I renamed it to `other`.

Once the inference pipeline was complete, in order to collect testing metrics, I found several open source datasets for the 3 document classes and extracted a balanced subset of ~100 samples each to calculate accuracy:
* [Bank statements](https://www.kaggle.com/datasets/mehaksingal/personal-financial-dataset-for-india)
* [Invoices](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr)
* [Driver's Licenses](https://github.com/lucky-verma/US-Driver-License-data-extraction/tree/master/DATASET/DL%20America/licenses)

While I considered synthetic datasets, I wanted more realistic examples with diverse templates. One shortcoming of these datasets is that they exclusively contain image files, not PDFs, so all PDF testing was through specific examples (such as the ones given or manually provided ones).

I used torch `Dataset`s to create randomly sampling iterators for the datasets and wrote a local eval entrypoint that could calculate the multiclass accuracy. Overall, I found the pipeline had a 92.7% accuracy, which became 89.0% once I tuned the false positive class label from `UNKNOWN` to `other`. Granted, this did not include full FP rate as I only had a couple local examples to test against. Ideally, I would've run more extensive testing with a balanced dataset including FP examples to gauge prod readiness.

Finally, I modified the Flask endpoint to support passing multiple files to classify. This increases the throughput of the API, making it ready for production use cases. While it was implemented iteratively here, the next step would be to profile performance with large batches of inference to see if we can optimize with concurrency. Throughout all changes made to this package, I added unit test coverage to validate behavior and increase resiliency against code bugs entering a production environment.

### Next Steps

With more time and compute, I would use the datasets collected for testing to fine tune the model and increase performance. Another approach, without a training pass, could be to use an LLM like Llama to perform few-shot classification. This could be done by providing a couple examples of each document class to the model before querying it for a classification. Finally, we could try using a text encoder followed by a traditional ML model like an SVM to perform the classification. For a prod-ready classifier, we would need to evaluate at least a few options to catch labelling edge cases before deploying it.
