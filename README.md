clinical-trials-ocr-idp  

🩺 **Automating clinical trial questionnaire answers using OCR, embeddings, and LLMs.**  

>*WARNING*: Due to the anonymization of the document, some of the text might be missing. It is not due to the OCR/extractor performances.

## 🚀 Introduction : 
Recent technological advances in the field of AI have allowed for the systematic treatment of unformatted documents or images. This project aims to respond to clinical trial requirement questionnaires using IDP with VLM and LLM models on any possible medical document. This task usually requires a lot of a doctor's very valuable time to get this done. This required constant trial and error with different models. 

## 🔧 Tech Stack 
- **OCR**: layoutparser, Tesseract-OCR, PaddleOCR, EasyOCR, Amazon Textract, llama-4-scout-17b-16e-instruct, llama-4-maverick-17b-128e-instruct, Qwen2.5-VL-72B-Instruct, Mistral OCR, ChatGPT 4.1 Vision, Azure AI Document Intelligence, Google Cloud Document AI, Google Cloud Vision, AWS Textract, Google Vertex AI.
  
- **Embeddings**: Sentence-Transformers, Universal Sentence Encoder, Gensim, spaCy, Mistral Embed, Meta Llama-3.3-70B-Versatile, OpenAI text-embedding-3-small, OpenAI text-embedding-3-large, gemini-embedding-exp-03-07

- **Named Entity Recognition**: spaCy, deepseek-r1-distill-llama-70b, Google gemma2-9b-it, ChatGPT 4.1
  
- **LLM**: Meta Llama, Mistral, Qwen, Google Gemini 2.5 Flash
  
- **Backend**: Python (PyTorch/TensorFlow)

- **Cloud Services**: AWS, Google Cloud, Azure, Groq, Huggingface, Mistral

# Research on embedding and similarity search on Python for eCRF 

# Index

[1. Research on embedding and similarity search on Python for eCRF](#research-on-embedding-and-similarity-search-on-python-for-ecrf)  
[2. REQUIREMENTS](#requirements)  
[3. Task 1A: Text extraction from a PDF](#task-1a-text-extraction-from-a-pdf)  
[4. Task 1B: PDF to image + OCR on the image.](#task-1b-pdf-to-image--ocr-on-the-image)  
[5. Task 2: Embedding for a similarity search using AI](#task-2-embedding-for-a-similarity-search-using-ai)  
[6. Task 3: Get the questionnaire from Milo's API](#task-3-get-the-questionnaire-from-milos-api)  
[7. References & Sources][def]  


[def]: #references--sources
## REQUIREMENTS

- Ideally, install Bash. 
```bash
brew install bash #MacOS
https://www.howtogeek.com/790062/how-to-install-bash-on-windows-11/ #Windows
```
- The one from Windows is a tutorial, you will need to use a virtual machine. You could also do the installation on PowerShell but some commands don't even exist for it. 

- Should work for most previous versions but I am using a Python 3.12.3 kernel.

- Create a .env file with the environment variables. 


# Task 1A : Text extraction from a PDF
We must retrieve the plain text from a PDF file and identify the questions in a questionnaire. 
For our final product this might not even be useful because a lot of the documents will simply be scanned. 

[Task 1: PDF Extraction](./task1_PDFextraction.ipynb)

That being said, if one had to be chosen, We would go with PyMuPDF for its undeniable superiority. It is quicker and the text extracted is more accurate.  



# Task 1B: PDF to image + OCR on the image.

[Task 1B: PDF to Image](./task1b_PDFtoImage.ipynb)

The best way to turn a PDF into an image is not as easily found. The quickest way for it is the library pdfium2, however, PyMuPDF is an option worth considering because it can easily denoise the image and make it black and white. These two extra features and the fact that it is the best text retriever make it the best option. 

[Task 1C: OCR on the Image](./task1c_OCRonImage.ipynb)


After carefully reviewing, all the open source models can simply be forgotten. Those who perform well like EasyOCR can get very slow running on a common laptop. Getting a working product would be very demanding. EasyOCR used with a good GPU would be good, but still it has trouble detecting double lined cells in tables. 

[Task 1C: Paid OCR Services](./task1c_PaidOCR.ipynb)

In terms of paid OCR, the best options I have found are Google Gemini 2.5 from Google Vertex AI and Meta Llama 4 Maverick. Maverick doesn't have coordinates but is the best at properly detecting layout. As a consequence, I used Google Cloud and its models for the working prototype I developed.

# Task 2: Embedding for a similarity search using AI

### Embedding: Stands for associating values to multidimensional vectors to perform searches on the text of said documents. It enhances the performance given the 3D proximity context which can easily match synonyms. 

[Task 2a: Denoising text with an LLM](./task2a_denoisingLLM.ipynb) 

I did this study in case I couldn't get a good OCR to work and extract the text properly but it won't be needed in the end. In addition, some LLMs seem to be incapable of sticking to a given format no matter the prompt. 

[Task 2b: Embedding](./task2b_Embedding.ipynb)

The embeddings file is split in open source and pricing models. The first half is for open source and the second is for pricing. They still could go through some testing but I have found that most of the open source models are just datasets. Instead of actually embedding, they just assign a value they have to each word from the training set and stick with it no matter the context. We can see this is the case when getting errors like no value for key "word", it can't embed your token because it is not there. 

On the other hand though, pricing models do perform the embedding processes better. Mixed with the embeddings are benchmarks for Named Entity Recognition, which turns out not to be useful for what we are producing. 

# Task 3: Get the questionnaire from Milo's API

[Task 3 : API retrieval](./task3:getQuestionnaireMilo.ipynb)

A simple browse through the titles is enough to get what we are looking for. Treating the JSON format and performing an embedding would be great to find potential matches.  

## Conclusion: 

I have created a model that can find the answers with a given questionnaire. I needed 3 different API calls. The process can still be optimized and the search algorithm still needs to be written. However, the results are still very promising. We have seen that the optimal solutions are the ones offered by the main tech corporations for their price and capabilities. In addition, in the case of my prototype, the entire process can be done using Google Cloud's API, which could simplify the process due to the centralized cloud-based approach. The downside of this is the weaker data protection that these companies have been at times known for. With enough growth, running these models on our own hardware would be the right choice given the confidentiality needed for medical data. Here is what I am referring to: [Major Data Leaks](https://tech.co/news/data-breaches-updated-list). Among the different cloud providers needed to develop this product, I have made the following [Price Comparison](./Price_Comparison/OCR_price_comparison.ods). It can be opened in the LibreOffice suite. It shows that for similar results, Google Cloud and AWS provide the best services to perform OCR and call LLM/Embedding models.  

[Prototype 1](./Prototype1/Prototype1.ipynb)

Developing a working product to solve a real world problem was a great learning experience. I can't wait to use this newly acquired knowledge to solve other issues using the latest AI developments.

# References & Sources

[1] Deepseek Team, Deepseek: Quick doubts, definitions, and explanations. [Online]. Available: https://deepseek.com

[2] GeeksforGeeks Team, “Extract text from PDF file using Python,” GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/

[3] J. Singer-Vine, pdfplumber Documentation. [Online]. Available: https://github.com/jsvine/pdfplumber

[4] B. Rogojan, “How to automate PDF data extraction: 3 different methods to parse PDFs for analytics,” Seattle Data Guy. [Online]. Available: https://www.theseattledataguy.com/how-to-automate-pdf-data-extraction-3-different-methods-to-parse-pdfs-for-analytics/#page-content

[5] Python Software Foundation, “time — Time access and conversions,” Python Documentation. [Online]. Available: https://docs.python.org/3/library/time.html

[6] Google, Tesseract OCR. [Online]. Available: https://github.com/tesseract-ocr/tesseract

[7] S. Hoffstaetter, pytesseract GitHub Repository. [Online]. Available: https://github.com/madmaze/pytesseract

[8] J. Jerphanion, pdf2image GitHub Repository. [Online]. Available: https://github.com/Belval/pdf2image

[9] S. Dufour, PyPDF2 GitHub Repository. [Online]. Available: https://github.com/sdpython/PyPDF2

[10] PyMuPDF Team, PyMuPDF GitHub Repository. [Online]. Available: https://github.com/pymupdf/PyMuPDF

[11] J. J. Vens, pdfminer.six GitHub Repository. [Online]. Available: https://github.com/pdfminer/pdfminer.six

[12] E. Berger, Scalene GitHub Repository. [Online]. Available: https://github.com/plasma-umass/scalene?tab=readme-ov-file

[13] E. Berger, “Scalene: A high-performance, high-precision CPU, GPU, and memory profiler for Python,” YouTube. [Online]. Available: https://www.youtube.com/watch?v=5iEf-_7mM1k

[14] Z. Shen, R. Zhang, M. Dell, B. C. G. Lee, J. Carlson, and W. Li, “LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis,” arXiv preprint arXiv:2103.15348, 2021. [Online]. Available: https://github.com/Layout-Parser/layout-parser/tree/main

[15] LayoutParser Team, “EfficientDet Model for PubLayNet,” Hugging Face. [Online]. Available: https://huggingface.co/layoutparser/efficientdet/tree/main/PubLayNet/tf_efficientdet_d1

[16] PyPDFium2 Team, PyPDFium2 GitHub Repository. [Online]. Available: https://github.com/pypdfium2-team

[17] E. McConville, Wand GitHub Repository. [Online]. Available: https://github.com/emcconville/wand?tab=readme-ov-file

[18] Wand Documentation, “Install Wand on Debian,” Wand. [Online]. Available: https://docs.wand-py.org/en/latest/guide/install.html#install-wand-debian

[19] J. Alankrita, pdftotext GitHub Repository. [Online]. Available: https://github.com/jalan/pdftotext

[20] JaidedAI Team, EasyOCR GitHub Repository. [Online]. Available: https://github.com/JaidedAI/EasyOCR

[21] PaddlePaddle Team, PaddleOCR GitHub Repository. [Online]. Available: https://github.com/PaddlePaddle/PaddleOCR

[22] UKPLab Team, SentenceTransformers GitHub Repository. [Online]. Available: https://github.com/UKPLab/sentence-transformers

[23] TensorFlow Team, “Semantic Similarity with TensorFlow Hub Universal Encoder,” GitHub Repository. [Online]. Available: https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder.ipynb

[24] Explosion Team, spaCy GitHub Repository. [Online]. Available: https://github.com/explosion/spaCy

[25] Hugging Face, Qwen2.5-VL-72B-Instruct Model Card. [Online]. Available: https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct?inference_provider=nebius

[26] Meta, Llama-3.2-11B-Vision-Instruct Model Card. [Online]. Available: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct?inference_provider=hf-inference

[27] Groq, Llama-4-Scout-17B-16E-Instruct Model Documentation. [Online]. Available: https://console.groq.com/docs/model/llama-4-scout-17b-16e-instruct

[28] Groq, Llama-4-Maverick-17B-128E-Instruct Model Documentation. [Online]. Available: https://console.groq.com/docs/model/llama-4-maverick-17b-128e-instruct

