# DocuMind AI — RAG Document Assistant

A beginner-friendly document assistant built with **Python, Streamlit, Groq API, and local TF-IDF retrieval**. The interface uses a blue-and-white colour theme.

Users can upload documents, then ask the assistant to answer questions, create short/long questions, generate MCQs, summarize content, explain material in easy wording, or draft an assignment based on retrieved document sections.

## Features

- Upload PDF, DOCX, PPTX, XLSX, XLS, JPG/JPEG, PNG, and TXT files.
- Extract text from PDFs, Word documents, spreadsheets, and images using OCR.
- Split document text into smaller sections.
- Retrieve relevant sections with local TF-IDF search.
- Generate answers and study material using Groq.
- Download the generated response as a TXT file.
- Blue-and-white Streamlit interface.

## Important notes

- This starter version supports modern Word files (`.docx`). Convert old `.doc` files to `.docx` before uploading.
- Scanned PDFs may contain images rather than selectable text. This version extracts regular PDF text; OCR for image files is included. OCR for scanned PDFs can be added in a later version.
- The retrieval step uses TF-IDF keyword matching, not neural embeddings. This keeps the first version easier to install and run. It is a basic retrieval-augmented generation (RAG) workflow, and semantic embeddings can be added later.
- The app uses Groq's hosted model API. API availability, model IDs, and free-tier limits can change. Check the [Groq models page](https://console.groq.com/docs/models) and [rate limits](https://console.groq.com/docs/rate-limits).
- Generated answers can contain mistakes. Verify them against your source material before submitting assignments.
- Uploaded documents are processed in the running app; do not upload sensitive or confidential files to a public app unless you understand the platform's privacy and data handling.

## Python version

Use **Python 3.12** for the most straightforward deployment. Streamlit Community Cloud currently defaults to Python 3.12; use a released version that is supported by Community Cloud and compatible with all dependencies. Python 3.14 may not be compatible with every package in this beginner project.

## 1. Project files

```text
rag-document-assistant/
├── app.py
├── requirements.txt
├── README.md
├── packages.txt
└── .gitignore
```

`packages.txt` installs the Tesseract OCR system package on Streamlit Community Cloud.

## 2. Create a Groq API key

1. Open https://console.groq.com/keys
2. Sign in or create a Groq account.
3. Create an API key and copy it somewhere safe.
4. Never paste your real API key into `app.py` or commit it to GitHub.

The app's default model is `openai/gpt-oss-120b`. If it is unavailable to your account, choose another model that is active in your Groq console, or change the model selection in the app.

## 3. Run locally (optional)

1. Install Python 3.12.
2. Open a terminal in this project folder.
3. Create a virtual environment:

   Windows:
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

4. Install Python packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Install Tesseract OCR on your computer if you want to read JPG/PNG images. The Python package alone does not install the Tesseract program. On Windows, install Tesseract and add its installation folder to PATH.

6. Create a folder named `.streamlit`, then create `.streamlit/secrets.toml` inside it with this content:
   ```toml
   GROQ_API_KEY = "paste_your_real_key_here"
   ```

7. Start the app:
   ```powershell
   streamlit run app.py
   ```

8. Streamlit will show a local address, usually `http://localhost:8501`.

Do not upload `.streamlit/secrets.toml` to GitHub.

## 4. Upload the project to GitHub (browser method)

1. Open https://github.com and sign in.
2. Click **+** in the top-right corner, then **New repository**.
3. Name it `rag-document-assistant`.
4. Choose **Public** or **Private**. A public repository makes the source code visible to everyone.
5. Click **Create repository**.
6. Open the new repository and choose **Add file → Upload files**.
7. Upload `app.py`, `requirements.txt`, `README.md`, `packages.txt`, and `.gitignore`.
8. Click **Commit changes**.

Make sure your real Groq API key is not among the uploaded files.

## 5. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io.
2. Sign in using GitHub.
3. Click **Create app**.
4. Select your repository: `rag-document-assistant`.
5. Select branch: `main`.
6. Set the main file path to `app.py`.
7. Open **Advanced settings**.
8. Select **Python 3.12**.
9. In the **Secrets** box, enter:
   ```toml
   GROQ_API_KEY = "paste_your_real_key_here"
   ```
10. Click **Save**, then **Deploy**.
11. Wait for installation and startup. Streamlit will give you a public app URL.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

Official secrets guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

## 6. How to use the app

1. Open your deployed app.
2. Upload one or more supported files.
3. Click **Read and process files**.
4. Choose a task, such as **Make short questions**.
5. Select the number of questions and whether answers should be included.
6. Click **Generate result**.
7. Read the response and download it as a TXT file.

## Troubleshooting

### `GROQ_API_KEY` not found
Check that the key is set in Streamlit Community Cloud → app settings → Secrets. Locally, check `.streamlit/secrets.toml`.

### Model not found or unavailable
Open the Groq model list and choose a currently available model ID. Update the model options in `app.py` if needed.

### Rate limit reached
Wait for the limit to reset, reduce the number of requests, or check your Groq account's current rate limits.

### JPG/PNG OCR fails
Install the Tesseract OCR application itself, not only the Python `pytesseract` package. On Community Cloud, `packages.txt` installs `tesseract-ocr`.

### Spreadsheet error
Check that the file is a valid `.xls` or `.xlsx` spreadsheet and is not password-protected.

### No text extracted from PDF
The PDF may be scanned. This starter version does not yet OCR scanned PDF pages. Add scanned-PDF OCR if you need that capability.

## Next improvements

- Add OCR for scanned PDFs.
- Export answers and assignments to DOCX/PDF.
- Add source citations and page numbers.
- Add chat history and multiple document collections.
- Replace TF-IDF retrieval with semantic embeddings and a vector database such as ChromaDB.
- Add safeguards for very large files and token limits.
