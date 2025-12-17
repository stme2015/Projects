# GLR Pipeline - Insurance Document Automation

## Overview
A professional Streamlit web application that automates the processing of insurance General Loss Reports (GLR) by extracting data from PDF inspection reports and populating DOCX templates using AI-powered analysis.

## Architecture
- **Frontend**: Streamlit web interface with multi-step workflow
- **AI Processing**: Groq LLM for intelligent data extraction from unstructured PDF content
- **Document Processing**: PDF text extraction with OCR fallback, DOCX template manipulation
- **Workflow**: 5-step pipeline (Upload → Extract → Process → Generate → Complete)

## Tech Stack
- **Python** - Core language
- **Streamlit** - Web interface and session management
- **Groq API** - AI-powered data extraction (Llama 3.3 70B model)
- **pdfplumber** - Primary PDF text extraction
- **Tesseract OCR** - Fallback text extraction from images
- **python-docx** - Template manipulation and population
- **PIL** - Image preprocessing for OCR

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR**:
   - Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
   - Update the path in `document_processing.py` if needed

3. **Set up Groq API key**:
   - Get API key from https://console.groq.com/
   - Replace the hardcoded key in the script or set as environment variable

4. **Run the application**:
   ```bash
   streamlit run document_processing.py
   ```

## Key Decisions
- **AI-First Approach**: Uses LLM for intelligent field mapping rather than rule-based extraction
- **OCR Fallback**: Ensures processing works even with image-based PDFs
- **Template-Driven**: Flexible system that adapts to any DOCX template structure
- **Multi-Step UI**: Clear user experience with progress tracking and validation
- **Session State Management**: Maintains workflow state across page interactions

## Usage
1. Upload a DOCX template containing placeholder fields (e.g., `[CLAIM_NUMBER]`, `{DATE_LOSS}`)
2. Upload one or more PDF inspection reports
3. The AI analyzes the PDFs to extract relevant data matching template fields
4. Review extracted data and generate the completed document
5. Download the populated template and processing log
