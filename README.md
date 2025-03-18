# Hackathon Assignment

## Overview
This project contains a Python script (`pdfTableExtractor.py`) designed to extract tables from PDF files. The repository includes sample PDF files for testing purposes.

## Files in This Repository
- **pdfTableExtractor.py** - Python script for extracting tables from PDF files.
- **test3.pdf, test6.pdf** - Sample PDF files for testing table extraction.


## Prerequisites
Before running the script, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install pdfplumber pandas
  ```

## How to Use
1. Place the PDFs containing tables in the same directory as `pdfTableExtractor.py`.
2. Run the script using:
   ```bash
   python pdfTableExtractor.py
   ```
3. The extracted tables will be displayed or saved as output files.

## Expected Output
The script will process the PDF files and extract tables into structured data (e.g., CSV or DataFrames).

## License
This project is open-source and available under the MIT License.
