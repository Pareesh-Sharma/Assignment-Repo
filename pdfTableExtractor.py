import os
import re
import argparse
from collections import defaultdict

import PyPDF2
import pandas as pd


class PDFTableExtractor:
    """
    A class to extract tables from system-generated PDFs without using Tabula or Camelot,
    and without converting PDF pages to images. It handles LaTeX-style tables and also
    attempts to detect 'regular' tables based on delimiters/whitespace.
    """

    def __init__(self, verbose=False):
        self.tables = []
        self.verbose = verbose

    def extract_from_pdf(self, pdf_path):
        """
        Extracts tables from the given PDF file.

        :param pdf_path: path to the .pdf file
        :return: a list of dictionaries:
            [
              {
                'page': <page_number>,
                'table_num': <table_index_in_that_page>,
                'data': <pandas.DataFrame with the table contents>
              },
              ...
            ]
        """
        self.tables = []  # Reset for each PDF
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(reader.pages):
                # Extract text using PyPDF2
                text = page.extract_text() or ""
                if self.verbose:
                    print(f"[DEBUG] Page {page_num+1} text length: {len(text)}")

                # Decide whether it's LaTeX-style or "regular" table(s)
                if self._is_latex_table(text):
                    page_tables = self._extract_latex_tables(text)
                else:
                    page_tables = self._extract_regular_tables(text)

                for i, table_df in enumerate(page_tables):
                    self.tables.append({
                        'page': page_num + 1,
                        'table_num': i + 1,
                        'data': table_df
                    })

        return self.tables

    def save_to_excel(self, output_path):
        """
        Saves all extracted tables to an Excel file, each table on its own sheet.
        
        :param output_path: the .xlsx file path to write
        """
        if not self.tables:
            print("No tables to save.")
            return False

        with pd.ExcelWriter(output_path) as writer:
            for table_info in self.tables:
                sheet_name = f"Page{table_info['page']}_Table{table_info['table_num']}"
                # Ensure sheet name <= 31 chars (Excel constraint)
                sheet_name = sheet_name[:31]
                table_info['data'].to_excel(writer, sheet_name=sheet_name,
                                            index=False, header=False)

        print(f"Tables saved to: {output_path}")
        return True

    def _is_latex_table(self, text):
        """Check if the text likely contains LaTeX-style tables."""
        return "\\begin{tabular}" in text or "\\hline" in text

    def _extract_latex_tables(self, text):
        """
        Extract tables from LaTeX-style formatting.

        :param text: page text containing LaTeX table structures
        :return: list of DataFrames
        """
        tables = []
        # Regex to find \begin{tabular}... \end{tabular} blocks
        table_pattern = r"\\begin\{tabular\}(.*?)\\end\{tabular\}"
        table_matches = re.finditer(table_pattern, text, re.DOTALL)

        for match in table_matches:
            table_content = match.group(1)
            rows = []
            # Split rows by \hline or newline
            row_texts = re.split(r"\\hline|\n", table_content)

            for row_text in row_texts:
                row_text = row_text.strip()
                if not row_text:
                    continue

                # Split cells by '&' or '|'
                cells = re.split(r"&|\|", row_text)
                # Clean up cell text
                cells = [c.strip() for c in cells if c.strip()]

                if cells:
                    rows.append(cells)

            if rows:
                df = pd.DataFrame(rows)
                tables.append(df)

        return tables

    def _extract_regular_tables(self, text):
        """
        Extract tables from 'regular' formatting with possible delimiters,
        repeated dashes, or spacing.
        
        :param text: page text
        :return: list of DataFrames
        """
        tables = []
        lines = text.split('\n')

        table_lines = []
        in_table = False

        for line in lines:
            # A naive check: if line has multiple spaces, or '|' or repeated dash
            if re.search(r"\|\s+\||\s{3,}|\t{2,}|[-]{3,}", line):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            elif in_table and line.strip():
                # Potentially part of the table if not blank
                table_lines.append(line)
            else:
                # if we were in a table and hit a blank line or no match, 
                # consider that table ended
                if in_table and table_lines:
                    df = self._process_table_lines(table_lines)
                    if not df.empty:
                        tables.append(df)
                in_table = False
                table_lines = []

        # Handle a trailing table
        if in_table and table_lines:
            df = self._process_table_lines(table_lines)
            if not df.empty:
                tables.append(df)

        return tables

    def _process_table_lines(self, lines):
        """Turn a batch of lines likely forming a table into a DataFrame."""
        # 1) Detect column positions
        col_positions = self._detect_column_positions(lines)

        # 2) Extract each line's columns
        rows = []
        for line in lines:
            # skip lines that are only dashes or delimiters
            if not line.strip() or all(c in '-|+=' for c in line.strip()):
                continue
            row = []
            for (start, end) in col_positions:
                if start >= len(line):
                    row.append("")
                else:
                    segment = line[start:end] if end <= len(line) else line[start:]
                    row.append(segment.strip())
            if any(cell for cell in row):
                rows.append(row)

        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame()

    def _detect_column_positions(self, lines):
        """
        Identify column boundaries either by consistent vertical delimiters (|, +)
        or, if not found, fallback to whitespace detection.
        """
        delimiter_pattern = re.compile(r'[|+]')
        delimiter_positions = defaultdict(int)

        # Count how often each position has a delimiter
        for line in lines:
            for match in delimiter_pattern.finditer(line):
                delimiter_positions[match.start()] += 1

        # Positions that appear in many lines
        threshold = len(lines) * 0.5
        consistent_positions = sorted(
            pos for (pos, count) in delimiter_positions.items() if count >= threshold
        )

        if len(consistent_positions) > 1:
            # Create pairs (start, end) from consecutive positions
            return [
                (consistent_positions[i], consistent_positions[i+1])
                for i in range(len(consistent_positions) - 1)
            ]

        # Fallback: detect columns by whitespace
        return self._detect_columns_by_whitespace(lines)

    def _detect_columns_by_whitespace(self, lines):
        """
        If no consistent delimiter columns found, attempt to find columns
        by analyzing frequent 'blank space' gaps across lines.
        """
        whitespace_ranges_per_line = []
        for line in lines:
            # track consecutive whitespace segments
            in_space = False
            space_start = None
            line_ranges = []

            for i, ch in enumerate(line):
                if ch.isspace():
                    if not in_space:
                        in_space = True
                        space_start = i
                else:
                    if in_space:
                        # end the space segment
                        in_space = False
                        if i - space_start >= 3:  # minimum gap
                            line_ranges.append((space_start, i))
            # Check if ended with whitespace
            if in_space:
                end_i = len(line)
                if end_i - space_start >= 3:
                    line_ranges.append((space_start, end_i))

            whitespace_ranges_per_line.append(line_ranges)

        # Tally frequency of each range start/end
        freq_map = defaultdict(int)
        line_count = len(lines)
        for ranges in whitespace_ranges_per_line:
            for (start, end) in ranges:
                # We'll store the midpoint or do some approximation
                mid = (start + end) // 2
                freq_map[mid] += 1

        # Keep only positions that appear in at least 30% of lines
        cutoff = line_count * 0.3
        col_positions = sorted(pos for (pos, freq) in freq_map.items() if freq >= cutoff)

        # Build pairs
        if not col_positions:
            # If we cannot detect any meaningful columns, assume single column
            return [(0, 1000)]

        # Start from 0 to first col
        col_ranges = []
        prev_end = 0
        for pos in col_positions:
            if pos > prev_end:
                col_ranges.append((prev_end, pos))
                prev_end = pos
        # final range
        col_ranges.append((prev_end, 2000))

        return col_ranges


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract tables from PDF files without Tabula, Camelot, or image conversion.")
    parser.add_argument('--input', '-i', required=True, help='Path to a single PDF file or a directory containing PDFs')
    parser.add_argument('--output', '-o', required=True, help='Path to a single .xlsx file or a directory for saving multiple Excel files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debug output')
    return parser.parse_args()


def main():
    args = parse_arguments()
    extractor = PDFTableExtractor(verbose=args.verbose)

    input_path = args.input
    output_path = args.output

    # Check if input is a folder or single file
    if os.path.isdir(input_path):
        # We assume user wants to process multiple PDFs
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        for fname in os.listdir(input_path):
            if fname.lower().endswith(".pdf"):
                pdf_file_path = os.path.join(input_path, fname)
                excel_file_name = os.path.splitext(fname)[0] + ".xlsx"
                excel_out_path = os.path.join(output_path, excel_file_name)

                print(f"Processing: {pdf_file_path}")
                extractor.extract_from_pdf(pdf_file_path)
                extractor.save_to_excel(excel_out_path)
                extractor.tables = []  # Reset for next file

    else:
        # Single PDF input
        if os.path.isdir(output_path):
            # If output is a folder, build an Excel filename from the PDF base name
            pdf_base = os.path.basename(input_path)
            pdf_stem = os.path.splitext(pdf_base)[0]
            excel_file_name = pdf_stem + ".xlsx"
            excel_out_path = os.path.join(output_path, excel_file_name)
        else:
            # Output is presumably a single .xlsx file
            excel_out_path = output_path

        print(f"Processing: {input_path}")
        extractor.extract_from_pdf(input_path)
        extractor.save_to_excel(excel_out_path)

    print("Done!")


if __name__ == "__main__":
    main()
