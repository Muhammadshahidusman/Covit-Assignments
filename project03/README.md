# nb2pdf_agent - Jupyter Notebook to Professional PDF Converter

An AI-powered agent that converts Jupyter Notebooks (.ipynb) into professionally formatted PDF reports with syntax highlighting, styled markdown, and cell outputs.

## Features

- **Parses** .ipynb JSON structure (markdown cells, code cells, outputs)
- **Converts** markdown to formatted text (headings, bold, italic, lists, tables)
- **Syntax highlighting** for Python code using Pygments
- **Includes** cell outputs (text, tables, error tracebacks)
- **Professional PDF** with header, page numbers, and table of contents
- **Optional LangChain** integration for advanced extraction

## Installation

### Prerequisites

- Python 3.10 or higher

### Install Dependencies

```bash
pip install -e .
```

Or install directly:

```bash
pip install reportlab pygments langchain-core langchain-community
```

## Usage

### Basic Usage

```bash
# Convert a notebook to PDF
python nb2pdf_agent.py notebook.ipynb
```

### With Custom Output

```bash
# Specify output filename
python nb2pdf_agent.py notebook.ipynb -o my_report.pdf
```

### With Custom Title

```bash
# Set custom report title
python nb2pdf_agent.py notebook.ipynb --title "My Lab Report"
```

### Verbose Mode

```bash
# Show detailed processing information
python nb2pdf_agent.py notebook.ipynb -v
```

### Use LangChain

```bash
# Enable LangChain extraction (optional)
python nb2pdf_agent.py notebook.ipynb --langchain
```

## Examples

### Command Line

```bash
# Process sample notebook
python nb2pdf_agent.py sample.ipynb -o sample_output.pdf -v
```

### Python API

```python
from nb2pdf_agent import NotebookParser, PDFGenerator

# Parse notebook
parser = NotebookParser("notebook.ipynb")
parser.load()

# Generate PDF
generator = PDFGenerator("output.pdf", "My Report")
generator.generate(parser)
```

## Output

The generated PDF includes:

- **Title Page** - Report title and generation timestamp
- **Table of Contents** - Hyperlinked sections
- **Markdown Cells** - Formatted headings, lists, bold/italic text
- **Code Cells** - Syntax-highlighted Python code
- **Cell Outputs** - Text output, images, error messages
- **Headers/Footers** - Document title and page numbers

## Sample Output

See `sample_output.pdf` for a demo.

## Error Handling

The agent handles:

- Missing files
- Invalid JSON
- Large notebooks
- Multiple output types (text, images, errors)

## Requirements Met

| Requirement | Implementation |
|-------------|---------------|
| Accept .ipynb input | Parser extracts JSON structure |
| Parse markdown/code/outputs | Separate extraction for each type |
| Markdown formatting | Headings, bold, italic, lists |
| Syntax highlighting | Pygments with Python lexer |
| Include outputs | Text, images, errors |
| Professional PDF | reportlab with headers, page numbers, TOC |
| LangChain integration | Optional `--langchain` flag |

## License

MIT License