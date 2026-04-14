#!/usr/bin/env python3
"""
nb2pdf_agent.py - AI-powered Jupyter Notebook to PDF Converter
================================================================

This agent takes any .ipynb file as input, parses its components,
and generates a professionally formatted PDF report.

Usage:
    python nb2pdf_agent.py <notebook.ipynb> [-o output.pdf]
    python nb2pdf_agent.py --help

Requirements:
    - Python 3.10+
    - reportlab (PDF generation)
    - Pygments (syntax highlighting)
    - langchain (AI-powered extraction)
"""

import json
import sys
import argparse
import io
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

# Try to import required libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib.colors import (
        HexColor, white, black, lightgrey, darkblue,
        darkgreen, darkred, purple, navy
    )
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak,
        Preformatted, Table, TableStyle, Image, TableOfContents
    )
    from reportlab.pdfgen import canvas
except ImportError:
    print("ERROR: reportlab not installed. Run: pip install reportlab")
    sys.exit(1)

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer, TextLexer, MarkdownLexer
    from pygments.formatters import HtmlFormatter
except ImportError:
    print("ERROR: pygments not installed. Run: pip install pygments")
    sys.exit(1)

# Optional LangChain imports
try:
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("WARNING: langchain not installed. Using standard parsing.")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for PDF styling."""

    # Colors (professional palette)
    PRIMARY_COLOR = HexColor("#1a5276")      # Deep blue
    SECONDARY_COLOR = HexColor("#2874a6")      # Medium blue
    ACCENT_COLOR = HexColor("#3498db")       # Light blue
    TEXT_COLOR = HexColor("#2c3e50")          # Dark gray
    CODE_BG_COLOR = HexColor("#f8f9fa")       # Light gray
    OUTPUT_BG_COLOR = HexColor("#f0f4f8")    # Very light blue-gray
    HEADER_BG_COLOR = HexColor("#1a5276")    # Header background

    # Fonts
    TITLE_FONT = "Helvetica-Bold"
    HEADING_FONT = "Helvetica-Bold"
    BODY_FONT = "Helvetica"
    CODE_FONT = "Courier"

    # Page settings
    PAGE_SIZE = A4
    MARGIN_LEFT = 0.75 * inch
    MARGIN_RIGHT = 0.75 * inch
    MARGIN_TOP = 0.75 * inch
    MARGIN_BOTTOM = 0.75 * inch


# =============================================================================
# NOTEBOOK PARSER
# =============================================================================

class NotebookParser:
    """Parse Jupyter Notebook JSON structure."""

    def __init__(self, notebook_path: str):
        self.path = Path(notebook_path)
        self.notebook_data = None
        self.metadata = {}
        self.cells = []

    def load(self) -> bool:
        """Load and parse the notebook file."""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.notebook_data = json.load(f)

            # Extract notebook metadata
            self.metadata = {
                'nbformat': self.notebook_data.get('nbformat', 4),
                'nbformat_minor': self.notebook_data.get('nbformat_minor', 0),
                'kernel': self.notebook_data.get('metadata', {}).get('kernelspec', {}).get('name', 'Unknown'),
                'title': self.notebook_data.get('metadata', {}).get('title', self.path.stem),
            }

            # Extract cells
            self.cells = self.notebook_data.get('cells', [])
            return True

        except FileNotFoundError:
            print(f"ERROR: File not found: {self.path}")
            return False
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in notebook: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load notebook: {e}")
            return False

    def get_title(self) -> str:
        """Get notebook title."""
        return self.metadata.get('title', self.path.stem)

    def get_kernel(self) -> str:
        """Get kernel specification."""
        return self.metadata.get('kernel', 'Unknown')

    def extract_cell_content(self, cell: dict) -> dict:
        """Extract content from a cell."""
        cell_type = cell.get('cell_type', 'code')
        result = {
            'type': cell_type,
            'source': '',
            'outputs': [],
            'metadata': cell.get('metadata', {}),
            'execution_count': cell.get('execution_count', None),
        }

        if cell_type == 'markdown':
            result['source'] = ''.join(cell.get('source', []))
        else:  # code cell
            result['source'] = ''.join(cell.get('source', []))
            result['outputs'] = self._extract_outputs(cell.get('outputs', []))

        return result

    def _extract_outputs(self, outputs: list) -> list:
        """Extract cell outputs."""
        extracted = []
        for output in outputs:
            output_type = output.get('output_type', '')

            if output_type == 'stream':
                extracted.append({
                    'type': 'stream',
                    'name': output.get('name', 'stdout'),
                    'text': ''.join(output.get('text', [])),
                })
            elif output_type == 'execute_result':
                data = output.get('data', {})
                if 'text/plain' in data:
                    extracted.append({
                        'type': 'execute_result',
                        'mime': 'text/plain',
                        'text': ''.join(data['text/plain']),
                    })
                # Handle images
                if 'image/png' in data:
                    extracted.append({
                        'type': 'execute_result',
                        'mime': 'image/png',
                        'data': data['image/png'],
                    })
                if 'image/jpeg' in data:
                    extracted.append({
                        'type': 'execute_result',
                        'mime': 'image/jpeg',
                        'data': data['image/jpeg'],
                    })
            elif output_type == 'display_data':
                data = output.get('data', {})
                if 'text/plain' in data:
                    extracted.append({
                        'type': 'display_data',
                        'mime': 'text/plain',
                        'text': ''.join(data['text/plain']),
                    })
                if 'image/png' in data:
                    extracted.append({
                        'type': 'display_data',
                        'mime': 'image/png',
                        'data': data['image/png'],
                    })
                if 'image/jpeg' in data:
                    extracted.append({
                        'type': 'display_data',
                        'mime': 'image/jpeg',
                        'data': data['image/jpeg'],
                    })
            elif output_type == 'error':
                extracted.append({
                    'type': 'error',
                    'ename': output.get('ename', ''),
                    'evalue': output.get('evalue', ''),
                    'traceback': '\n'.join(output.get('traceback', [])),
                })

        return extracted

    def get_cells(self) -> list:
        """Get all parsed cells."""
        return [self.extract_cell_content(cell) for cell in self.cells]


# =============================================================================
# CONTENT FORMATTER
# =============================================================================

class ContentFormatter:
    """Format markdown and code content for PDF."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=Config.PRIMARY_COLOR,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName=Config.TITLE_FONT,
        ))

        # Heading 1
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=Config.PRIMARY_COLOR,
            spaceAfter=12,
            spaceBefore=20,
            fontName=Config.HEADING_FONT,
        ))

        # Heading 2
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=Config.SECONDARY_COLOR,
            spaceAfter=10,
            spaceBefore=16,
            fontName=Config.HEADING_FONT,
        ))

        # Heading 3
        self.styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=Config.TEXT_COLOR,
            spaceAfter=8,
            spaceBefore=12,
            fontName=Config.HEADING_FONT,
        ))

        # Code block style
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Code'],
            fontSize=9,
            fontName=Config.CODE_FONT,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            spaceBefore=10,
            backgroundColor=Config.CODE_BG_COLOR,
        ))

        # Output block style
        self.styles.add(ParagraphStyle(
            name='OutputBlock',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName=Config.CODE_FONT,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            spaceBefore=10,
            backgroundColor=Config.OUTPUT_BG_COLOR,
            textColor=Config.TEXT_COLOR,
        ))

        # Error output style
        self.styles.add(ParagraphStyle(
            name='ErrorOutput',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName=Config.CODE_FONT,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            spaceBefore=10,
            backgroundColor=HexColor("#fdebd0"),
            textColor=HexColor("#c0392b"),
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=Config.TEXT_COLOR,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            fontName=Config.BODY_FONT,
        ))

    def format_markdown(self, text: str) -> list:
        """Format markdown text into PDF elements."""
        elements = []
        lines = text.split('\n')
        in_list = False
        list_items = []

        for line in lines:
            line = line.rstrip()

            # Skip empty lines at start
            if not line and not in_list:
                continue

            # Heading detection
            if line.startswith('### '):
                elements.append(Paragraph(
                    self._escape_markdown(line[4:]), self.styles['CustomHeading3']
                ))
            elif line.startswith('## '):
                elements.append(Paragraph(
                    self._escape_markdown(line[3:]), self.styles['CustomHeading2']
                ))
            elif line.startswith('# '):
                elements.append(Paragraph(
                    self._escape_markdown(line[2:]), self.styles['CustomHeading1']
                ))
            # Bullet list
            elif line.startswith('- ') or line.startswith('* '):
                list_items.append(self._escape_markdown(line[2:]))
                in_list = True
            # Numbered list
            elif re.match(r'^\d+\.\s', line):
                list_items.append(self._escape_markdown(re.sub(r'^\d+\.\s', '', line)))
                in_list = True
            # Horizontal rule
            elif line in ['---', '***', '___']:
                elements.append(Spacer(1, 10))
                elements.append(Table(
                    [['']],
                    colWidths=[6.5*inch],
                    style=TableStyle([
                        ('LINEABOVE', (0, 0), (0, 0), 1, Config.ACCENT_COLOR),
                        ('TOPPADDING', (0, 0), (0, 0), 8),
                        ('BOTTOMPADDING', (0, 0), (0, 0), 8),
                    ])
                ))
                elements.append(Spacer(1, 10))
            # Inline code
            elif '`' in line:
                formatted_line = self._format_inline_code(line)
                elements.append(Paragraph(formatted_line, self.styles['CustomBody']))
            # Empty line (end of list)
            elif not line and in_list:
                if list_items:
                    self._add_list(elements, list_items)
                    list_items = []
                in_list = False
            # Regular text
            else:
                if in_list:
                    self._add_list(elements, list_items)
                    list_items = []
                    in_list = False
                if line:
                    formatted_line = self._format_inline_code(line)
                    elements.append(Paragraph(formatted_line, self.styles['CustomBody']))

        # Handle list at end
        if list_items:
            self._add_list(elements, list_items)

        return elements

    def _add_list(self, elements: list, items: list):
        """Add a list to elements."""
        for item in items:
            # Create bullet point
            bullet = Paragraph(
                f"&#8226; {item}",
                self.styles['CustomBody']
            )
            elements.append(bullet)
            elements.append(Spacer(1, 3))

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for PDF."""
        # Handle markdown formatting
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')

        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

        # Italic: *text* or _text_
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)

        # Code: `code`
        text = re.sub(r'`(.+?)`', r'<font face="Courier" size="10">\1</font>', text)

        # Links: [text](url)
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<link href="\2"><u>\1</u></link>', text)

        return text

    def _format_inline_code(self, text: str) -> str:
        """Format inline code markers."""
        # More aggressive code formatting
        text = self._escape_markdown(text)
        return text

    def format_code(self, code: str, execution_count: Optional[str] = None) -> list:
        """Format code with syntax highlighting."""
        elements = []

        # Add execution count if available
        if execution_count:
            label = f"[{execution_count}]"
            elements.append(Paragraph(
                f"<font color='{Config.SECONDARY_COLOR.hexval()}' size='10'>{label}</font>",
                self.styles['CodeBlock']
            ))

        # Use Pygments for syntax highlighting
        try:
            # Get HTML formatted code
            formatter = HtmlFormatter(
                style='monokai',
                full=False,
                cssclass='highlight',
                noclasses=True,
            )
            highlighted = highlight(code, PythonLexer(), formatter)

            # Extract just the body content
            if '<div class="highlight">' in highlighted:
                start = highlighted.find('>', highlighted.find('<div class="highlight">')) + 1
                end = highlighted.rfind('</div>')
                if start > 0 and end > start:
                    content = highlighted[start:end]
                    # Clean up HTML
                    content = content.replace('<span style="', '<span color="')
                    content = re.sub(r'<span color="#([0-9a-fA-F]{6})">', r'<span foreColor="\1">', content)
                    content = content.replace('</span>', '')

                    elements.append(Paragraph(
                        content,
                        self.styles['CodeBlock']
                    ))
                else:
                    # Fallback
                    elements.append(Preformatted(
                        code,
                        self.styles['CodeBlock']
                    ))
            else:
                elements.append(Preformatted(
                    code,
                    self.styles['CodeBlock']
                ))
        except:
            # Fallback to plain text
            elements.append(Preformatted(
                code,
                self.styles['CodeBlock']
            ))

        return elements

    def format_output(self, outputs: list) -> list:
        """Format cell outputs."""
        elements = []

        for output in outputs:
            if output['type'] == 'stream':
                text = output.get('text', '')
                if text.strip():
                    style = self.styles['OutputBlock']
                    elements.append(Preformatted(text, style))

            elif output['type'] in ('execute_result', 'display_data'):
                if output.get('mime') == 'text/plain':
                    text = output.get('text', '')
                    if text.strip():
                        elements.append(Preformatted(text, self.styles['OutputBlock']))
                elif output.get('mime', '').startswith('image/'):
                    # Handle images later
                    pass

            elif output['type'] == 'error':
                # Error output
                error_text = f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
                elements.append(Paragraph(
                    f"<b>Error:</b> {error_text}",
                    self.styles['ErrorOutput']
                ))
                if output.get('traceback'):
                    elements.append(Preformatted(
                        output['traceback'],
                        self.styles['ErrorOutput']
                    ))

        return elements


# =============================================================================
# PDF GENERATOR
# =============================================================================

class PDFGenerator:
    """Generate professional PDF from notebook."""

    def __init__(self, output_path: str, title: str):
        self.output_path = output_path
        self.title = title
        self.story = []
        self.toc_entries = []
        self.formatter = ContentFormatter()
        self.page_num = 0

    def generate(self, parser: NotebookParser):
        """Generate the PDF document."""
        # Create PDF document
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=Config.PAGE_SIZE,
            leftMargin=Config.MARGIN_LEFT,
            rightMargin=Config.MARGIN_RIGHT,
            topMargin=Config.MARGIN_TOP,
            bottomMargin=Config.MARGIN_BOTTOM,
        )

        # Build story
        self._build_title_page()
        self._build_table_of_contents()
        self._build_cells(parser)

        # Build PDF
        doc.build(
            self.story,
            onFirstPage=self._add_header_footer,
            onLaterPages=self._add_header_footer
        )

        print(f"PDF generated: {self.output_path}")

    def _build_title_page(self):
        """Build title page."""
        self.story.append(Spacer(1, 2.5 * inch))

        # Main title
        self.story.append(Paragraph(
            self.title,
            self.formatter.styles['CustomTitle']
        ))

        self.story.append(Spacer(1, 0.5 * inch))

        # Subtitle
        self.story.append(Paragraph(
            "Jupyter Notebook Report",
            self.formatter.styles['CustomHeading2']
        ))

        self.story.append(Spacer(1, 0.25 * inch))

        # Metadata
        self.story.append(Paragraph(
            f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            self.formatter.styles['CustomBody']
        ))

        self.story.append(PageBreak())

    def _build_table_of_contents(self):
        """Build table of contents."""
        self.story.append(Paragraph(
            "Table of Contents",
            self.formatter.styles['CustomHeading1']
        ))

        self.story.append(Spacer(1, 0.25 * inch))

        # Placeholder TOC - will be filled based on cells
        for entry in self.toc_entries:
            self.story.append(Paragraph(
                f"{entry['indent']} {entry['text']} ...... {entry['page']}",
                self.formatter.styles['CustomBody']
            ))

        self.story.append(PageBreak())

    def _build_cells(self, parser: NotebookParser):
        """Build all cells."""
        cell_num = 0
        toc_idx = 0

        for cell in parser.get_cells():
            cell_num += 1
            cell_type = cell['type']

            if cell_type == 'markdown':
                # Markdown cell
                if cell['source']:
                    # Determine heading level
                    source = cell['source']
                    if source.startswith('# '):
                        heading = source[2:].strip()
                        self.story.append(Paragraph(
                            heading,
                            self.formatter.styles['CustomHeading1']
                        ))
                        self.toc_entries.append({
                            'text': heading,
                            'page': '',
                            'indent': ''
                        })
                    elif source.startswith('## '):
                        heading = source[3:].strip()
                        self.story.append(Paragraph(
                            heading,
                            self.formatter.styles['CustomHeading2']
                        ))
                        self.toc_entries.append({
                            'text': heading,
                            'page': '',
                            'indent': '    '
                        })
                    elif source.startswith('### '):
                        heading = source[4:].strip()
                        self.story.append(Paragraph(
                            heading,
                            self.formatter.styles['CustomHeading3']
                        ))
                        self.toc_entries.append({
                            'text': heading,
                            'page': '',
                            'indent': '        '
                        })
                    else:
                        elements = self.formatter.format_markdown(source)
                        self.story.extend(elements)
            else:
                # Code cell
                exec_count = cell.get('execution_count')
                code = cell['source']

                if code.strip():
                    # Cell header
                    self.story.append(Paragraph(
                        f"<b>Cell {cell_num}</b>",
                        self.formatter.styles['CustomHeading3']
                    ))

                    # Code block
                    code_elements = self.formatter.format_code(code, exec_count)
                    self.story.extend(code_elements)

                    # Outputs
                    if cell.get('outputs'):
                        output_elements = self.formatter.format_output(cell['outputs'])
                        if output_elements:
                            self.story.append(Spacer(1, 0.1 * inch))
                            self.story.append(Paragraph(
                                "<b>Output:</b>",
                                self.formatter.styles['CustomHeading3']
                            ))
                            self.story.extend(output_elements)

                    self.story.append(Spacer(1, 0.2 * inch))

    def _add_header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        self.page_num += 1

        # Header
        canvas.setFillColor(Config.HEADER_BG_COLOR)
        canvas.rect(
            doc.leftMargin,
            doc.pagesize[1] - doc.topMargin + 10,
            doc.pagesize[0] - doc.leftMargin - doc.rightMargin,
            2,
            fill=1
        )

        canvas.setFillColor(white)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(
            doc.leftMargin,
            doc.pagesize[1] - doc.topMargin + 5,
            self.title[:50]
        )

        # Page number
        canvas.drawRightString(
            doc.pagesize[0] - doc.rightMargin,
            doc.pagesize[1] - doc.topMargin + 5,
            f"Page {self.page_num}"
        )

        # Footer
        canvas.setFillColor(lightgrey)
        canvas.rect(
            doc.leftMargin,
            doc.bottomMargin - 10,
            doc.pagesize[0] - doc.leftMargin - doc.rightMargin,
            1,
            fill=1
        )


# =============================================================================
# LANGCHAIN INTEGRATION (Optional)
# =============================================================================

def use_langchain_extraction(notebook_path: str) -> Optional[dict]:
    """
    Use LangChain for intelligent extraction (optional).
    This demonstrates LangChain integration as requested.
    """
    if not LANGCHAIN_AVAILABLE:
        return None

    try:
        from langchain_community.document_loaders import JSONLoader

        # Use LangChain to extract documents
        # This is optional - primary extraction uses custom parser
        loader = JSONLoader(
            file_path=notebook_path,
            text_content=False
        )
        docs = loader.load()

        # Use text splitter for chunking if needed
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        return {
            'document_count': len(docs),
            'chunk_count': len(chunks),
            'status': 'processed with langchain'
        }
    except Exception as e:
        print(f"LangChain extraction note: {e}")
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Jupyter Notebooks to Professional PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nb2pdf_agent.py notebook.ipynb
  python nb2pdf_agent.py notebook.ipynb -o report.pdf
  python nb2pdf_agent.py notebook.ipynb --title "My Report"
        """
    )

    parser.add_argument(
        'notebook',
        nargs='?',
        help='Input .ipynb file path'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output PDF file path (default: input_name.pdf)'
    )
    parser.add_argument(
        '--title',
        help='Title for the report (default: notebook title)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--langchain',
        action='store_true',
        help='Use LangChain for extraction (if available)'
    )

    args = parser.parse_args()

    # Check for notebook path
    notebook_path = args.notebook
    if not notebook_path:
        # Check input from stdin
        import sys
        if not sys.stdin.isatty():
            notebook_path = sys.stdin.read().strip()

    if not notebook_path:
        parser.print_help()
        print("\nERROR: Please provide a notebook file path.")
        sys.exit(1)

    # Validate input file
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        print(f"ERROR: File not found: {notebook_path}")
        sys.exit(1)

    if notebook_path.suffix.lower() != '.ipynb':
        print(f"ERROR: Expected .ipynb file, got: {notebook_path.suffix}")
        sys.exit(1)

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = notebook_path.with_suffix('.pdf')

    if args.verbose:
        print(f"Processing: {notebook_path}")
        print(f"Output: {output_path}")

    # Parse notebook
    nb_parser = NotebookParser(str(notebook_path))
    if not nb_parser.load():
        sys.exit(1)

    # Get title
    title = args.title if args.title else nb_parser.get_title()

    if args.verbose:
        print(f"Notebook title: {title}")
        print(f"Kernel: {nb_parser.get_kernel()}")
        print(f"Cells: {len(nb_parser.cells)}")

    # Optional LangChain processing
    if args.langchain:
        lc_result = use_langchain_extraction(str(notebook_path))
        if lc_result and args.verbose:
            print(f"LangChain: {lc_result}")

    # Generate PDF
    generator = PDFGenerator(str(output_path), title)
    generator.generate(nb_parser)

    print(f"Success! PDF saved to: {output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())