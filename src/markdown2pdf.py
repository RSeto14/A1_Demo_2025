import markdown
import pdfkit
from weasyprint import HTML

def markdown_to_html(markdown_text):
    """Convert Markdown text to HTML."""
    html = markdown.markdown(markdown_text)
    return html

def html_to_pdf(html, output_path):
    """Convert HTML to PDF using pdfkit."""
    pdfkit.from_string(html, output_path)

def html_to_pdf_weasyprint(html, output_path):
    """Convert HTML to PDF using WeasyPrint."""
    HTML(string=html).write_pdf(output_path)

def convert_markdown_file(input_path, output_path, output_format='pdf'):
    """Convert a Markdown file to PDF or HTML."""
    with open(input_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    html = markdown_to_html(markdown_text)

    if output_format == 'pdf':
        # Choose one of the methods to convert HTML to PDF
        html_to_pdf(html, output_path)
        # or
        # html_to_pdf_weasyprint(html, output_path)
    elif output_format == 'html':
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(html)
    else:
        raise ValueError("Unsupported output format. Use 'pdf' or 'html'.")

# Example usage
convert_markdown_file('example.md', 'output.pdf', 'pdf')
convert_markdown_file('example.md', 'output.html', 'html')
