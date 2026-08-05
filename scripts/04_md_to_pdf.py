#!/usr/bin/env python3.13
"""
Converte o whitepaper Black Hole (Markdown) para PDF profissional.
Usa ReportLab com fontes Noto Serif SC + FreeSerif.
Inclui capa, sumário, gráficos, e formatação de whitepaper técnico.
"""
import os
import re
import sys
import hashlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    KeepTogether, ListFlowable, ListItem, Flowable
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib.colors import HexColor

# === FONT REGISTRATION ===
FONT_DIR = '/usr/share/fonts'
pdfmetrics.registerFont(TTFont('NotoSerifSC', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf'))
pdfmetrics.registerFont(TTFont('NotoSerifSC-Bold', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Bold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif', f'{FONT_DIR}/truetype/freefont/FreeSerif.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Bold', f'{FONT_DIR}/truetype/freefont/FreeSerifBold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Italic', f'{FONT_DIR}/truetype/freefont/FreeSerifItalic.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-BoldItalic', f'{FONT_DIR}/truetype/freefont/FreeSerifBoldItalic.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', f'{FONT_DIR}/truetype/dejavu/DejaVuSansMono.ttf'))

registerFontFamily('FreeSerif', normal='FreeSerif', bold='FreeSerif-Bold',
                    italic='FreeSerif-Italic', boldItalic='FreeSerif-BoldItalic')
registerFontFamily('NotoSerifSC', normal='NotoSerifSC', bold='NotoSerifSC-Bold')

# === COLOR PALETTE (technical whitepaper, brutally honest) ===
PAGE_BG = HexColor('#FFFFFF')
TEXT_PRIMARY = HexColor('#1A1A1A')
TEXT_MUTED = HexColor('#666666')
ACCENT = HexColor('#C0392B')   # vermelho para destaques (warning, key finding)
ACCENT_2 = HexColor('#2E5C8A') # azul para headers
BORDER = HexColor('#CCCCCC')
CODE_BG = HexColor('#F5F5F5')
TABLE_HEADER = HexColor('#2E5C8A')
TABLE_ROW_ALT = HexColor('#F8F9FA')

# === STYLES ===
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    name='WhitepaperTitle',
    fontName='FreeSerif-Bold',
    fontSize=28,
    leading=34,
    alignment=TA_CENTER,
    textColor=TEXT_PRIMARY,
    spaceAfter=18,
)
style_subtitle = ParagraphStyle(
    name='WhitepaperSubtitle',
    fontName='FreeSerif-Italic',
    fontSize=14,
    leading=18,
    alignment=TA_CENTER,
    textColor=TEXT_MUTED,
    spaceAfter=24,
)
style_h1 = ParagraphStyle(
    name='H1',
    fontName='FreeSerif-Bold',
    fontSize=22,
    leading=28,
    alignment=TA_LEFT,
    textColor=TEXT_PRIMARY,
    spaceBefore=24,
    spaceAfter=12,
)
style_h2 = ParagraphStyle(
    name='H2',
    fontName='FreeSerif-Bold',
    fontSize=16,
    leading=22,
    alignment=TA_LEFT,
    textColor=ACCENT_2,
    spaceBefore=18,
    spaceAfter=8,
)
style_h3 = ParagraphStyle(
    name='H3',
    fontName='FreeSerif-Bold',
    fontSize=13,
    leading=18,
    alignment=TA_LEFT,
    textColor=TEXT_PRIMARY,
    spaceBefore=12,
    spaceAfter=6,
)
style_body = ParagraphStyle(
    name='Body',
    fontName='FreeSerif',
    fontSize=11,
    leading=16,
    alignment=TA_JUSTIFY,
    textColor=TEXT_PRIMARY,
    spaceAfter=8,
    firstLineIndent=0,
)
style_quote = ParagraphStyle(
    name='Quote',
    fontName='FreeSerif-Italic',
    fontSize=10.5,
    leading=15,
    alignment=TA_LEFT,
    textColor=TEXT_MUTED,
    leftIndent=24,
    rightIndent=24,
    spaceBefore=8,
    spaceAfter=8,
    borderColor=ACCENT,
    borderWidth=0,
)
style_code = ParagraphStyle(
    name='Code',
    fontName='DejaVuSans',
    fontSize=9,
    leading=12,
    alignment=TA_LEFT,
    textColor=TEXT_PRIMARY,
    backColor=CODE_BG,
    leftIndent=12,
    rightIndent=12,
    spaceBefore=4,
    spaceAfter=4,
    borderColor=BORDER,
    borderWidth=0.5,
    borderPadding=6,
)
style_caption = ParagraphStyle(
    name='Caption',
    fontName='FreeSerif-Italic',
    fontSize=9,
    leading=12,
    alignment=TA_CENTER,
    textColor=TEXT_MUTED,
    spaceBefore=4,
    spaceAfter=12,
)
style_toc1 = ParagraphStyle(
    name='TOC1',
    fontName='FreeSerif-Bold',
    fontSize=12,
    leading=16,
    leftIndent=0,
    spaceBefore=6,
    spaceAfter=2,
)
style_toc2 = ParagraphStyle(
    name='TOC2',
    fontName='FreeSerif',
    fontSize=11,
    leading=14,
    leftIndent=18,
    spaceBefore=2,
    spaceAfter=2,
)

# === HELPERS ===
def escape_md(text):
    """Escape characters that would be interpreted as XML in Paragraph."""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text

def md_inline(text):
    """Convert inline Markdown to ReportLab tags."""
    text = escape_md(text)
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<font name="DejaVuSans" size="9">\1</font>', text)
    return text

# === DOC TEMPLATE WITH TOC ===
class TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'bookmark_name'):
            level = getattr(flowable, 'bookmark_level', 0)
            text = getattr(flowable, 'bookmark_text', '')
            key = getattr(flowable, 'bookmark_key', '')
            self.notify('TOCEntry', (level, text, self.page, key))

def make_heading(text, style, level=0):
    """Create heading flowable with TOC bookmark."""
    key = f'h_{hashlib.md5(text.encode()).hexdigest()[:8]}'
    p = Paragraph(f'<a name="{key}"/>{md_inline(text)}', style)
    p.bookmark_name = key
    p.bookmark_level = level
    p.bookmark_text = text
    p.bookmark_key = key
    return p

# === PAGE NUMBER FOOTER ===
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('FreeSerif', 9)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawRightString(A4[0] - 20*mm, 12*mm, f'{doc.page}')
    canvas.drawString(20*mm, 12*mm, 'Black Hole Whitepaper v1.0')
    canvas.restoreState()

# === COVER PAGE ===
def build_cover():
    elements = []
    elements.append(Spacer(1, 80*mm))
    elements.append(Paragraph('Black Hole', ParagraphStyle(
        name='CoverTitle', fontName='FreeSerif-Bold', fontSize=56, leading=64,
        alignment=TA_CENTER, textColor=TEXT_PRIMARY, spaceAfter=18,
    )))
    elements.append(Paragraph('(BLKH)', ParagraphStyle(
        name='CoverTitle2', fontName='FreeSerif', fontSize=24, leading=30,
        alignment=TA_CENTER, textColor=TEXT_MUTED, spaceAfter=48,
    )))
    elements.append(Paragraph('Whitepaper Técnico v1.0', ParagraphStyle(
        name='CoverSub', fontName='FreeSerif-Italic', fontSize=18, leading=24,
        alignment=TA_CENTER, textColor=TEXT_PRIMARY, spaceAfter=12,
    )))
    elements.append(Paragraph('Compressão Neural Oportunista com<br/>Pré-cálculo em Ciclos Ociosos', ParagraphStyle(
        name='CoverSub2', fontName='FreeSerif', fontSize=14, leading=20,
        alignment=TA_CENTER, textColor=TEXT_MUTED, spaceAfter=48,
    )))
    elements.append(Spacer(1, 80*mm))
    # Bottom block
    elements.append(Paragraph('23 de Junho de 2026', ParagraphStyle(
        name='CoverDate', fontName='FreeSerif', fontSize=11,
        alignment=TA_CENTER, textColor=TEXT_MUTED, spaceAfter=4,
    )))
    elements.append(Paragraph('Research Artefact — Público', ParagraphStyle(
        name='CoverStatus', fontName='FreeSerif-Italic', fontSize=10,
        alignment=TA_CENTER, textColor=TEXT_MUTED,
    )))
    return elements

# === MAIN PARSER ===
def parse_markdown(md_text, image_dir):
    """Parse markdown and produce flowables list."""
    flowables = []
    lines = md_text.split('\n')
    i = 0
    in_code_block = False
    code_buffer = []
    code_lang = None

    # Track current section to know when to insert images
    current_section = ""

    while i < len(lines):
        line = lines[i]

        # Track section context
        if line.startswith('## '):
            current_section = line[3:].strip()

        # Insert images at relevant sections
        if line.startswith('## ') and 'Resultados por tamanho' in line:
            # Insert chart 1 after this header
            img_path = os.path.join(image_dir, 'chart_01_text_compression.png')
            if os.path.exists(img_path):
                flowables.append(Spacer(1, 6))
                flowables.append(make_heading(line[3:].strip(), style_h2, level=1))
                flowables.append(Image(img_path, width=440, height=264))
                flowables.append(Paragraph('<i>Figura 1: Razão de compressão SIREN vs gzip/lzma em texto PT. Valores abaixo de 1.0 significam expansão, não compressão.</i>', style_caption))
                i += 1
                continue

        if line.startswith('### ') and 'PSNR cai com o tamanho' in line:
            img_path = os.path.join(image_dir, 'chart_03_psnr_vs_size.png')
            if os.path.exists(img_path):
                flowables.append(make_heading(line[4:].strip(), style_h3, level=2))
                flowables.append(Spacer(1, 6))
                flowables.append(Image(img_path, width=440, height=264))
                flowables.append(Paragraph('<i>Figura 2: PSNR da reconstrução SIREN vs tamanho do texto. PSNR cai com o tamanho — a rede não consegue absorver a complexidade.</i>', style_caption))
                i += 1
                continue

        if line.startswith('### ') and 'Tempo de treinamento explode' in line:
            img_path = os.path.join(image_dir, 'chart_02_compression_time.png')
            if os.path.exists(img_path):
                flowables.append(make_heading(line[4:].strip(), style_h3, level=2))
                flowables.append(Spacer(1, 6))
                flowables.append(Image(img_path, width=440, height=264))
                flowables.append(Paragraph('<i>Figura 3: Custo temporal de compressão. SIREN é 3 a 5 ordens de magnitude mais lento que gzip.</i>', style_caption))
                i += 1
                continue

        if line.startswith('## ') and 'Resultados' in line and 'Imagem' in current_section:
            img_path = os.path.join(image_dir, 'chart_04_image_compression.png')
            if os.path.exists(img_path):
                flowables.append(make_heading(line[3:].strip(), style_h2, level=1))
                flowables.append(Spacer(1, 6))
                flowables.append(Image(img_path, width=440, height=264))
                flowables.append(Paragraph('<i>Figura 4: Comparação de compressão de imagem 32x32 estruturada. SIREN comprime contra raw RGB mas perde para PNG.</i>', style_caption))
                i += 1
                continue

        if line.startswith('## ') and 'Resultados' in line and 'Diret' in (current_section if 'Diret' in current_section else ''):
            img_path = os.path.join(image_dir, 'chart_05_directory_compression.png')
            if os.path.exists(img_path):
                flowables.append(make_heading(line[3:].strip(), style_h2, level=1))
                flowables.append(Spacer(1, 6))
                flowables.append(Image(img_path, width=440, height=264))
                flowables.append(Paragraph('<i>Figura 5: Compressão de diretório misto (10 arquivos, 10KB). gzip e lzma batem SIREN por 2-3x.</i>', style_caption))
                i += 1
                continue

        if line.startswith('## Capítulo 15'):
            img_path = os.path.join(image_dir, 'chart_06_architecture.png')
            if os.path.exists(img_path):
                flowables.append(make_heading(line[3:].strip(), style_h2, level=1))
                flowables.append(Spacer(1, 6))
                flowables.append(Image(img_path, width=480, height=280))
                flowables.append(Paragraph('<i>Figura 6: Arquitetura híbrida do Black Hole. Dados brutos são ingeridos, classificados, comprimidos com o codec ideal, e ejatados sob demanda via io_uring/DirectStorage.</i>', style_caption))
                i += 1
                continue

        if line.startswith('## Capítulo 13'):
            img_path = os.path.join(image_dir, 'chart_07_verdict.png')
            if os.path.exists(img_path):
                flowables.append(make_heading(line[3:].strip(), style_h2, level=1))
                flowables.append(Spacer(1, 6))
                flowables.append(Image(img_path, width=480, height=262))
                flowables.append(Paragraph('<i>Figura 7: Veredicto empírico consolidado. SIREN é viável apenas marginalmente para imagens pequenas; em todos os outros cenários é superado por compressores tradicionais.</i>', style_caption))
                i += 1
                continue

        # Code block fence
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                code_text = '\n'.join(code_buffer)
                # Create code paragraph
                # Escape and preserve formatting
                code_escaped = escape_md(code_text).replace('\n', '<br/>')
                # Use Preformatted-like approach
                from reportlab.platypus import Preformatted
                flowables.append(Preformatted(code_text, style_code))
                flowables.append(Spacer(1, 8))
                in_code_block = False
                code_buffer = []
                code_lang = None
            else:
                # Start code block
                in_code_block = True
                code_lang = line.strip()[3:].strip() or None
                code_buffer = []
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        # Horizontal rule
        if line.strip() == '---' or line.strip() == '***':
            flowables.append(Spacer(1, 12))
            # Horizontal line
            from reportlab.platypus import HRFlowable
            flowables.append(HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceBefore=4, spaceAfter=12))
            i += 1
            continue

        # Headers
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            flowables.append(make_heading(text, style_h1, level=0))
            i += 1
            continue
        if line.startswith('## '):
            text = line[3:].strip()
            flowables.append(make_heading(text, style_h2, level=1))
            i += 1
            continue
        if line.startswith('### '):
            text = line[4:].strip()
            flowables.append(make_heading(text, style_h3, level=2))
            i += 1
            continue

        # Tables
        if '|' in line and i + 1 < len(lines) and '---' in lines[i+1]:
            # Parse table
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            # Process table
            rows = []
            for tl in table_lines:
                if '---' in tl:
                    continue
                cells = [c.strip() for c in tl.strip('|').split('|')]
                rows.append(cells)
            if len(rows) >= 1:
                # Convert cells to Paragraphs
                table_data = []
                for r_idx, row in enumerate(rows):
                    row_paras = []
                    for cell in row:
                        if r_idx == 0:
                            # Header
                            p = Paragraph(f'<b>{md_inline(cell)}</b>', ParagraphStyle(
                                name=f'th{r_idx}', fontName='FreeSerif-Bold', fontSize=10,
                                textColor=colors.white, alignment=TA_CENTER, leading=13,
                            ))
                        else:
                            p = Paragraph(md_inline(cell), ParagraphStyle(
                                name=f'td{r_idx}', fontName='FreeSerif', fontSize=9.5,
                                textColor=TEXT_PRIMARY, alignment=TA_LEFT, leading=12,
                            ))
                        row_paras.append(p)
                    table_data.append(row_paras)
                # Compute column widths (equal distribution)
                n_cols = len(table_data[0])
                available = A4[0] - 40*mm  # margins
                col_w = available / n_cols
                col_widths = [col_w] * n_cols
                t = Table(table_data, colWidths=col_widths, hAlign='CENTER', repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'FreeSerif-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9.5),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, TABLE_ROW_ALT]),
                    ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
                ]))
                flowables.append(Spacer(1, 12))
                flowables.append(t)
                flowables.append(Spacer(1, 12))
            continue

        # Empty line - paragraph break
        if line.strip() == '':
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # Regular paragraph
        para_text = md_inline(line.strip())
        # Combine multiple consecutive non-empty lines into single paragraph
        while i + 1 < len(lines) and lines[i+1].strip() != '' and not lines[i+1].startswith('#') and not lines[i+1].startswith('|') and not lines[i+1].startswith('```') and lines[i+1].strip() != '---':
            i += 1
            para_text += ' ' + md_inline(lines[i].strip())
        flowables.append(Paragraph(para_text, style_body))
        i += 1

    return flowables


# === MAIN ===
def main():
    md_path = '/home/z/my-project/download/BlackHole_Whitepaper.md'
    pdf_path = '/home/z/my-project/download/BlackHole_Whitepaper.pdf'
    image_dir = '/home/z/my-project/download'

    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Strip the front matter (first big title block, since cover is separate)
    # Find first '#' heading
    lines = md_text.split('\n')
    # Skip lines until we reach "## Aviso de Honestidade Técnica"
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('## Aviso de Honestidade'):
            start_idx = i
            break
    body_md = '\n'.join(lines[start_idx:])

    # Build story
    story = []

    # Cover page
    story.extend(build_cover())
    story.append(PageBreak())

    # TOC
    toc = TableOfContents()
    toc.levelStyles = [style_toc1, style_toc2]
    story.append(Paragraph('Sumário', style_h1))
    story.append(Spacer(1, 12))
    story.append(toc)
    story.append(PageBreak())

    # Body
    story.extend(parse_markdown(body_md, image_dir))

    # Build PDF
    doc = TocDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=22*mm,
        bottomMargin=22*mm,
        title='Black Hole Whitepaper v1.0',
        author='Projeto Black Hole',
        subject='Compressão Neural Oportunista',
        creator='Z.ai',
    )

    doc.multiBuild(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f'PDF gerado: {pdf_path}')
    print(f'Tamanho: {os.path.getsize(pdf_path)} bytes')

if __name__ == '__main__':
    main()
