from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"C:\Users\abond\Desktop\WORK FILES\Systemic\Project 1")
WORK = ROOT / "work_quant_review"
OUT = ROOT / "deliverables" / "Quant_Project_1_Statistical_Notebook_Senior_Quant_Review.docx"
WORK.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)


# Resolved standard_business_brief preset.
PAGE_WIDTH_DXA = 12240
PAGE_HEIGHT_DXA = 15840
MARGIN_DXA = 1440
CONTENT_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120
CELL_MARGINS_DXA = {"top": 80, "bottom": 80, "start": 120, "end": 120}

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
NAVY = "0B2545"
INK = "1E293B"
MUTED = "5B6573"
LIGHT_GRAY = "F2F4F7"
BLUE_GRAY = "E8EEF5"
CALLOUT = "F4F6F9"
PALE_BLUE = "EAF3FA"
PALE_GREEN = "EAF5EF"
PALE_GOLD = "FFF7E2"
PALE_RED = "FCECEC"
GREEN = "1E6B49"
GOLD = "7A5A00"
RED = "9B1C1C"
WHITE = "FFFFFF"
BLACK = "000000"
BORDER = "CDD5DF"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def set_run_font(run, name="Calibri", size=11, color=INK, bold=None, italic=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = rgb(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")


def set_cell_margins(cell, margins=CELL_MARGINS_DXA):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge in ("top", "start", "bottom", "end"):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(margins[edge]))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_table_geometry(table, widths_dxa: Sequence[int], indent_dxa=TABLE_INDENT_DXA):
    assert sum(widths_dxa) == CONTENT_WIDTH_DXA
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(CONTENT_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        prevent_row_split(row)
        for idx, (cell, width) in enumerate(zip(row.cells, widths_dxa, strict=False)):
            cell.width = Inches(width / 1440)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")


def set_table_borders(table, color=BORDER, size="4"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def add_field(paragraph, instruction: str):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    return run


def add_custom_numbering(doc: Document):
    numbering = doc.part.numbering_part.element
    existing_abs = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))]
    existing_num = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))]
    next_abs = max(existing_abs, default=0) + 1
    next_num = max(existing_num, default=0) + 1

    def create_abstract(abs_id: int, num_fmt: str, text_value: str):
        abstract = OxmlElement("w:abstractNum")
        abstract.set(qn("w:abstractNumId"), str(abs_id))
        multi = OxmlElement("w:multiLevelType")
        multi.set(qn("w:val"), "singleLevel")
        abstract.append(multi)
        lvl = OxmlElement("w:lvl")
        lvl.set(qn("w:ilvl"), "0")
        start = OxmlElement("w:start")
        start.set(qn("w:val"), "1")
        fmt = OxmlElement("w:numFmt")
        fmt.set(qn("w:val"), num_fmt)
        txt = OxmlElement("w:lvlText")
        txt.set(qn("w:val"), text_value)
        suff = OxmlElement("w:suff")
        suff.set(qn("w:val"), "tab")
        ppr = OxmlElement("w:pPr")
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "num")
        tab.set(qn("w:pos"), "720")
        tabs.append(tab)
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), "720")
        ind.set(qn("w:hanging"), "360")
        spacing = OxmlElement("w:spacing")
        spacing.set(qn("w:after"), "160")
        spacing.set(qn("w:line"), "280")
        spacing.set(qn("w:lineRule"), "auto")
        ppr.extend([tabs, ind, spacing])
        rpr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        fonts.set(qn("w:ascii"), "Calibri")
        fonts.set(qn("w:hAnsi"), "Calibri")
        rpr.append(fonts)
        lvl.extend([start, fmt, txt, suff, ppr, rpr])
        abstract.append(lvl)
        numbering.append(abstract)

    def create_num(num_id: int, abs_id: int):
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(num_id))
        abs_ref = OxmlElement("w:abstractNumId")
        abs_ref.set(qn("w:val"), str(abs_id))
        num.append(abs_ref)
        numbering.append(num)

    create_abstract(next_abs, "bullet", "•")
    create_num(next_num, next_abs)
    bullet_num = next_num
    create_abstract(next_abs + 1, "decimal", "%1.")
    create_num(next_num + 1, next_abs + 1)
    decimal_num = next_num + 1
    return bullet_num, decimal_num


def apply_num(paragraph, num_id: int):
    ppr = paragraph._p.get_or_add_pPr()
    num_pr = ppr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        ppr.append(num_pr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    num_pr.extend([ilvl, num])


def configure_document(doc: Document):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.font.color.rgb = rgb(INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    normal.paragraph_format.widow_control = True

    heading_tokens = {
        "Heading 1": (16, BLUE, 16, 8),
        "Heading 2": (13, BLUE, 12, 6),
        "Heading 3": (12, DARK_BLUE, 8, 4),
        "Heading 4": (11, NAVY, 7, 3),
    }
    for name, (size, color, before, after) in heading_tokens.items():
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = rgb(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.05
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.widow_control = True

    for name, size, color, italic in (
        ("Report Subtitle", 14, DARK_BLUE, False),
        ("Report Kicker", 10, GOLD, False),
        ("Small Meta", 9.5, MUTED, False),
        ("Figure Caption", 9, MUTED, True),
        ("Table Citation", 9, MUTED, True),
        ("Callout Text", 10.5, INK, False),
        ("Glossary Term", 10.5, NAVY, False),
    ):
        if name not in styles:
            style = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        else:
            style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = rgb(color)
        style.font.italic = italic
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.05

    # A compact code style for paths and formulas.
    if "Inline Code Block" not in styles:
        code = styles.add_style("Inline Code Block", WD_STYLE_TYPE.PARAGRAPH)
    else:
        code = styles["Inline Code Block"]
    code.font.name = "Consolas"
    code._element.rPr.rFonts.set(qn("w:ascii"), "Consolas")
    code._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
    code.font.size = Pt(9)
    code.font.color.rgb = rgb(NAVY)
    code.paragraph_format.left_indent = Inches(0.25)
    code.paragraph_format.right_indent = Inches(0.15)
    code.paragraph_format.space_before = Pt(3)
    code.paragraph_format.space_after = Pt(6)
    code.paragraph_format.line_spacing = 1.0

    # Document metadata.
    doc.core_properties.title = "Senior Quantitative Review of Statistical Research Notebook"
    doc.core_properties.subject = "Quant Project 1 - Partner work after Section 6"
    doc.core_properties.author = "Senior Quantitative Review"
    doc.core_properties.keywords = "GC, quantitative research, feature engineering, backtest, POI"

    settings = doc.settings.element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")

    # Quiet running header and footer.
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_after = Pt(0)
    r = hp.add_run("QUANT PROJECT 1  |  SENIOR QUANT REVIEW")
    set_run_font(r, size=8.5, color=MUTED, bold=True)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.paragraph_format.space_before = Pt(0)
    r = fp.add_run("CONFIDENTIAL RESEARCH REVIEW   •   ")
    set_run_font(r, size=8, color=MUTED)
    pr = add_field(fp, "PAGE")
    set_run_font(pr, size=8, color=MUTED)


def add_heading(doc: Document, text: str, level: int):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    return p


def add_para(doc: Document, text: str = "", *, style=None, align=None, keep=False):
    p = doc.add_paragraph(style=style)
    if text:
        p.add_run(text)
    if align is not None:
        p.alignment = align
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def add_labeled(doc: Document, label: str, text: str, *, color=NAVY):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(label + " ")
    set_run_font(r, size=11, color=color, bold=True)
    r = p.add_run(text)
    set_run_font(r, size=11, color=INK)
    return p


def add_bullets(doc: Document, items: Iterable[str], bullet_num: int):
    for text in items:
        p = doc.add_paragraph()
        apply_num(p, bullet_num)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.167
        p.add_run(text)


def add_numbered(doc: Document, items: Iterable[str], decimal_num: int):
    for text in items:
        p = doc.add_paragraph()
        apply_num(p, decimal_num)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.167
        p.add_run(text)


def add_callout(doc: Document, label: str, text: str, *, tone="blue"):
    fill, accent = {
        "blue": (PALE_BLUE, DARK_BLUE),
        "green": (PALE_GREEN, GREEN),
        "gold": (PALE_GOLD, GOLD),
        "red": (PALE_RED, RED),
        "gray": (CALLOUT, MUTED),
    }[tone]
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [CONTENT_WIDTH_DXA])
    set_table_borders(table, color=accent, size="6")
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.10
    r = p.add_run(label + "  ")
    set_run_font(r, size=10.5, color=accent, bold=True)
    r = p.add_run(text)
    set_run_font(r, size=10.5, color=INK)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    return table


def add_table(
    doc: Document,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    widths_dxa: Sequence[int],
    *,
    alignments: Sequence[str] | None = None,
    header_fill=LIGHT_GRAY,
    font_size=9,
):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths_dxa)
    set_table_borders(table)
    alignments = alignments or ["left"] * len(headers)
    header = table.rows[0]
    set_repeat_table_header(header)
    for idx, text in enumerate(headers):
        cell = header.cells[idx]
        set_cell_shading(cell, header_fill)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        p.alignment = {
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
        }[alignments[idx]]
        r = p.add_run(str(text))
        set_run_font(r, size=font_size, color=NAVY, bold=True)
    for row_values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row_values):
            p = cells[idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.03
            p.alignment = {
                "left": WD_ALIGN_PARAGRAPH.LEFT,
                "center": WD_ALIGN_PARAGRAPH.CENTER,
                "right": WD_ALIGN_PARAGRAPH.RIGHT,
            }[alignments[idx]]
            r = p.add_run(str(value))
            set_run_font(r, size=font_size, color=INK)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)
    return table


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph(style="Figure Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = False
    p.add_run(text)
    return p


def add_figure(doc: Document, path: Path, caption: str, width=6.35):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    inline_shape = p.add_run().add_picture(str(path), width=Inches(width))
    # Keep the generated report usable with screen readers.  Word stores
    # alternative text on the drawing's non-visual ``docPr`` element.
    inline_shape._inline.docPr.set("descr", caption)
    inline_shape._inline.docPr.set("title", caption.split(".", 1)[0])
    add_caption(doc, caption)


def page_break(doc: Document):
    doc.add_page_break()


def find_font(bold=False, size=28):
    candidates = [
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_centered(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    x, y = xy
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def create_pipeline_chart(path: Path):
    w, h = 1800, 520
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = find_font(True, 40)
    label = find_font(True, 26)
    small = find_font(False, 21)
    draw_centered(d, (w / 2, 20), "Research progression after the Section 6 handoff", title, "#0B2545")
    stages = [
        ("6", "85 causal\nfeatures", "#E8EEF5"),
        ("7", "0 direction\n55 expansion", "#EAF3FA"),
        ("8", "15-feature\nfrozen set", "#EAF3FA"),
        ("9", "3/4 relative\nrange models", "#FFF7E2"),
        ("10", "benchmark\ntrade rules", "#FFF7E2"),
        ("11", "all variants\nrejected", "#FCECEC"),
        ("12", "filter adds no\nconfirmed value", "#FCECEC"),
    ]
    x0, gap, bw, bh, y = 55, 30, 215, 245, 130
    for idx, (num, text, fill) in enumerate(stages):
        x = x0 + idx * (bw + gap)
        d.rounded_rectangle((x, y, x + bw, y + bh), radius=20, fill=fill, outline="#7D8A99", width=3)
        d.ellipse((x + 74, y - 34, x + 140, y + 32), fill="#2E74B5")
        draw_centered(d, (x + 107, y - 22), num, label, "white")
        for line_idx, line in enumerate(text.split("\n")):
            draw_centered(d, (x + bw / 2, y + 72 + line_idx * 43), line, label if line_idx == 0 else small, "#1E293B")
        if idx < len(stages) - 1:
            ax = x + bw + 5
            ay = y + bh / 2
            d.line((ax, ay, ax + gap - 10, ay), fill="#7D8A99", width=5)
            d.polygon([(ax + gap - 10, ay), (ax + gap - 24, ay - 10), (ax + gap - 24, ay + 10)], fill="#7D8A99")
    draw_centered(d, (w / 2, 430), "Engineering completion is real; production readiness is not yet achieved.", small, "#5B6573")
    img.save(path, dpi=(200, 200))


def create_verdict_chart(path: Path):
    w, h = 1400, 600
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = find_font(True, 38)
    label = find_font(True, 27)
    value_font = find_font(True, 30)
    draw_centered(d, (w / 2, 20), "Section 7 feature-level verdicts", title, "#0B2545")
    data = [("Directional advance", 0, "#9B1C1C"), ("Expansion advance", 55, "#2E74B5"), ("Weak / unstable", 28, "#7A5A00")]
    maxv = 60
    base_x, top, bar_h, gap = 380, 140, 75, 72
    for i, (name, value, color) in enumerate(data):
        y = top + i * (bar_h + gap)
        d.text((35, y + 17), name, font=label, fill="#1E293B")
        d.rounded_rectangle((base_x, y, base_x + 850, y + bar_h), radius=14, fill="#EEF1F5")
        if value:
            d.rounded_rectangle((base_x, y, base_x + 850 * value / maxv, y + bar_h), radius=14, fill=color)
        d.text((base_x + 870, y + 15), str(value), font=value_font, fill=color)
    draw_centered(d, (w / 2, 545), "The important discovery is the absence of direction, not the number of significant tests.", find_font(False, 21), "#5B6573")
    img.save(path, dpi=(200, 200))


def create_backtest_chart(path: Path):
    w, h = 1500, 720
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = find_font(True, 38)
    label = find_font(False, 22)
    small = find_font(True, 20)
    draw_centered(d, (w / 2, 18), "Section 11 base-cost mean net R", title, "#0B2545")
    categories = ["Long / ungated", "Long / gated", "Short / ungated", "Short / gated"]
    dev = [-0.224, -0.268, -0.211, -0.260]
    validation = [-0.192, -0.281, -0.202, -0.207]
    chart_left, chart_top, chart_right, chart_bottom = 170, 115, 1420, 600
    zero_y = chart_top + 55
    d.line((chart_left, zero_y, chart_right, zero_y), fill="#53606D", width=3)
    d.text((30, zero_y - 12), "0.00", font=label, fill="#53606D")
    scale = 1350
    group_w = (chart_right - chart_left) / len(categories)
    for i, cat in enumerate(categories):
        cx = chart_left + group_w * (i + 0.5)
        for off, score, color, key in [(-35, dev[i], "#2E74B5", "Dev"), (35, validation[i], "#9B1C1C", "Val")]:
            x1, x2 = cx + off - 28, cx + off + 28
            y2 = zero_y + abs(score) * scale
            d.rounded_rectangle((x1, zero_y, x2, y2), radius=8, fill=color)
            draw_centered(d, (cx + off, y2 + 12), f"{score:.3f}", small, color)
        draw_centered(d, (cx, chart_bottom + 28), cat, label, "#1E293B")
    d.rectangle((950, 72, 980, 95), fill="#2E74B5")
    d.text((990, 68), "Development", font=label, fill="#1E293B")
    d.rectangle((1190, 72, 1220, 95), fill="#9B1C1C")
    d.text((1230, 68), "Validation", font=label, fill="#1E293B")
    draw_centered(d, (w / 2, 665), "All four direction/gate variants are negative after the declared 2.6-tick round-trip cost.", find_font(False, 21), "#5B6573")
    img.save(path, dpi=(200, 200))


def build_document():
    pipeline_chart = WORK / "pipeline.png"
    verdict_chart = WORK / "section7_verdicts.png"
    backtest_chart = WORK / "section11_backtest.png"
    create_pipeline_chart(pipeline_chart)
    create_verdict_chart(verdict_chart)
    create_backtest_chart(backtest_chart)

    doc = Document()
    configure_document(doc)
    bullet_num, decimal_num = add_custom_numbering(doc)

    # Cover: editorial_cover pattern, named override on standard_business_brief.
    p = doc.add_paragraph(style="Report Kicker")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(104)
    p.paragraph_format.space_after = Pt(16)
    r = p.add_run("SENIOR QUANTITATIVE REVIEW")
    set_run_font(r, size=10.5, color=GOLD, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("Statistical Research Notebook")
    set_run_font(r, size=29, color=NAVY, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run("Partner Work After Section 6")
    set_run_font(r, size=19, color=DARK_BLUE, bold=False)
    p = doc.add_paragraph(style="Report Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(30)
    p.add_run("Notebook stages 7-12, independent validation, market-logic audit, and remediation plan")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Review date: 20 July 2026")
    set_run_font(r, size=11, color=NAVY, bold=True)
    p = doc.add_paragraph(style="Small Meta")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(52)
    p.add_run("Prepared for project-partner review  |  Discovery instrument: GC  |  Research-only")

    add_callout(
        doc,
        "Bottom line",
        "The partner made substantial and commendable engineering progress, and the negative directional findings are valuable. The notebook is not yet production-safe: the opportunity score uses future same-day observations in its normalization, the hybrid filter is applied after some POI entries would already have occurred, and more than half of the ATR-derived order distances are off the GC tick grid.",
        tone="gold",
    )

    page_break(doc)

    add_heading(doc, "Executive verdict", 1)
    add_para(
        doc,
        "The work after Section 6 is a real improvement in research infrastructure. The partner carried the notebook from a validated 85-feature matrix through univariate screening, redundancy reduction, multivariate benchmarking, signal formalization, a chronological cost-aware backtest, and a first POI/statistical integration test. The code was moved into reusable modules, outputs were persisted, criteria were frozen in configuration objects, and negative results were reported honestly. That is materially better than a notebook that produces only attractive charts and an optimistic PnL number.",
    )
    add_para(
        doc,
        "The strongest defensible conclusion is also the simplest: the OHLCV/state feature inventory did not produce a stable directional edge. The apparent statistical strength lies in forecasting future range relative to current ATR, not in forecasting whether price will rise or fall. When benchmark long and short trades were sequenced with costs, every variant lost money. When the relative-range gate was added to POI events, it did not deliver confirmed incremental value.",
    )
    add_callout(
        doc,
        "Approval position",
        "Keep Sections 7-8 as useful research evidence and keep the Section 11 rejection of the four tested benchmark variants. Do not approve the Section 9/10 opportunity score for live or production research use until its transformation is made causal. Treat Section 12 as an informative negative experiment, not a valid causal integration verdict, until entry-time alignment is corrected.",
        tone="red",
    )

    add_table(
        doc,
        ["Dimension", "Senior assessment", "Meaning"],
        [
            ["Engineering implementation", "Strong", "Modular code, deterministic seeds, persisted artifacts, sequential execution, and broad synthetic coverage."],
            ["Research governance", "Good with a material weakness", "Final-test discipline was mostly respected, but Validation was reused for confirmation, representative selection, model assessment, and later decisions."],
            ["Statistical interpretation", "Mixed", "The no-direction result is credible; the strong expansion result is mainly ATR-relative and partly mechanically coupled to its denominator."],
            ["Market and execution logic", "Not yet acceptable", "Full-day ranks are noncausal, boundary-touch POI entries precede the gate, and stop/target prices are not tick-quantized."],
            ["Production readiness", "No", "The current outputs are research artifacts only. A causal rebuild and revalidation are required."],
        ],
        [2350, 2100, 4910],
        alignments=["left", "center", "left"],
        font_size=9.2,
    )

    add_heading(doc, "What the partner completed after our Section 6 handoff", 2)
    add_figure(doc, pipeline_chart, "Figure 1. Research progression from feature construction to the first hybrid test.")
    add_bullets(
        doc,
        [
            "Section 7 screened 83 evaluable predictors on Development and Validation and found zero directional advancers, 55 relative-expansion advancers, and 28 weak or unstable features.",
            "Section 8 reduced 55 expansion candidates to 30 clusters and a frozen 15-feature set, including one experimental feature.",
            "Section 9 reported that a ridge combination beat the ATR-only anchor in three of four session/horizon cells for ATR-relative future range; the New York 60-minute probability output was visibly overconfident.",
            "Section 10 created benchmark long and short candidates, with and without a top-20% relative-expansion gate, using a 1.5 ATR stop, a 2R target, and a 120-minute cap.",
            "Section 11 sequenced 86,353 trades across four variants and rejected every one under the 2.6-tick base-cost scenario.",
            "Section 12 combined POI direction candidates with the frozen relative-expansion gate and found no confirmed incremental value in filter form.",
            "Outside this notebook, the repository also now contains a bounded Branch A POI sequential backtest: S7P02 was positive frictionless but negative after base costs, so it was rejected.",
        ],
        bullet_num,
    )

    add_heading(doc, "What we can genuinely learn", 2)
    add_numbered(
        doc,
        [
            "The feature-engineering foundation was not wasted. It enabled a disciplined rejection rather than an anecdotal opinion about whether momentum, VWAP, candle geometry, or volume predicts direction.",
            "Generic one-minute GC OHLCV state is much better at describing relative movement regime than trade direction. Direction still needs a separate source of information or a different problem formulation.",
            "The very strong ATR-relative range relationships must not be described as absolute-volatility forecasts. Independent checks show that ATR(20) has near-zero or negative daily rank relationship with future range in ticks in several session/horizon cells even while its ATR-relative IC is approximately -0.53 to -0.77.",
            "Fixed transaction costs are decisive when stop distances are small. The gate selects quiet, low-ATR observations; 83.3% of gate-passing candidates hit the 10-tick minimum stop, so the 2.6-tick round trip consumes roughly 0.25R at the median.",
            "Sample floors and negative-result discipline prevented a seductive small-sample POI result from being promoted. The continuation-short hybrid improvement looked large on Development and Validation but failed the declared event/date floors and collapsed on the later read.",
        ],
        decimal_num,
    )

    add_heading(doc, "1. Scope, evidence base, and independent checks", 1)
    add_para(
        doc,
        "This review uses the supplied Section 6 detailed report as the handoff baseline. It then inspects the current statistical research context report, the executed notebook, the reusable Python modules, the synthetic tests, the persisted Parquet/CSV outputs, and the relevant POI integration code. The repository's statistical context report has in fact been updated through Section 12, despite the expectation that it was still at the Section 6 state; its completion entries agree with the notebook summaries.",
    )
    add_labeled(doc, "Primary notebook.", "notebooks/exploration/statistical_feature_research.ipynb")
    add_labeled(doc, "Baseline handoff.", "Supplied Section 6 - Detailed Feature Engineering Findings Report")
    add_labeled(doc, "Context record.", "project_docs/statistical_feature_research_context_report.md")
    add_labeled(doc, "Reader summaries.", "reports/statistical_research/summaries/section7_* through section12_*")
    add_labeled(doc, "Core implementation.", "src/statistical_research/*.py and src/research/hybrid_integration.py")

    add_heading(doc, "1.1 Reconciliation results", 2)
    add_table(
        doc,
        ["Independent check", "Observed result", "Assessment"],
        [
            ["Notebook structure", "181 cells: 100 Markdown and 81 code", "Reader-facing structure is strong."],
            ["Execution integrity", "All 81 code cells executed sequentially as counts 1-81; no saved error output", "Pass."],
            ["Kernel", ".venv-1, Python 3.14.5", "Matches the saved notebook environment."],
            ["Automated tests", "188 unittest tests passed in 26.79 seconds", "Strong engineering evidence; tests do not replace a market-timing audit."],
            ["Saved output reconciliation", "Headline counts and metrics match Parquet/CSV artifacts", "Pass."],
            ["Future-mutation test at model layer", "Failed: future same-day feature mutations changed earlier predictions", "Critical causal defect not covered by the existing tests."],
            ["GC tick-grid audit", "52.8% of candidate stops and 51.3% of targets were fractional ticks", "Market-execution defect."],
            ["Session-conditional redundancy", "Two retained clock features have Spearman rho = -1.0 within both sessions", "Section 8 did not finish the intended redundancy removal."],
        ],
        [2500, 3370, 3490],
        alignments=["left", "left", "left"],
        font_size=8.8,
    )

    page_break(doc)

    add_heading(doc, "2. Section 7 - Univariate feature evaluation", 1)
    add_para(
        doc,
        "Section 7 is where the partner first used the Section 6 features against future outcomes. This is the most important research step after the handoff because it determines whether the 85 engineered hypotheses contain evidence of direction, expansion, both, or neither. The section is well structured and, importantly, did not manufacture a directional success.",
    )
    add_figure(doc, verdict_chart, "Figure 2. Feature-level outcome of the Section 7 screen.")

    add_heading(doc, "7.0 Evaluation mandate", 2)
    add_labeled(doc, "What it did.", "Restricted the work to univariate screening. It explicitly prohibited signal construction, stop/target selection, modelling, and PnL estimation.")
    add_labeled(doc, "What it achieved.", "Preserved the intellectual boundary established in Section 6: feature definitions were evaluated after they were frozen rather than edited repeatedly until profitable.")
    add_labeled(doc, "What we learn.", "The later no-direction result is not the product of an obviously cherry-picked indicator library. It is a meaningful falsification of the current broad OHLCV/state hypothesis set.")

    add_heading(doc, "7.1 Frozen criteria and feature scope", 2)
    add_para(doc, "The evaluator screened 80 numeric and three Boolean predictors. The categorical entry-session field was used as a stratification variable, and weekday was excluded because the earlier baseline work did not authorize a weekday filter. Criteria were stored in Section7Config before the output was computed: Development q <= 0.10, observation/date floors, Validation sign agreement with at least 25% IC retention, Development quintile monotonicity of at least 0.8 in absolute value, and a two-tick directional top-minus-bottom spread in both partitions.")
    add_labeled(doc, "Senior view.", "Freezing the rules and recording exclusions is good. The two-tick directional hurdle is nevertheless below the later 2.6-tick base round trip and therefore is not an adequate economic hurdle on its own. Validation monotonicity and spread-sign agreement were reported but not required by code.")

    add_heading(doc, "7.2 Locked evaluation frame", 2)
    add_para(doc, "The section joined the feature matrix to the forward labels by observation ID, excluded all 162,224 Final-test rows, and retained a common complete-label population for every horizon. The final evaluation frame contains 419,393 rows: 302,774 Development, 116,619 Validation, 157,527 London, 261,866 New York, and 884 trading dates. A further 4,913 Development/Validation observations were removed because one or more horizon labels or ATR normalization values were unavailable.")
    add_labeled(doc, "Senior view.", "Using a common population makes horizon comparisons cleaner. The trade-off is mild complete-case selection near data gaps and boundaries; at roughly 1.2% of the Development/Validation population, this is not a major concern but should be documented as a sensitivity check.")

    add_heading(doc, "7.3 Daily rank-IC screen", 2)
    add_para(doc, "For each feature, outcome family, horizon, session, and partition, the engine computed a per-New-York-date Spearman information coefficient, then summarized the daily IC distribution with a t statistic, p-value, and a date-resampling bootstrap interval. The two outcome families were signed forward return in ATR units (direction) and future high-low range in ATR units (relative expansion). This produced 3,984 screen cells.")
    add_labeled(doc, "What it achieved.", "It correctly avoided treating hundreds of thousands of overlapping one-minute observations as independent. The trading date, not the minute, is the resampling block.")
    add_labeled(doc, "P1 statistical limitation.", "A within-day cross-sectional rank IC is a natural tool across many assets at one timestamp; here it is applied across sequential minutes of one asset. It can be dominated by time ordering, overlapping forward paths, and deterministic intraday structure. The extreme directional ICs that fail bucket economics are a warning: distance from session open reaches Validation IC -0.711 at 180 minutes, yet its Development and Validation top-bottom tick spreads have opposite signs. Fixed-clock-bin cross-date tests or causal pooled models with date-clustered inference would align better with a single-instrument time series.", color=RED)

    add_heading(doc, "7.4 Structural validation", 2)
    add_para(doc, "Twelve structural checks confirmed partition containment, expected cell counts, q-values only on Development, shortlist consistency, IC bounds, and no forward-looking column in the predictor list. Eighteen new synthetic evaluator tests covered planted signals, BH noise rejection, Development-only bucket edges, bootstrap determinism, Final-test rejection, session confinement, and the economic gate.")
    add_labeled(doc, "Senior view.", "This is strong engineering work. The missing test is a market-logic test: a feature/outcome setup that produces a high within-day IC solely because of clock order or overlapping paths should be recognized as a negative control.")

    add_heading(doc, "7.5 Development-fitted quintiles and long/short framing", 2)
    add_para(doc, "Each feature was divided into quintiles using Development-only session-specific cut points, which were applied unchanged to Validation. The signed-return analysis correctly recognized that long and short returns are algebraic opposites, so a negative relationship is short-side information rather than a separate target.")
    add_labeled(doc, "What it achieved.", "This prevented Validation values from moving bucket edges and avoided duplicating tests for long and short labels that contain the same information.")
    add_labeled(doc, "P1 limitation.", "The code requires strong monotonicity only on Development. It does not require Validation monotonicity or require the Validation top-bottom spread to have the same sign as Development. No directional feature advanced, so this weakness did not create a false directional candidate, but it should be corrected before another feature library is screened.", color=RED)

    add_heading(doc, "7.6 Stability by year", 2)
    add_para(doc, "The notebook reported per-year daily IC for leading features. That helps identify effects driven by one exceptional calendar year. Because Development ends in 2023 and Validation is 2024, the year view also exposes whether an apparent relationship survives the main regime transition.")
    add_labeled(doc, "Senior view.", "The output is useful but descriptive. A formal advancement rule does not require sign/effect stability in every year, and the date bootstrap resamples dates independently rather than preserving multi-day volatility clustering. Moving-block or stationary bootstrap over trading dates would give more conservative intervals.")

    add_heading(doc, "7.7 Session stability", 2)
    add_para(doc, "London and New York were evaluated separately in every cell. This was the correct market choice: their liquidity, scheduled-news exposure, time remaining in the day, and path to the 15:30 exit differ materially.")
    add_labeled(doc, "Senior view.", "Session separation is one of the strongest aspects of the notebook. Later clustering should have been performed the same way; pooling sessions during Section 8 allowed exact within-session duplicates to survive.")

    add_heading(doc, "7.8 Multiple-testing controls", 2)
    add_para(doc, "Benjamini-Hochberg q-values were computed on the Development screen within each outcome-family x session family of 498 related tests. This is preferable to uncorrected p-values and appropriately treats the six horizons as part of one family rather than six independent discoveries.")
    add_labeled(doc, "P1 limitation.", "The BH p-values come from a t test on daily ICs and therefore assume weak day-to-day dependence. Validation was then searched across 1,992 confirmation cells without a second multiplicity adjustment. The strong expansion effects are unlikely to vanish, but their apparent breadth - 55 of 83 features - is almost certainly overstated by redundancy and repeated confirmation searches.", color=RED)

    add_heading(doc, "7.9 Ranking and verdicts", 2)
    add_table(
        doc,
        ["Verdict", "Features", "Senior interpretation"],
        [
            ["ADVANCE_DIRECTIONAL", "0", "No generic directional feature cleared the full stack. This is the most important and credible finding."],
            ["ADVANCE_EXPANSION", "55", "Many features predict future range divided by current ATR; this is relative-path structure, not automatically a trading edge."],
            ["WEAK_UNSTABLE", "28", "Development significance did not survive confirmation/economic requirements."],
            ["NO_EVIDENCE", "0", "With the large sample, every feature was statistically detectable somewhere; this illustrates why p-values alone are not useful."],
        ],
        [2500, 1100, 5760],
        alignments=["left", "center", "left"],
    )
    add_para(doc, "The leading relative-expansion feature is ATR(20). In New York at 180 minutes, its daily IC is approximately -0.734 in Development and -0.763 in Validation. The negative sign means high current ATR is associated with a smaller future range when that future range is divided by current ATR. This is partly volatility mean reversion and partly a denominator relationship.")
    add_callout(doc, "Interpretation boundary", "Independent checks show that ATR(20)'s 60-minute relationship with future range in absolute ticks is only +0.035/-0.059 in London Development/Validation and +0.203/+0.202 in New York. At 180 minutes it is negative in both sessions. The notebook found relative range structure; it did not establish a universal absolute-volatility forecast.", tone="gold")

    add_heading(doc, "7.10 Completion gate", 2)
    add_para(doc, "The section saved univariate cell results, bucket summaries, yearly stability, the 358 passing confirmation cells, and feature-level verdicts. It correctly left the Final test locked and routed only the 55 relative-expansion advancers to redundancy analysis.")
    add_labeled(doc, "Overall Section 7 verdict.", "Substantial, useful progress. Accept the no-direction finding and retain the relative-expansion relationships as descriptive hypotheses. Do not yet call them tradable opportunity forecasts.", color=GREEN)

    page_break(doc)

    add_heading(doc, "3. Section 8 - Redundancy and incremental information", 1)
    add_para(doc, "Section 8 attempted to turn the broad Section 7 expansion result into a compact model-ready set. This was the right next stage. It reduced the feature count materially, but it did not fully solve redundancy because its clustering population and incremental test were not aligned with the later per-session model.")

    add_heading(doc, "8.1 Frozen contract and inputs", 2)
    add_para(doc, "The input was the same 419,393-row Development/Validation frame. All 55 ADVANCE_EXPANSION features were candidates, while the directional set was explicitly empty. Spearman correlations were fitted on Development only; hierarchical clustering used average linkage on one minus absolute correlation with a nominal |rho| threshold of 0.70. The primary representative score used 60-minute Validation IC, and incremental gates were defined for 60 and 180 minutes.")
    add_labeled(doc, "Commendation.", "The partner did not silently carry a directional set forward. Recording an empty frozen set is exactly right.", color=GREEN)
    add_labeled(doc, "P1 governance concern.", "Representative selection uses the best Validation IC. Validation had already confirmed the Section 7 shortlist and is later used again to judge the Section 9 model. This makes it a development resource, not a clean one-touch model-validation sample.", color=RED)

    add_heading(doc, "8.2 Run of clustering and incremental analysis", 2)
    add_para(doc, "The 55 candidates formed 30 clusters. One representative was chosen from each cluster, after which the global anchor ATR(20) was selected. Every other representative was tested using a per-date partial rank IC against the anchor. A representative advanced if its Development absolute partial IC was at least 0.05 and Validation agreed in sign, retained at least 25%, and cleared date floors.")
    add_labeled(doc, "What it achieved.", "The process removed obvious duplicates such as log volume, which looked strong alone but added no information after ATR state. Two of the three experimental representatives also failed the incremental gate.")

    add_heading(doc, "8.3 Validation checks", 2)
    add_para(doc, "Eight structural checks and ten new synthetic tests passed. They confirmed that each advancer belonged to one cluster, one representative was selected per cluster, the anchor remained in the frozen set, clustering used Development, and Final-test rows raised an error.")
    add_labeled(doc, "Senior view.", "The tests validate the implemented algorithm. They do not validate whether the algorithm is the correct redundancy definition for session-specific downstream models.")

    add_heading(doc, "8.4 Correlations and clusters", 2)
    add_para(doc, "The largest cluster contained seven ATR/realized-volatility features. That is a sensible result and confirms the feature inventory contains many alternative views of the same volatility state. The notebook also recognized that clock features were redundant by construction.")
    add_callout(doc, "Independent audit finding", "The frozen set still contains minute_from_execution_window_open and minutes_to_1530_forced_exit. Within London their Spearman correlation is exactly -1.0; within New York it is also exactly -1.0. They survived because clustering pooled the two sessions while Section 9 fits separate session models. Both features then receive equal-and-opposite coefficients, duplicating one clock factor and changing ridge-penalty geometry.", tone="red")

    add_heading(doc, "8.5 Incremental information beyond the anchor", 2)
    add_para(doc, "Fourteen non-anchor representatives showed partial IC beyond ATR(20) somewhere in the 60/180-minute, London/New York grid. This demonstrates that the relative-range target is not explained by ATR alone; clock, efficiency, choppiness, activity, and VWAP elasticity contain additional descriptive structure.")
    add_labeled(doc, "P1 limitation.", "Each feature was conditioned only on the anchor, not on the other selected representatives. A set of 14 features can therefore contain several mutually redundant non-anchor variables. A forward-selection or nested ridge-ablation procedure inside Development would test incremental value against the current selected set, not just ATR.", color=RED)

    add_heading(doc, "8.6 Frozen candidate set", 2)
    add_table(
        doc,
        ["Information group", "Retained features", "Senior interpretation"],
        [
            ["Volatility", "ATR(20), ATR ratios, current range / ATR", "Core relative-range state; partly coupled to the target denominator."],
            ["Clock", "Minutes from window open; minutes to 15:30", "Exact duplicates within each session; retain one."],
            ["Trend / path", "Efficiency 15/30/60, R-squared, choppiness, sign-change rate", "Plausible state information, but mutual incremental value is unproven."],
            ["Activity", "Relative volume 20; volume acceleration 5/20", "Adds some conditional information; must be tested in absolute and cost-adjusted units."],
            ["Experimental", "VWAP elasticity 30", "The only experimental feature to survive; interesting but not yet economically validated."],
        ],
        [1900, 3380, 4080],
        alignments=["left", "left", "left"],
        font_size=8.8,
    )
    add_labeled(doc, "Overall Section 8 verdict.", "Good reduction from 55 to 15 and a useful experimental-feature filter. Rerun clustering per session, remove exact duplicates, and perform joint incremental selection before treating the set as genuinely frozen.", color=GOLD)

    page_break(doc)

    add_heading(doc, "4. Section 9 - Multivariate research", 1)
    add_para(doc, "Section 9 asked whether the 15-feature ridge model predicts ATR-relative future range better than ATR(20) alone. The numerical answer was yes in three of four session/horizon cells. The causal answer is not yet established because the input normalization uses the full day's feature cross-section.")

    add_heading(doc, "9.0 Model mandate and specification", 2)
    add_para(doc, "Separate London and New York ridge regressions were fitted at 60 and 180 minutes. Features and the regression outcome were transformed into per-date rank z-scores. Ridge lambda was selected from 0.0001 to 10 by expanding-year walk-forward inside Development. The classification benchmark used ridge-stabilized logistic regression on the frozen expansion label.")
    add_labeled(doc, "What was right.", "Simple linear models preceded trees; chronological folds replaced random row splits; sessions were separated; the anchor-only benchmark prevented the full model from receiving credit for information already in ATR.", color=GREEN)

    add_heading(doc, "9.1 Validation checks", 2)
    add_para(doc, "Nine structural checks passed, including partition containment, anchor membership, lambda-grid selection, an embargo on every fold, low incomplete-row loss, AUC bounds, and one verdict per session/horizon. Twelve synthetic tests covered planted second drivers, anchor-only overfitting, coefficient recovery, chronology, classification, Final-test rejection, and determinism.")
    add_labeled(doc, "Senior view.", "This is strong software verification. The causality property of the rank transform was not tested.")

    add_heading(doc, "9.2 Regression benchmark and verdicts", 2)
    add_table(
        doc,
        ["Session", "Horizon", "Model IC", "ATR anchor", "Improvement", "Verdict"],
        [
            ["London", "60m", "0.593", "0.548", "+0.045", "Model advances"],
            ["London", "180m", "0.732", "0.723", "+0.008", "Anchor sufficient"],
            ["New York", "60m", "0.626", "0.545", "+0.081", "Model advances"],
            ["New York", "180m", "0.849", "0.761", "+0.088", "Model advances"],
        ],
        [1600, 1200, 1400, 1400, 1600, 2160],
        alignments=["left", "center", "right", "right", "right", "left"],
    )
    add_para(doc, "The paired date-bootstrap intervals on the three advancing improvements were above zero. London 180 minutes showed a statistically real but sub-threshold improvement and was correctly labelled anchor sufficient.")

    add_heading(doc, "9.3 Walk-forward selection detail", 2)
    add_para(doc, "Inside Development, the folds were 2021 training to 2022 testing, then 2021-2022 training to 2023 testing, with one trading date embargoed at the start of each test year. London 60 minutes selected lambda 0.01; the other three cells selected 0.0001, effectively close to an unregularized solution.")
    add_labeled(doc, "P1 governance concern.", "The feature list and its representatives had already been selected using Validation. Calling the model's Validation pass a one-touch out-of-sample test overstates independence. A nested walk-forward pipeline should select clusters, representatives, hyperparameters, and the model entirely within rolling training windows, reserving the next block for one decision.", color=RED)

    add_heading(doc, "9.4 Classification and calibration", 2)
    add_para(doc, "Validation AUCs ranged from 0.800 to 0.928 and exceeded the anchor in every cell. Brier scores also improved. The notebook correctly flagged New York 60-minute upper probabilities as overconfident: predictions around 0.65-0.82 corresponded to realized rates around 0.41-0.55. A later calibration module implements Development-fitted isotonic calibration, but the current notebook and Section 10 gate do not use it.")
    add_labeled(doc, "Senior view.", "AUC is a ranking metric and remains compatible with a badly calibrated probability. Do not use these probabilities for sizing. More fundamentally, their input ranking is noncausal, so recalibration alone cannot repair the model.")

    add_heading(doc, "9.5 Coefficients, saved outputs, and completion gate", 2)
    add_para(doc, "The section saved regression results, walk-forward scores, coefficients, classification metrics, calibration bins, and verdicts. ATR(20) carries the largest negative coefficient. The two exact clock duplicates receive equal-and-opposite coefficients, confirming that the model effectively counts one clock factor twice.")
    add_callout(
        doc,
        "P0 causal blocker - full-day rank normalization",
        "For each trading date, the model ranks every minute's feature value against all other minutes from that date. At 07:15, the score therefore depends on feature values from 08:00, 10:00, and 11:59. In an independent mutation audit on 29 December 2023, changing only later-half New York feature values changed earlier-half predictions by as much as 0.693 and flipped 4 of 150 early gate decisions. A prediction available only after the day is complete cannot be used to enter a trade during that day.",
        tone="red",
    )
    add_labeled(doc, "Overall Section 9 verdict.", "The model establishes an ex-post descriptive ranking of ATR-relative range. It does not yet establish an online opportunity forecast. Rebuild the input transform causally before relying on any IC, AUC, threshold, or coefficient from this section.", color=RED)

    page_break(doc)

    add_heading(doc, "5. Section 10 - Statistical signal construction", 1)
    add_heading(doc, "10.0 What was constructed", 2)
    add_para(doc, "Because no directional feature advanced, the partner did not claim that the model knew whether to go long or short. Instead, Section 10 created four negative-control variants: long and short benchmark directions, each ungated or gated by the top 20% of the 60-minute relative-range ridge prediction. Stops were 1.5 times decision ATR, clamped to 10-100 ticks; targets were 2R; maximum holding was 120 minutes; one R of risk was assumed.")
    add_labeled(doc, "Commendation.", "Calling the directions benchmarks rather than signals is honest and technically correct.", color=GREEN)
    add_labeled(doc, "Interpretation.", "This is not a discovered statistical strategy. It is a falsification harness asking whether opportunity timing alone can make an arbitrary direction profitable.")

    add_heading(doc, "10.1 Outputs and validation", 2)
    add_table(
        doc,
        ["Metric", "Result", "What it means"],
        [
            ["Candidate observations", "419,063", "Almost the full common evaluation frame."],
            ["Gate-passing observations", "85,365", "Approximately the declared top 20% in Development; about 21% in Validation."],
            ["Median stop", "10.575 ticks", "Off tick grid; practical GC orders require an integer tick distance."],
            ["All-candidate minimum clamp", "46%", "The 10-tick floor materially changes the ATR rule."],
            ["Gate-passing minimum clamp", "83.3%", "The gate disproportionately selects low-ATR states."],
            ["Median gate-passing ATR", "4.8 ticks", "The relative-expansion score is largely a quiet-regime selector."],
            ["Median cost load", "~0.25R gated", "Higher than the ~0.22R median outside the gate."],
        ],
        [2550, 1700, 5110],
        alignments=["left", "center", "left"],
    )
    add_callout(doc, "Market-logic finding", "The gate forecasts future range in units of current ATR, while the trade pays a fixed cost in ticks and applies a 10-tick stop floor. It selects low ATR because low ATR makes the future-range/ATR ratio large; the stop floor then prevents risk from shrinking proportionally. This raises cost per R and helps explain why the gated variants are worse.", tone="gold")
    add_callout(doc, "P0 execution defect - tick grid", "52.8% of saved stop distances and 51.3% of target distances are fractional GC ticks. The simulator fills these impossible prices exactly. Stops should be quantized conservatively to the 0.10 tick, targets recomputed on the tick grid, and a unit test should require every order price and distance to be an integer number of ticks.", tone="red")
    add_labeled(doc, "Horizon mismatch.", "The gate is trained on 60-minute relative range but the trade can hold 120 minutes. That may be a deliberate attempt to give a 2R target time to fill, but it is not economically derived from the model and should be aligned or treated as a predeclared sensitivity grid.")
    add_labeled(doc, "Overall Section 10 verdict.", "Useful negative-control construction, not an approved signal. Rebuild the gate target around tradable, cost-adjusted path outcomes before another sequential test.", color=GOLD)

    page_break(doc)

    add_heading(doc, "6. Section 11 - Independent sequential backtest", 1)
    add_heading(doc, "11.0 Simulator and execution rules", 2)
    add_para(doc, "The simulator processes candidates chronologically with one position at a time per variant. Entry timestamps are matched to actual GC bars and entry price is verified against the bar open. Exits are the stop, 2R target, 120-minute close, or 15:30 close. A bar that hits both stop and target is scored stop-first. Costs are reported frictionless, at 2.6 ticks round trip, and at 4.6 ticks round trip.")
    add_labeled(doc, "What was done well.", "The partner correctly separated event labels from a sequential backtest, enforced non-overlap, used conservative ambiguity treatment, and showed frictionless results alongside costs.", color=GREEN)
    add_labeled(doc, "Execution caveat.", "A stop crossed by a one-minute gap is filled at the stop level rather than the worse bar open/first executable price. This is optimistic tail treatment. Fixed slippage is not a substitute for a gap-aware stop rule.")

    add_heading(doc, "11.1 Validation checks", 2)
    add_para(doc, "Eight structural checks and thirteen synthetic path tests passed. The tests cover long/short symmetry, stop and target paths, ambiguity, time exit, forced exit, gate exclusion, overlap blocking, exact cost arithmetic, entry mismatch, Final-test rejection, and the two-partition verdict rule.")
    add_labeled(doc, "Senior view.", "The simulator core is a useful reusable asset. The tick-grid and causal-gate defects sit upstream and are not caught by these tests. The forced-exit structural check is also weaker than its name suggests because it does not directly assert an exit timestamp at or before 15:30, although the 120-minute cap makes late crossing unlikely in the current entry windows.")

    add_heading(doc, "11.2 Performance and verdict", 2)
    add_figure(doc, backtest_chart, "Figure 3. All four Section 11 variants lose after base costs in both partitions.")
    page_break(doc)
    add_table(
        doc,
        ["Variant", "Dev net R", "Val net R", "Verdict"],
        [
            ["Long, ungated", "-0.224", "-0.192", "Rejected"],
            ["Long, expansion gated", "-0.268", "-0.281", "Rejected"],
            ["Short, ungated", "-0.211", "-0.202", "Rejected"],
            ["Short, expansion gated", "-0.260", "-0.207", "Rejected"],
        ],
        [3800, 1500, 1500, 2560],
        alignments=["left", "right", "right", "center"],
    )
    add_para(doc, "There were 86,353 trades across the four variants. Frictionless mean R was approximately zero, as expected for directionless 2R/1R barriers with win rates near one third. Base costs then made every variant materially negative. The expansion-gated variants were not better, consistent with the gate selecting low-ATR states that carry a higher fixed cost per R.")
    add_labeled(doc, "Independent tick-rounding sensitivity.", "Rerunning the simulator in memory after rounding stops up to the next whole tick and rebuilding 2R targets left every verdict unchanged; base mean R remained approximately -0.20 to -0.28. The off-grid defect must still be fixed, but it does not rescue these variants.")
    add_labeled(doc, "Scope of rejection.", "The evidence rejects these four benchmark variants. It does not prove that every standalone statistical use of the feature set is impossible. No directional model existed, only one stop/target/hold family was tested, and non-directional uses such as absolute-volatility forecasting, options, target sizing, or no-trade suppression were not evaluated.", color=RED)
    add_labeled(doc, "Missing promised diagnostics.", "Relative to the context report's backtest standard, the section lacks formal parameter sensitivity, return-tail confidence intervals, a reader-facing session breakdown, and a true walk-forward strategy evaluation. An independent session breakdown is negative in every session/partition/variant, so the headline rejection remains intact.")
    add_labeled(doc, "Overall Section 11 verdict.", "Accept the rejection of the tested benchmark family. Retain the simulator after fixing tick quantization and stop-gap handling. Do not describe the result as rejection of all statistical research.", color=GREEN)

    page_break(doc)

    add_heading(doc, "7. Section 12 - Hybrid POI/statistical integration", 1)
    add_heading(doc, "12.0 Integration question and governance", 2)
    add_para(doc, "The first integration test used True POI retests as direction candidates and the frozen Section 9/10 relative-expansion model as a gate. Verdicts were based on Development and Validation, with minimum gated-event and trading-date floors. The Final-test report was produced separately after the verdicts were fixed. The outcome was the Branch A headline capped 60-minute R, robust-capped again at +/-5R for the comparison.")
    add_labeled(doc, "What was right.", "The integration boundary was crossed only after the branches were frozen, and the gate was not tuned to the POI sample. The partner also caught a population-mismatch issue in the gate reproduction and forced exact Development/Validation reproduction before proceeding.", color=GREEN)

    add_heading(doc, "12.1 Family results and verdicts", 2)
    add_table(
        doc,
        ["POI family", "Dev improvement", "Val improvement", "Verdict"],
        [
            ["Continuation long", "-0.073R", "-0.160R", "No incremental value"],
            ["Continuation short", "+1.170R", "+0.875R", "No value - sample floors"],
            ["Reversal long", "-1.676R", "-0.988R", "No incremental value"],
            ["Reversal short", "-0.713R", "+0.505R", "No incremental value"],
        ],
        [3400, 1600, 1600, 2760],
        alignments=["left", "right", "right", "left"],
    )
    add_para(doc, "Continuation short was the seductive case: only 410 Development and 241 Validation events, across 33 and 30 dates, produced the large improvements. Those counts failed the predeclared 500/300 event and 150/100 date floors. Refusing to advance it was the correct decision.")
    add_labeled(doc, "What we learn.", "The top-20% general-market relative-expansion gate passes only about 5% of POI events. POIs tend to occur after extension, while the gate favors relatively quiet current-ATR states. The two systems are structurally misaligned in filter form.")

    add_heading(doc, "12.2 One-time later-period read", 2)
    add_para(doc, "In the later-period report, continuation-short improvement fell to -0.024R; the best family improvement was only +0.044R. This supports the decision not to promote the small Development/Validation subset.")
    add_labeled(doc, "Governance qualification.", "The 2025-2026 period is not a completely untouched external test. Section 5 already exposed Final-test baseline behavior, and the POI branch had previously examined later-period candidate policies. It is a frozen follow-up for this specific gate, not a pristine project-wide test.")

    add_callout(
        doc,
        "P0 causal blocker - hybrid entry timing",
        "The hybrid joins the statistical decision bar to the POI retest bar. But the Branch A headline directional label uses a boundary-touch entry at the first-contact edge inside that same retest bar. The statistical feature vector is available only after the retest bar closes. The filter is therefore applied after the trade would already have entered. For boundary-touch policies, use features through the prior completed bar and set the gate before the retest bar; alternatively restrict integration to next-bar-confirmation entries, for which the retest-close information is available.",
        tone="red",
    )
    add_callout(
        doc,
        "P1 population-stability concern",
        "The True POI population changes dramatically: 13,643 Development retests across 352 dates, 11,487 Validation retests across 194 dates, and 153,906 later-period retests across 336 dates. That is roughly 39, 59, and 458 retests per active date. This order-of-magnitude shift needs a root-cause decomposition by detector version, geometry, volatility, contract/data coverage, and year before pooled inference is trusted.",
        tone="gold",
    )
    add_labeled(doc, "Event-level limitation.", "Section 12 compares overlapping event outcomes, not a sequential POI strategy. Date resampling helps with dependence, but capital use, competing retests, order conflicts, fill realism, and costs are not represented. It can screen a gate; it cannot establish portfolio value.")
    add_labeled(doc, "Overall Section 12 verdict.", "The observed gate-filter form shows no useful value and should not be promoted. Because the gate itself is noncausal and the boundary-touch alignment is after entry, rerun the integration before calling it a formally valid causal rejection. The negative evidence is informative; the experiment is not production-grade.", color=RED)

    add_heading(doc, "8. Cross-cutting senior assessment", 1)
    add_heading(doc, "8.1 Where the partner materially improved the project", 2)
    add_bullets(
        doc,
        [
            "Converted notebook logic into reusable modules instead of leaving opaque cell-local code.",
            "Created frozen configuration contracts and deterministic random seeds for each stage.",
            "Separated Development, Validation, and Final-test roles mechanically and raised on locked rows in the main evaluators.",
            "Used date-level evidence and block resampling rather than pretending every overlapping minute was independent.",
            "Added false-discovery control and explicit sample/date/economic gates.",
            "Preserved negative findings: zero directional features, rejected benchmark strategy, and rejected first hybrid filter.",
            "Built a chronological single-position simulator that is much closer to actual strategy testing than event-level averages.",
            "Maintained a readable notebook narrative and updated the context/summary artifacts through the new work.",
        ],
        bullet_num,
    )

    add_heading(doc, "8.2 Where lack of market knowledge shows", 2)
    add_bullets(
        doc,
        [
            "Using a full-day cross-sectional normalization for an intraday decision score. This is a classic cross-sectional-quant technique misapplied to a single-asset time series.",
            "Treating ATR-relative future range as generic 'opportunity' without first reconciling absolute ticks, fixed costs, stop floors, direction, and barrier ordering.",
            "Allowing half-tick and other fractional-tick stop/target distances in a GC futures simulator.",
            "Applying a completed-retest-bar gate to POI orders that would have filled inside the retest bar.",
            "Keeping two clock variables that are exactly the same factor within each separately fitted session model.",
            "Closing the 'standalone statistical system' too broadly after testing arbitrary benchmark directions rather than an evidence-based directional signal.",
            "Not decomposing the enormous post-2024 change in POI/retest event density before interpreting the later-period result.",
        ],
        bullet_num,
    )

    add_heading(doc, "8.3 Claims that are safe versus unsafe", 2)
    add_table(
        doc,
        ["Safe to say", "Do not say yet"],
        [
            ["The 85-feature matrix is a strong, tested research foundation.", "The feature pipeline is production-ready."],
            ["No feature cleared the declared directional advancement gates.", "There is no directional information anywhere in GC."],
            ["Many features explain future range relative to current ATR.", "The model forecasts absolute volatility or profitable opportunity."],
            ["The four Section 11 benchmark variants lose after costs.", "Every standalone statistical strategy has been disproved."],
            ["The tested POI gate did not show confirmed incremental value.", "All forms of POI/statistical integration are closed."],
            ["Small-sample floors prevented promotion of a false-looking winner.", "The later-period report is a pristine untouched final test."],
        ],
        [4680, 4680],
        alignments=["left", "left"],
        font_size=9,
    )

    add_heading(doc, "9. Required remediation plan", 1)
    add_para(doc, "The correct response is not to discard the partner's work. Keep the feature matrix, tests, summaries, and simulator, then repair the decision-time contract and rerun the affected stages. The order matters.")

    add_heading(doc, "P0 - repair before any further model or integration claim", 2)
    add_numbered(
        doc,
        [
            "Replace full-day rank z-scores with a causal transformation. Preferred baseline: Development-fitted robust location/scale parameters by session and 15-minute clock bin, frozen for later data. A secondary sensitivity can use a past-only expanding percentile within the current session. Add a prediction-layer future-mutation test: changing any feature after time t must leave the score at t unchanged.",
            "Correct hybrid availability. For boundary-touch entries, compute the statistical gate from information through the bar before the retest and require the order to be authorized before the retest bar opens. For next-bar confirmation entries, retest-close features are allowed. Store feature_availability_timestamp, order_decision_timestamp, and entry_timestamp, then assert availability <= decision <= entry.",
            "Quantize all futures order levels. Convert ATR risk to integer ticks using a documented conservative rounding rule, calculate stop/target prices from integer ticks, and assert every entry, stop, target, and exit price lies on the 0.10 GC grid. Model gap-through-stop fills at the worse of the stop and next executable price.",
        ],
        decimal_num,
    )

    add_heading(doc, "P1 - redesign the research target and validation", 2)
    add_numbered(
        doc,
        [
            "Replace the generic future-range/ATR target with tradable path targets: probability that a specified R target is hit before the stop, cost-adjusted expected R, MFE and MAE conditional on the chosen stop, absolute future range in ticks, and a no-trade label when expected movement cannot cover costs.",
            "Run redundancy per downstream model population. Cluster London and New York separately or cluster on the maximum absolute within-session correlation. Remove exact clock duplicates. Test each candidate's incremental value against the current selected set, not only against ATR(20).",
            "Use nested chronological validation. Feature/cluster selection, scaling, hyperparameters, and probability calibration should occur inside rolling Development folds. Reserve a later block for one model decision. If 2024 remains Validation, do not use it simultaneously to select representatives and certify the final model.",
            "Align the prediction horizon, holding cap, stop family, and target family. Predeclare a small economically motivated grid inside Development, then lock one configuration before the next block. Report whether results survive whole-tick rounding and reasonable slippage changes.",
            "Use date-clustered or moving-block inference on daily outcomes, require Validation monotonicity and sign-consistent bucket spreads, and set economic thresholds above the actual all-in cost plus a safety margin.",
        ],
        decimal_num,
    )

    add_heading(doc, "P2 - hardening and transfer", 2)
    add_numbered(
        doc,
        [
            "Investigate the 2025-2026 POI population explosion by year, contract, volatility regime, geometry type, detector path, and source-data coverage. Freeze a population definition only after the shift is explained.",
            "Add reader-facing performance by session, year, exit reason, volatility regime, and stop-size bucket; include confidence intervals, tail quantiles, drawdown duration, and dependence-aware daily equity statistics.",
            "Validate the frozen causal feature relationships on MGC only after the GC redesign is complete. Use MGC as a transfer test, not another tuning sample.",
            "Integrate the isotonic calibration module only after causal prediction is fixed; calibrate inside Development folds and assess calibration on a truly held-out chronological block.",
            "Retain the Branch A S7P02 cost-aware rejection as separate evidence. Do not use its later-period outcome to tune the repaired Branch B gate.",
        ],
        decimal_num,
    )

    add_heading(doc, "9.1 Acceptance tests for the repaired pipeline", 2)
    add_table(
        doc,
        ["Gate", "Minimum acceptance condition"],
        [
            ["Causality", "Future mutation after t changes neither transformed features, prediction, gate, nor order decision at t."],
            ["Availability", "Every feature timestamp is at or before the order-decision timestamp; every order decision is at or before entry."],
            ["Tick grid", "100% of order and fill prices are integer multiples of 0.10 within strict tolerance."],
            ["Validation independence", "The reported holdout block did not select features, clusters, hyperparameters, scalers, or calibration."],
            ["Economic target", "The predicted quantity maps directly to barrier ordering, cost coverage, or executable sizing/no-trade logic."],
            ["Robustness", "Direction/effect, monotonicity, and cost-adjusted outcome survive session, year, and reasonable execution sensitivities."],
            ["Population stability", "Large event-rate changes are explained or explicitly stratified; no unexplained detector/version shift remains."],
        ],
        [2300, 7060],
        alignments=["left", "left"],
        font_size=9,
    )

    page_break(doc)

    add_heading(doc, "10. Final answer to the project questions", 1)
    add_heading(doc, "Has the partner improved our work?", 2)
    add_para(doc, "Yes. The project now has a far more disciplined research pipeline, substantially better test coverage, clear output lineage, a sequential simulator, and defensible negative findings. The partner's strongest contribution is not a profitable model; it is the conversion of the Section 6 hypothesis inventory into a process that can reject weak ideas and preserve the evidence.")

    add_heading(doc, "Was everything done correctly for the market problem?", 2)
    add_para(doc, "No. The software is better than the market-time logic. The same-day rank transform is unavailable online; the hybrid gate is later than some POI entries; order distances violate the tick grid; and the relative-range objective is not equivalent to cost-adjusted tradable opportunity. Passing tests confirm the code matches its implementation, not that the implementation matches the market.")

    add_heading(doc, "What findings improve the project?", 2)
    add_bullets(
        doc,
        [
            "The current generic feature library should not be used to choose long versus short direction.",
            "Relative movement regime is forecastable, but the target must be redesigned in absolute/cost-adjusted execution units before it is useful.",
            "The 10-tick floor and fixed 2.6-tick cost dominate quiet-regime trades; any gate must explicitly include cost per R.",
            "POI direction and generic relative-expansion state do not combine well through a blunt top-20% filter.",
            "A large apparent mean improvement on a few dozen dates is not enough; the Section 12 floors were valuable and should be retained.",
            "The project should now prioritize causal timing and target design, not add more indicators or more complex models.",
        ],
        bullet_num,
    )
    add_callout(doc, "Recommended project decision", "Pause new feature invention. Repair Sections 9-12 in the order listed, preserve the negative results, and authorize no production rule until the causal, tick-grid, and entry-timing acceptance tests pass.", tone="blue")

    page_break(doc)

    add_heading(doc, "Appendix A - Detailed stage scorecard", 1)
    add_table(
        doc,
        ["Stage", "Delivered", "Keep", "Rerun / qualify"],
        [
            ["Section 7", "83-feature univariate screen", "Zero directional advancers; Development-only bins; session split", "Use time-series-appropriate inference; strengthen Validation and economic gates"],
            ["Section 8", "55 to 15 features", "Broad redundancy reduction; experimental filter", "Per-session clustering; remove exact duplicates; joint incremental selection"],
            ["Section 9", "Ridge/logistic relative-range models", "Simple benchmark design and anchor comparison", "Entire model after causal scaling; nested validation"],
            ["Section 10", "Benchmark rules and gate", "Honest direction labels as benchmarks", "Causal gate; tradable target; tick-quantized levels; horizon alignment"],
            ["Section 11", "86,353-trade sequential test", "Reject all four tested variants; retain simulator", "Gap-aware fills; richer diagnostics; use only causal candidates"],
            ["Section 12", "First POI gate-filter integration", "No observed incremental value; retain sample floors", "Correct entry-time availability and causal gate before formal conclusion"],
        ],
        [1150, 2080, 2860, 3270],
        alignments=["left", "left", "left", "left"],
        font_size=8.5,
    )

    add_heading(doc, "Appendix B - Key artifacts reviewed", 1)
    artifacts = [
        "notebooks/exploration/statistical_feature_research.ipynb",
        "project_docs/statistical_feature_research_context_report.md",
        "src/statistical_research/feature_evaluation.py",
        "src/statistical_research/feature_redundancy.py",
        "src/statistical_research/multivariate.py",
        "src/statistical_research/signal_construction.py",
        "src/statistical_research/sequential_backtest.py",
        "src/research/hybrid_integration.py",
        "data/processed/statistical_research/univariate_results_gc.parquet",
        "data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet",
        "data/processed/statistical_research/multivariate_regression_gc.parquet",
        "data/processed/statistical_research/signal_candidates_gc.parquet",
        "data/processed/statistical_research/backtest_performance_gc.parquet",
        "data/processed/statistical_research/hybrid_family_results_gc.parquet",
        "reports/statistical_research/summaries/section7_univariate_evaluation_summary.md",
        "reports/statistical_research/summaries/section8_redundancy_summary.md",
        "reports/statistical_research/summaries/section9_multivariate_summary.md",
        "reports/statistical_research/summaries/section10_signal_construction_summary.md",
        "reports/statistical_research/summaries/section11_sequential_backtest_summary.md",
        "reports/statistical_research/summaries/section12_hybrid_integration_summary.md",
    ]
    for item in artifacts:
        p = doc.add_paragraph(style="Inline Code Block")
        p.add_run(item)

    add_heading(doc, "Appendix C - Glossary", 1)
    glossary = [
        ("ATR", "Average true range; a recent volatility measure expressed in price or ticks."),
        ("Information coefficient (IC)", "A correlation between a feature ranking and a future outcome ranking. Here it is calculated within each trading date."),
        ("Benjamini-Hochberg", "A procedure that controls the expected false-discovery proportion across a family of many tests."),
        ("R", "Return measured in units of initial stop risk. A +2R target earns twice the stop distance before costs."),
        ("MFE / MAE", "Maximum favorable / adverse excursion along a future path."),
        ("Ridge regression", "A linear model with a coefficient penalty that reduces instability from correlated predictors."),
        ("AUC", "A ranking metric for binary outcomes. It does not guarantee calibrated probabilities or profitability."),
        ("Brier score", "Mean squared error of predicted probabilities."),
        ("Calibration", "Agreement between predicted probabilities and realized frequencies."),
        ("Causal / available", "A value computable using only information known by the decision timestamp."),
        ("Tick grid", "The discrete price increments permitted by the contract. GC trades in 0.10-point increments."),
        ("Boundary-touch entry", "A resting order filled when price touches the POI edge during the retest bar."),
        ("Next-bar confirmation", "An entry after the retest bar completes, usually at the next bar open."),
    ]
    for term, definition in glossary:
        add_labeled(doc, term + ".", definition)

    # Final structural polish: keep table header rows and ensure all sections use the preset.
    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.header_distance = Inches(0.492)
        section.footer_distance = Inches(0.492)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build_document()
