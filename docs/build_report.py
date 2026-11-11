"""Genere ``docs/Rapport_Projet.pdf`` a partir de ``docs/Rapport_Projet.md``.

Contribution CS30. Le script ne contient aucun resultat : il ne fait que mettre
en page la source Markdown (titres, paragraphes, listes et tableaux). Les
chiffres restent donc verifiables dans le fichier Markdown versionne.

Usage :
    python docs/build_report.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DOCS_DIR = Path(__file__).resolve().parent
SOURCE = DOCS_DIR / "Rapport_Projet.md"
OUTPUT = DOCS_DIR / "Rapport_Projet.pdf"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2.0 * cm
AVAILABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN

NAVY = colors.HexColor("#1F3864")
LIGHT_BLUE = colors.HexColor("#D9E2F3")
GREY = colors.HexColor("#7F7F7F")
ROW_ALT = colors.HexColor("#F5F7FA")


# ---------------------------------------------------------------------------
# Mise en page
# ---------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=NAVY,
            spaceAfter=10,
            alignment=0,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12.5,
            leading=17,
            textColor=colors.HexColor("#404040"),
            spaceAfter=18,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#2E5496"),
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.6,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.2,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.6,
            alignment=0,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.6,
            alignment=0,
        ),
    }


def _inline(text: str) -> str:
    """Convertit le sous-ensemble Markdown utilise en balises ReportLab."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


# ---------------------------------------------------------------------------
# Lecture du Markdown
# ---------------------------------------------------------------------------


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if _is_separator(line):
            continue
        rows.append(_split_row(line))
    return rows


def parse_markdown(text: str) -> list[tuple[str, object]]:
    """Decoupe le Markdown en blocs simples, sans dependance externe."""
    lines = text.splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
            index += 1
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            index += 1
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            index += 1
        elif stripped == "---":
            blocks.append(("rule", ""))
            index += 1
        elif line.lstrip().startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            blocks.append(("table", _parse_table(table_lines)))
        elif re.match(r"^\s*[-*] ", line):
            items = []
            while index < len(lines) and re.match(r"^\s*[-*] ", lines[index]):
                items.append(re.sub(r"^\s*[-*] ", "", lines[index]).strip())
                index += 1
            blocks.append(("bullets", items))
        elif re.match(r"^\s*\d+\. ", line):
            items = []
            while index < len(lines) and re.match(r"^\s*\d+\. ", lines[index]):
                items.append(re.sub(r"^\s*\d+\. ", "", lines[index]).strip())
                index += 1
            blocks.append(("numbered", items))
        else:
            paragraph = []
            while index < len(lines):
                candidate = lines[index]
                if not candidate.strip():
                    break
                if (
                    candidate.startswith("#")
                    or candidate.lstrip().startswith("|")
                    or re.match(r"^\s*[-*] ", candidate)
                    or re.match(r"^\s*\d+\. ", candidate)
                ):
                    break
                paragraph.append(candidate.strip())
                index += 1
            blocks.append(("para", " ".join(paragraph)))

    return blocks


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------


def _column_widths(rows: list[list[str]]) -> list[float]:
    n_columns = len(rows[0])
    weights = []
    for column in range(n_columns):
        longest = max(len(re.sub(r"<[^>]+>", "", str(row[column]))) for row in rows)
        weights.append(min(max(longest, 8), 34))
    total = sum(weights)
    return [AVAILABLE_WIDTH * weight / total for weight in weights]


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    n_columns = len(rows[0])
    data = []
    for row_index, row in enumerate(rows):
        # Certaines lignes Markdown peuvent etre incompletes : on complete.
        row = row + [""] * (n_columns - len(row))
        style = styles["cell_header"] if row_index == 0 else styles["cell"]
        data.append([Paragraph(_inline(str(cell)), style) for cell in row])

    table = Table(data, colWidths=_column_widths(rows), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A6A6A6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
            ]
        )
    )
    return table


def build_story(blocks: list[tuple[str, object]], styles: dict[str, ParagraphStyle]):
    story = [Spacer(1, 1.0 * cm)]
    metadata_seen = False
    first_heading = True

    for kind, value in blocks:
        if kind == "title":
            story.append(Paragraph(_inline(str(value)), styles["title"]))
        elif kind == "h2":
            if first_heading:
                story.append(Paragraph(_inline(str(value)), styles["subtitle"]))
                first_heading = False
            else:
                story.append(Paragraph(_inline(str(value)), styles["h2"]))
        elif kind == "h3":
            story.append(Paragraph(_inline(str(value)), styles["h3"]))
        elif kind == "rule":
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.6,
                    color=colors.HexColor("#BFBFBF"),
                    spaceBefore=4,
                    spaceAfter=8,
                )
            )
        elif kind == "para":
            story.append(Paragraph(_inline(str(value)), styles["body"]))
        elif kind == "bullets":
            for item in value:
                story.append(
                    Paragraph(_inline(str(item)), styles["bullet"], bulletText="\u2022")
                )
            story.append(Spacer(1, 4))
        elif kind == "numbered":
            for number, item in enumerate(value, start=1):
                story.append(
                    Paragraph(
                        f"{number}. {_inline(str(item))}",
                        styles["bullet"],
                    )
                )
            story.append(Spacer(1, 4))
        elif kind == "table":
            rows = value
            table = _table(rows, styles)
            story.append(KeepTogether([table]) if len(rows) <= 12 else table)
            story.append(Spacer(1, 9))
            if not metadata_seen and rows and rows[0] and rows[0][0] == "Champ":
                metadata_seen = True
                story.append(PageBreak())

    return story


def _on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#BFBFBF"))
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 1.45 * cm, PAGE_WIDTH - MARGIN, 1.45 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, 1.05 * cm, "Rapport de mini-projet Data Mining - Startup Success Prediction")
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 1.05 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    blocks = parse_markdown(SOURCE.read_text(encoding="utf-8"))
    styles = _styles()

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.0 * cm,
        title="Rapport de mini-projet Data Mining - Startup Success Prediction",
        author="CS30",
        subject="EDA, tests statistiques et resultats de classification",
        creator="docs/build_report.py",
    )
    document.build(build_story(blocks, styles), onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"PDF genere : {OUTPUT} ({OUTPUT.stat().st_size:,} octets)")


if __name__ == "__main__":
    main()
