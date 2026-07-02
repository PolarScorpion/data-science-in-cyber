"""Build the required final PDF report from existing project documentation.

The script is deterministic and does not read raw data, train models, or modify
analysis outputs. It uses the existing Markdown reports as the source of truth
for metrics, caveats, and file references.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "final_report.pdf"

GITHUB_BRANCH_URL = (
    "https://github.com/PolarScorpion/data-science-in-cyber/tree/cyber-final-project"
)

SOURCE_FILES = [
    "README.md",
    "notebooks/main_analysis.ipynb",
    "reports/final_report.md",
    "reports/final_synthesis.md",
    "reports/behavioral_features.md",
    "reports/baseline_modeling.md",
    "reports/eda_findings.md",
    "reports/reproducibility_audit.md",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_table_after_heading(markdown: str, heading: str) -> list[list[str]]:
    marker = f"## {heading}"
    start = markdown.index(marker)
    segment = markdown[start:]
    next_heading = segment.find("\n## ", len(marker))
    if next_heading != -1:
        segment = segment[:next_heading]

    table_lines = [line.strip() for line in segment.splitlines() if line.startswith("|")]
    rows: list[list[str]] = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)

    if not rows:
        raise ValueError(f"No Markdown table found after heading: {heading}")
    return rows


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def bullet_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(paragraph(item, style), leftIndent=12) for item in items],
        bulletType="bullet",
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=8,
    )


def markdown_table(rows: list[list[str]], style: ParagraphStyle, col_widths: list[float]) -> Table:
    converted = [[paragraph(cell, style) for cell in row] for row in rows]
    table = Table(converted, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b7c0cc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_section(story: list, title: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Spacer(1, 0.10 * inch))
    story.append(paragraph(title, styles["SectionTitle"]))
    story.append(Spacer(1, 0.06 * inch))


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = landscape(letter)
    canvas.setStrokeColor(colors.HexColor("#d0d7de"))
    canvas.line(doc.leftMargin, 0.45 * inch, width - doc.rightMargin, 0.45 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4f5b66"))
    canvas.drawString(doc.leftMargin, 0.28 * inch, "Data Science in Cybersecurity Final Project")
    canvas.drawRightString(width - doc.rightMargin, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(relative)

    readme_text = read_text(README)
    main_result_rows = parse_table_after_heading(readme_text, "Main result")
    workflow_rows = parse_table_after_heading(readme_text, "Implementation workflow summary")

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(letter),
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.65 * inch,
        title="Data Science in Cybersecurity Final Project Report",
        author="PolarScorpion",
        subject="Fraud detection reproducibility and critical evaluation",
    )

    sample = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {
        "Title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#16324f"),
            spaceAfter=10,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4f5b66"),
            spaceAfter=14,
        ),
        "SectionTitle": ParagraphStyle(
            "SectionTitle",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=12.2,
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=8.8,
        ),
        "Meta": ParagraphStyle(
            "Meta",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=11,
            textColor=colors.HexColor("#4f5b66"),
            spaceAfter=4,
        ),
    }

    story: list = []
    story.append(paragraph("Data Science in Cybersecurity Final Project", styles["Title"]))
    story.append(
        paragraph(
            "Required PDF Report - Fraud Detection Handbook reproducibility and critical evaluation",
            styles["Subtitle"],
        )
    )
    story.append(paragraph(f"GitHub branch: {GITHUB_BRANCH_URL}", styles["Meta"]))
    story.append(
        paragraph(
            "Key files: README.md; notebooks/main_analysis.ipynb; reports/final_report.md; "
            "reports/final_synthesis.md; reports/behavioral_features.md; "
            "reports/baseline_modeling.md; reports/eda_findings.md; "
            "reports/reproducibility_audit.md",
            styles["Meta"],
        )
    )

    add_section(story, "1. Summary", styles)
    story.append(
        paragraph(
            "This project reproduces and critically evaluates a fraud-detection workflow from "
            "the Fraud Detection Handbook using its public simulated transaction dataset. The "
            "scope is limited to simulated transaction data, baseline modeling, temporal "
            "validation, fraud-appropriate metrics, and behavioral feature engineering.",
            styles["Body"],
        )
    )
    story.append(
        paragraph(
            "The central research question is whether the handbook's conclusions about temporal "
            "validation, metric choice, and behavioral features remain supported under the "
            "simulated fraud scenarios, and how far those findings can generalize beyond data "
            "generated by known rules.",
            styles["Body"],
        )
    )
    story.append(paragraph("Project workflow summary", styles["Body"]))
    story.append(markdown_table(workflow_rows, styles["Small"], [0.65 * inch, 7.4 * inch]))

    add_section(story, "2. Critical Evaluation", styles)
    story.append(
        paragraph(
            "The strongest evidence is internal reproducibility on a public synthetic benchmark, "
            "not proof of production fraud performance. The train, validation, and test periods "
            "are chronological, which prevents random temporal leakage and better matches the "
            "online scoring setting.",
            styles["Body"],
        )
    )
    story.append(
        bullet_list(
            [
                "Fraud prevalence is below 1%, so AP / PR-AUC is the primary metric.",
                "Accuracy is misleading and is not used as the main metric.",
                "Thresholds are selected on validation only, and final metrics are evaluated on the held-out test period only.",
                "Synthetic data may overstate deployability because engineered features can align with the simulator rules that generated the labels.",
                "The project does not claim production readiness or real-world fraud performance.",
            ],
            styles["Body"],
        )
    )

    add_section(story, "3. Feature Engineering Analysis", styles)
    story.append(
        paragraph(
            "The baseline feature set uses transaction amount, log amount, and cyclic encodings "
            "of hour and day. The past-only behavioral feature set adds customer and terminal "
            "histories computed only from earlier transactions, such as prior counts, prior "
            "amount summaries, recent activity windows, time since previous transaction, and "
            "customer-terminal interaction counts.",
            styles["Body"],
        )
    )
    story.append(
        paragraph(
            "The prior fraud-label history feature set adds previous fraud counts and fraud rates "
            "for customers and terminals. These features give much larger gains, but they require "
            "timely known prior labels. If labels arrive after review, disputes, or chargeback "
            "windows, those features would need explicit lagging or exclusion.",
            styles["Body"],
        )
    )
    story.append(
        paragraph(
            "Leakage controls include sorting by transaction time and transaction id, computing "
            "histories from past transactions only, excluding TX_FRAUD and TX_FRAUD_SCENARIO as "
            "predictors, and avoiding full-window customer or terminal aggregates.",
            styles["Body"],
        )
    )

    add_section(story, "4. Reproducibility Analysis", styles)
    story.append(
        paragraph(
            "The source code, notebooks, and simulated dataset instructions are public. Raw data "
            "are obtained separately from the Fraud Detection Handbook dataset repository and are "
            "not stored in this submission repository. The notebook and source files reproduce "
            "data loading, exploratory analysis, baseline modeling, behavioral features, and "
            "saved metric tables without requiring fitted model binaries.",
            styles["Body"],
        )
    )
    story.append(
        bullet_list(
            [
                "Dataset source: https://github.com/Fraud-Detection-Handbook/simulated-data-raw",
                "Original handbook: https://fraud-detection-handbook.github.io/fraud-detection-handbook/",
                "Original code repository: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook",
                "Submission branch: " + GITHUB_BRANCH_URL,
            ],
            styles["Body"],
        )
    )
    story.append(
        paragraph(
            "The main remaining reproducibility limitation is external validity: the data are "
            "synthetic and generated from known fraud scenarios, so high performance may partly "
            "reflect recovery of simulator structure rather than robust real-world fraud behavior.",
            styles["Body"],
        )
    )

    story.append(PageBreak())
    add_section(story, "5. Experimental Results", styles)
    story.append(
        paragraph(
            "The final test period contains 345,144 transactions and 3,091 fraud cases, for "
            "0.896% fraud prevalence. Average Precision / PR-AUC is emphasized because this is "
            "an alert-ranking problem under extreme imbalance. ROC-AUC, precision, recall, F1, "
            "confusion matrix counts, and Precision@k are also reported in the project reports.",
            styles["Body"],
        )
    )
    story.append(
        markdown_table(
            main_result_rows,
            styles["Small"],
            [2.65 * inch, 1.75 * inch, 0.82 * inch, 0.78 * inch, 0.78 * inch, 0.72 * inch, 0.60 * inch],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        paragraph(
            "Past-only behavioral features improve performance modestly. Prior fraud-label "
            "history gives much larger gains, but that result depends on the operational "
            "assumption that earlier fraud labels are known before later transactions are scored.",
            styles["Body"],
        )
    )
    story.append(
        paragraph(
            "MCC is mentioned in the assignment's recommended metrics but was omitted from the "
            "implemented evaluation. The project prioritized AP / PR-AUC, ROC-AUC, precision, "
            "recall, F1, confusion matrix counts, and Precision@k because the task is alert "
            "ranking under extreme imbalance. MCC can be added as a minor extension.",
            styles["Body"],
        )
    )

    add_section(story, "6. Conclusions", styles)
    story.append(
        paragraph(
            "The project supports three limited conclusions. First, chronological validation is "
            "necessary for fraud data with time-dependent behavior. Second, precision-recall "
            "metrics are more informative than accuracy when fraud prevalence is below 1%. Third, "
            "past behavioral histories are useful, but the interpretation changes sharply when "
            "features depend on prior known labels.",
            styles["Body"],
        )
    )
    story.append(
        paragraph(
            "The project does not prove real-world deployability. Delayed labels, calibration, "
            "drift, alert-budgeting, review cost, privacy controls, and cost-sensitive thresholds "
            "remain future work.",
            styles["Body"],
        )
    )

    add_section(story, "7. Executive Summary", styles)
    story.append(
        bullet_list(
            [
                "Best conservative result: past-only behavioral features modestly improve the baseline.",
                "Best numerical result: prior fraud-label history reaches much higher AP / PR-AUC but requires timely known labels.",
                "Main methodological contribution: leakage-aware temporal evaluation with fraud-appropriate metrics.",
                "Main limitation: synthetic data may make the feature-label relationships cleaner than real payment data.",
            ],
            styles["Body"],
        )
    )

    story.append(PageBreak())
    add_section(story, "8. Summing It Up", styles)
    story.append(
        paragraph(
            "This submission is best understood as a reproducible and critical benchmark study. "
            "It shows why temporal validation, precision-recall evaluation, and past-only feature "
            "construction matter in fraud detection. The strongest model results are internally "
            "impressive but externally conditional, especially when they depend on prior fraud "
            "labels that may not be available at authorization time.",
            styles["Body"],
        )
    )

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


if __name__ == "__main__":
    build_pdf()
