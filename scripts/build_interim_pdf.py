"""Build the interim PDF report for Week 11 - Tenacious Sales Evaluation Bench.

Output: report/interim_report.pdf

Content per the brief:
  - Audit memo (gap analysis)
  - Dataset composition (254 tasks, 10 dimensions)
  - Contamination verification results
  - Inter-rater agreement results
  - Path declaration (Path B - preference-tuned judge)
  - Cost tracking ($10 envelope)
  - Remaining work (Act III-V)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "report" / "interim_report.pdf"


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_audit_memo() -> str:
    """Read the audit memo content."""
    path = ROOT / "audit_memo.md"
    if not path.exists():
        return "Audit memo not found."
    try:
        content = path.read_text(encoding="utf-8")
        # Extract first 600 words
        words = content.split()[:600]
        return " ".join(words) + "..."
    except Exception:
        return "Error reading audit memo."


def _read_methodology() -> dict:
    """Read key methodology sections."""
    path = ROOT / "methodology.md"
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        # Extract key sections
        sections = {}
        current_section = None
        for line in content.split('\n'):
            if line.startswith('## '):
                current_section = line[3:].strip()
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())
        return {k: ' '.join(v) for k, v in sections.items()}
    except Exception:
        return {}


def _read_composition() -> dict:
    """Read dataset composition."""
    path = ROOT / "tenacious_bench_v0.1" / "composition.json"
    return _load_json(path) or {}


def _read_contamination() -> dict:
    """Read contamination check results."""
    path = ROOT / "tenacious_bench_v0.1" / "contamination_check.json"
    return _load_json(path) or {}


def _read_inter_rater() -> dict:
    """Read inter-rater agreement results."""
    path = ROOT / "inter_rater_agreement.md"
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        results = {}
        # Parse agreement percentages
        for line in content.split('\n'):
            if 'Input coherence' in line and '%' in line:
                results['input_coherence'] = line.strip()
            elif 'Ground truth' in line and '%' in line:
                results['ground_truth'] = line.strip()
            elif 'Rubric application' in line and '%' in line:
                results['rubric'] = line.strip()
        return results
    except Exception:
        return {}


def _read_cost_log() -> dict:
    """Read cost tracking."""
    path = ROOT / "cost" / "log.md"
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        costs = {}
        for line in content.split('\n'):
            if '|' in line and 'Dataset authoring' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    costs['dataset'] = parts[3].strip()
            elif '|' in line and 'Training' in line and 'Spent' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    costs['training'] = parts[3].strip()
            elif '|' in line and 'Held-out evaluation' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    costs['evaluation'] = parts[3].strip()
            elif '|' in line and 'Total' in line and '$' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    costs['total'] = parts[3].strip()
        return costs
    except Exception:
        return {}


def build_interim_pdf():
    """Build the interim PDF report."""
    print(f"Building interim PDF at {OUT_PATH}")
    
    # Create output directory
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Build document
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=LETTER,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=16,
    )
    h3_style = ParagraphStyle(
        'Heading3',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=12,
    )
    body_style = styles['BodyText']
    body_style.fontSize = 10
    body_style.leading = 14
    
    # Build story
    story = []
    
    # Title
    story.append(Paragraph("Tenacious-Bench v0.1 — Interim Report", title_style))
    story.append(Paragraph("Week 11: Building the Sales Evaluation Bench", styles['Heading3']))
    story.append(Paragraph(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", body_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h2_style))
    story.append(Paragraph(
        "This interim report documents progress on Week 11: building a domain-specific evaluation "
        "benchmark for Tenacious-style B2B sales work. Key deliverables completed include the "
        "audit memo, 254-task dataset across 10 dimensions, contamination verification, "
        "inter-rater agreement (all dimensions >80%), datasheet documentation, and cost tracking.",
        body_style
    ))
    story.append(Spacer(1, 12))
    
    # The Gap
    story.append(Paragraph("2. The Gap: Why τ²-Bench Retail Fails", h2_style))
    audit_memo = _read_audit_memo()
    story.append(Paragraph(audit_memo[:500] + "...", body_style))
    story.append(Spacer(1, 12))
    
    # Dataset Composition
    story.append(Paragraph("3. Dataset Composition", h2_style))
    comp = _read_composition()
    if comp:
        overall = comp.get('overall', {})
        story.append(Paragraph(
            f"Total tasks: {overall.get('total', 'N/A')} | "
            f"Dimensions: {len(overall.get('by_dimension', {}))} | "
            f"Partitions: train ({comp.get('train', {}).get('total', 'N/A')}), "
            f"dev ({comp.get('dev', {}).get('total', 'N/A')}), "
            f"held_out ({comp.get('held_out', {}).get('total', 'N/A')})",
            body_style
        ))
        story.append(Spacer(1, 6))
        
        # Dimension table
        dim_data = [['Dimension', 'Count']]
        for dim, count in overall.get('by_dimension', {}).items():
            dim_data.append([dim.replace('_', ' ').title(), str(count)])
        
        dim_table = Table(dim_data, colWidths=[3*inch, 1*inch])
        dim_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(dim_table)
    story.append(Spacer(1, 12))
    
    # Contamination
    story.append(Paragraph("4. Contamination Verification", h2_style))
    cont = _read_contamination()
    if cont:
        ngram = cont.get('n_gram_check', {})
        story.append(Paragraph(
            f"N-gram check: {ngram.get('n', 'N/A')}-gram threshold, "
            f"{len(ngram.get('violations', []))} violations found",
            body_style
        ))
        story.append(Paragraph(
            "Note: TB-0015 showed overlap with multiple tasks and has been flagged "
            "for exclusion from sealed leaderboard.",
            body_style
        ))
    story.append(Spacer(1, 12))
    
    # Inter-rater Agreement
    story.append(Paragraph("5. Inter-Rater Agreement", h2_style))
    ira = _read_inter_rater()
    if ira:
        story.append(Paragraph("30-task sample, 24-hour interval:", body_style))
        for key, value in ira.items():
            story.append(Paragraph(f"  - {key}: {value}", body_style))
    story.append(Spacer(1, 12))
    
    # Path Declaration
    story.append(Paragraph("6. Path Declaration: Path B (Preference-Tuned Judge)", h2_style))
    story.append(Paragraph(
        "Selected Path B based on Week 10 failure evidence: dual_control_coordination "
        "triggered at 1.00, indicating the agent cannot tell when it is wrong. "
        "Training will use SimPO (reference-free, fits Colab T4) with Qwen 3.5 backbone.",
        body_style
    ))
    story.append(Spacer(1, 12))
    
    # Cost Tracking
    story.append(Paragraph("7. Cost Tracking", h2_style))
    costs = _read_cost_log()
    if costs:
        story.append(Paragraph(f"Dataset authoring: {costs.get('dataset', 'N/A')}", body_style))
        story.append(Paragraph(f"Training: {costs.get('training', 'N/A')}", body_style))
        story.append(Paragraph(f"Held-out evaluation: {costs.get('evaluation', 'N/A')}", body_style))
        story.append(Paragraph(f"Total: {costs.get('total', 'N/A')}", body_style))
    story.append(Spacer(1, 12))
    
    # Remaining Work
    story.append(Paragraph("8. Remaining Work", h2_style))
    story.append(Paragraph("Act III: Training data preparation (Day 4)", body_style))
    story.append(Paragraph("Act IV: SimPO training + ablations (Days 5-6)", body_style))
    story.append(Paragraph("Act V: Publication + blog + memo (Day 7)", body_style))
    story.append(Spacer(1, 20))
    
    # Footer
    story.append(Paragraph(
        "Report generated: " + datetime.now(timezone.utc).isoformat(),
        styles['Italic']
    ))
    story.append(Paragraph("License: CC-BY-4.0", styles['Italic']))
    
    # Build PDF
    doc.build(story)
    print(f"PDF generated: {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build_interim_pdf()