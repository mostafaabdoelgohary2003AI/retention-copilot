from docx import Document

def build(path, summary, top):
    d = Document()
    d.add_heading("Customer Retention Report", 0)
    d.add_paragraph(f"{summary['total']} customers reviewed. High risk: {summary['high']}, medium: {summary['medium']}, low: {summary['low']}.")
    d.add_heading("Why customers may leave", 1)
    for r, c in summary["top_reasons"]:
        d.add_paragraph(f"{r} ({c} customers)", style="List Bullet")
    d.add_heading("Top customers to contact first", 1)
    t = d.add_table(rows=1, cols=4); t.style = "Light Grid Accent 1"
    for i, h in enumerate(["Customer", "Risk", "Why", "Suggested action"]):
        t.rows[0].cells[i].text = h
    for r in top:
        row = t.add_row().cells
        row[0].text = str(r["id"]); row[1].text = f"{r['risk']:.0%}"
        row[2].text = "; ".join(r["reasons"]); row[3].text = " ".join(r["actions"])
    d.save(path)
