# pdf_generator.py

from typing import List, Tuple
from datetime import datetime
from llmconfig import generate_pdf_and_upload_to_blob
def build_team_timesheet_pdf(
    rows: List[Tuple],
    start: str,
    end: str,
    status_filter: str
) -> str:
    """
    Generate and upload PDF for team timesheets.
    Returns the blob link.
    """
    pdf_html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        table {{
        width: 100%;
        border-collapse: collapse;
        }}
        th, td {{
        border: 1px solid black;
        padding: 8px;
        text-align: left;
        }}
        th {{
        background-color: #f2f2f2;
        }}
    </style>
    </head>
    <body>
    <h2>Team Timesheet Report - {status_filter}</h2>
    <p><strong>From:</strong> {start} &nbsp;&nbsp; <strong>To:</strong> {end}</p>
    <table>
        <thead>
        <tr>
            <th>Name</th><th>Project</th><th>Clock-in</th><th>Clock-out</th><th>Duration</th><th>Status</th>
        </tr>
        </thead>
        <tbody>
    """

    for eid, emp, ename, pid, pname, in_dt, out_dt, dur, st in rows:
        pdf_html += (
            "<tr>"
            f"<td>{ename}</td>"
            f"<td>{pname or '—'}</td>"
            f"<td>{in_dt.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td>{out_dt.strftime('%Y-%m-%d %H:%M') if out_dt else '—'}</td>"
            f"<td>{dur or '—'} min</td>"
            f"<td>{st}</td>"
            "</tr>"
        )

    pdf_html += "</tbody></table></body></html>"

    return generate_pdf_and_upload_to_blob(pdf_html, "team_timesheet")
