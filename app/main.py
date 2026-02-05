from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

# Persisted file next to this script (NOTE: on Render free tier, disk can reset on redeploy)
DB_PATH = os.path.join(os.path.dirname(__file__), "judging.sqlite")

app = FastAPI()


# -----------------------
# DB helpers
# -----------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                admin_pw_hash TEXT NOT NULL,
                join_code TEXT NOT NULL,
                locked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS competitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                bib TEXT NOT NULL,
                UNIQUE(event_id, bib)
            );

            CREATE TABLE IF NOT EXISTS judges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                judge_name TEXT NOT NULL,
                judge_token TEXT NOT NULL,
                last_submit_at TEXT,
                UNIQUE(event_id, judge_name)
            );

            -- marks stores raw score, but admin/results will only show derived placements + totals
            CREATE TABLE IF NOT EXISTS marks (
                event_id INTEGER NOT NULL,
                judge_id INTEGER NOT NULL,
                bib TEXT NOT NULL,
                score INTEGER,
                PRIMARY KEY (event_id, judge_id, bib)
            );
            """
        )

        # Simple migration safety if an older table exists
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(marks)").fetchall()]
        if "score" not in cols:
            conn.execute("ALTER TABLE marks ADD COLUMN score INTEGER")


@app.on_event("startup")
def _startup():
    init_db()


def require_admin(event_id: int, admin_password: str) -> None:
    with db() as conn:
        row = conn.execute("SELECT admin_pw_hash FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Event not found.")
        if sha256(admin_password) != row["admin_pw_hash"]:
            raise HTTPException(status_code=403, detail="Invalid admin password.")


def require_judge(event_id: int, judge_id: int, token: str) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute(
            "SELECT id, judge_name FROM judges WHERE id=? AND event_id=? AND judge_token=?",
            (judge_id, event_id, token),
        ).fetchone()
    if not row:
        raise HTTPException(403, "Invalid judge session.")
    return row


# -----------------------
# Scoring -> placements -> final rank
# -----------------------
def scores_to_placements(scores_by_bib: Dict[str, int]) -> Dict[str, int]:
    """
    Convert one judge's unique scores (0-100) into placements:
    highest score => placement 1, lowest => placement N.
    Assumes scores are unique (enforced elsewhere).
    """
    ordered = sorted(scores_by_bib.items(), key=lambda kv: (-kv[1], str(kv[0])))
    placements: Dict[str, int] = {}
    for idx, (bib, _score) in enumerate(ordered, start=1):
        placements[str(bib)] = idx
    return placements


def compute_final_results(judge_scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    judge_scores:
      rows = judge_name
      cols = bib
      values = score (0-100)

    Returns:
      results_df columns: FinalRank, Bib, TotalScore, SumPlacements
      placements_matrix rows=judge_name cols=bib values=placement
    """
    if judge_scores.shape[0] == 0 or judge_scores.shape[1] == 0:
        raise ValueError("No judge scores available.")

    # Ensure numeric
    scores = judge_scores.apply(pd.to_numeric, errors="coerce")

    # Create placement matrix per judge
    placement_rows = {}
    for judge_name in scores.index:
        row = scores.loc[judge_name].to_dict()
        # drop missing
        row_clean = {str(b): int(v) for b, v in row.items() if pd.notna(v)}
        if len(row_clean) != len(scores.columns):
            raise ValueError(f"Judge '{judge_name}' is missing scores for one or more bibs.")
        if len(set(row_clean.values())) != len(row_clean.values()):
            raise ValueError(f"Judge '{judge_name}' has duplicate scores (not allowed).")
        placement_rows[judge_name] = scores_to_placements(row_clean)

    placements = pd.DataFrame.from_dict(placement_rows, orient="index")[scores.columns]

    # Final ranking based on total score across judges
    total_score = scores.sum(axis=0, skipna=False)  # should be complete
    sum_placements = placements.sum(axis=0)

    results = pd.DataFrame(
        {
            "Bib": scores.columns.astype(str),
            "TotalScore": [float(total_score[b]) for b in scores.columns],
            "SumPlacements": [int(sum_placements[b]) for b in scores.columns],
        }
    )

    # Sort: higher total score wins; tie-breaker: lower sum placements; then bib
    results = results.sort_values(
        by=["TotalScore", "SumPlacements", "Bib"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    results.insert(0, "FinalRank", range(1, len(results) + 1))
    return results, placements


# -----------------------
# UI helpers
# -----------------------
def page(title: str, body: str) -> HTMLResponse:
    html = f"""
    <html>
      <head>
        <title>{title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body {{ font-family: system-ui, Arial; max-width: 980px; margin: 0 auto; padding: 22px; }}
          input, textarea, button {{ font-size: 16px; padding: 10px; }}
          textarea {{ width: 100%; }}
          .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 16px 0; }}
          .row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
          .row > * {{ flex: 1; min-width: 220px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; }}
          th {{ text-align: left; background: #f7f7f7; }}
          .muted {{ color: #666; }}
          .pill {{ display:inline-block; padding:4px 10px; border:1px solid #ddd; border-radius:999px; }}
          a {{ text-decoration: none; }}
          .danger {{ color: #b00020; }}
          .ok {{ color: #2e7d32; }}
          .score-pill {{ display:inline-block; min-width: 44px; text-align:center; border:1px solid #ddd; border-radius:999px; padding:2px 8px; margin-left: 10px; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        {body}
      </body>
    </html>
    """
    return HTMLResponse(html)


# -----------------------
# Routes: Home
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return page(
        "Judging App",
        """
        <div class="card">
          <p><a href="/admin">Admin</a> | <a href="/judge">Judge</a></p>
          <p class="muted">
            Finals-only scoring MVP: Judges score each competitor 0–100 with sliders (no duplicates).
            System converts scores to placements and computes final rank by total score.
          </p>
        </div>
        """,
    )


# -----------------------
# Routes: Admin
# -----------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_home():
    with db() as conn:
        events = conn.execute("SELECT * FROM events ORDER BY id DESC").fetchall()

    rows = ""
    for e in events:
        rows += f"""
        <tr>
          <td>{e["id"]}</td>
          <td>{e["name"]}</td>
          <td><span class="pill">{e["join_code"]}</span></td>
          <td>{'🔒' if e["locked"] else '🟢'}</td>
          <td><a href="/admin/event/{e['id']}">Open</a></td>
        </tr>
        """

    body = f"""
    <div class="card">
      <h2>Create Event</h2>
      <form method="post" action="/admin/create">
        <div class="row">
          <input name="event_name" placeholder="Event name (e.g., Jack & Jill Finals)" required />
          <input name="admin_password" placeholder="Admin password" type="password" required />
        </div>
        <button type="submit">Create</button>
      </form>
      <p class="muted">Save the admin password. You’ll need it to edit bibs, lock, and compute results.</p>
    </div>

    <div class="card">
      <h2>Existing Events</h2>
      <table>
        <thead><tr><th>ID</th><th>Name</th><th>Join Code</th><th>Status</th><th></th></tr></thead>
        <tbody>{rows or '<tr><td colspan="5" class="muted">No events yet.</td></tr>'}</tbody>
      </table>
    </div>
    """
    return page("Admin", body)


@app.post("/admin/create")
def admin_create(event_name: str = Form(...), admin_password: str = Form(...)):
    join_code = secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:8].upper()
    with db() as conn:
        conn.execute(
            "INSERT INTO events(name, admin_pw_hash, join_code, locked, created_at) VALUES(?,?,?,?,?)",
            (event_name.strip(), sha256(admin_password), join_code, 0, datetime.utcnow().isoformat()),
        )
        event_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return RedirectResponse(url=f"/admin/event/{event_id}", status_code=303)


@app.get("/admin/event/{event_id}", response_class=HTMLResponse)
def admin_event(event_id: int):
    with db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        bibs = [r["bib"] for r in conn.execute(
            "SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)
        ).fetchall()]

        judges = conn.execute(
            "SELECT id, judge_name, last_submit_at FROM judges WHERE event_id=? ORDER BY judge_name",
            (event_id,),
        ).fetchall()

    bib_list = ", ".join(bibs) if bibs else "(none yet)"

    judge_rows = ""
    for j in judges:
        judge_rows += f"<tr><td>{j['judge_name']}</td><td>{j['last_submit_at'] or ''}</td></tr>"
    if not judge_rows:
        judge_rows = '<tr><td colspan="2" class="muted">No judges yet.</td></tr>'

    body = f"""
    <div class="card">
      <p><a href="/admin">← Back to Admin</a></p>
      <h2>{event["name"]}</h2>
      <p>Join Code: <span class="pill">{event["join_code"]}</span> (Judges go to <a href="/judge">/judge</a>)</p>
      <p>Status: {'🔒 Locked' if event["locked"] else '🟢 Open'}</p>
      <p class="muted">Bibs: {bib_list}</p>
    </div>

    <div class="card">
      <h3>Add / Replace Bibs</h3>
      <form method="post" action="/admin/event/{event_id}/set_bibs">
        <div class="row">
          <input name="admin_password" placeholder="Admin password" type="password" required />
        </div>
        <textarea name="bibs" rows="5" placeholder="One bib per line (or separated by commas)"></textarea>
        <p class="muted">This replaces the bib list.</p>
        <button type="submit">Save Bibs</button>
      </form>
    </div>

    <div class="card">
      <h3>Judges</h3>
      <table>
        <thead><tr><th>Judge</th><th>Last Submit</th></tr></thead>
        <tbody>{judge_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Controls</h3>
      <form method="post" action="/admin/event/{event_id}/toggle_lock" style="margin-bottom:12px;">
        <div class="row">
          <input name="admin_password" placeholder="Admin password" type="password" required />
        </div>
        <button type="submit">{'Unlock' if event["locked"] else 'Lock'} Event</button>
      </form>

      <form method="post" action="/admin/event/{event_id}/compute">
        <div class="row">
          <input name="admin_password" placeholder="Admin password" type="password" required />
        </div>
        <button type="submit">Compute Results</button>
      </form>
      <p class="muted">Admin will only see placements + computed final rank (no raw scores shown).</p>
    </div>
    """
    return page(f"Admin Event #{event_id}", body)


@app.post("/admin/event/{event_id}/set_bibs")
def admin_set_bibs(event_id: int, admin_password: str = Form(...), bibs: str = Form("")):
    require_admin(event_id, admin_password)

    raw = bibs.replace(",", "\n")
    bib_list = [b.strip() for b in raw.splitlines() if b.strip()]
    bib_list = list(dict.fromkeys(bib_list))  # dedupe preserve order

    with db() as conn:
        event = conn.execute("SELECT id FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        conn.execute("DELETE FROM competitors WHERE event_id=?", (event_id,))
        for b in bib_list:
            conn.execute("INSERT OR IGNORE INTO competitors(event_id, bib) VALUES(?,?)", (event_id, str(b)))

        # Remove marks for bibs that no longer exist
        if bib_list:
            placeholders = ",".join(["?"] * len(bib_list))
            conn.execute(
                f"DELETE FROM marks WHERE event_id=? AND bib NOT IN ({placeholders})",
                (event_id, *bib_list),
            )
        else:
            conn.execute("DELETE FROM marks WHERE event_id=?", (event_id,))

    return RedirectResponse(url=f"/admin/event/{event_id}", status_code=303)


@app.post("/admin/event/{event_id}/toggle_lock")
def admin_toggle_lock(event_id: int, admin_password: str = Form(...)):
    require_admin(event_id, admin_password)
    with db() as conn:
        row = conn.execute("SELECT locked FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Event not found.")
        new_val = 0 if row["locked"] else 1
        conn.execute("UPDATE events SET locked=? WHERE id=?", (new_val, event_id))
    return RedirectResponse(url=f"/admin/event/{event_id}", status_code=303)


def load_scores_dataframe(event_id: int) -> pd.DataFrame:
    """Return judge_scores DataFrame rows=judge_name cols=bib values=score."""
    with db() as conn:
        bibs = [r["bib"] for r in conn.execute(
            "SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)
        ).fetchall()]

        judges = conn.execute(
            "SELECT id, judge_name FROM judges WHERE event_id=? ORDER BY judge_name",
            (event_id,),
        ).fetchall()

        if not bibs:
            raise ValueError("No bibs found.")
        if not judges:
            raise ValueError("No judges found.")

        data = {}
        for j in judges:
            row = {}
            for b in bibs:
                m = conn.execute(
                    "SELECT score FROM marks WHERE event_id=? AND judge_id=? AND bib=?",
                    (event_id, j["id"], b),
                ).fetchone()
                row[str(b)] = None if (m is None or m["score"] is None) else int(m["score"])
            data[j["judge_name"]] = row

    df = pd.DataFrame.from_dict(data, orient="index")
    # Ensure consistent column order
    df = df[[str(b) for b in bibs]]
    return df


@app.post("/admin/event/{event_id}/compute", response_class=HTMLResponse)
def admin_compute(event_id: int, admin_password: str = Form(...)):
    require_admin(event_id, admin_password)

    with db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

    try:
        judge_scores = load_scores_dataframe(event_id)
        results_df, placements_df = compute_final_results(judge_scores)
    except Exception as e:
        return page(
            "Compute Results",
            f'<div class="card"><p class="danger">Error computing results:</p><pre>{e}</pre>'
            f'<p class="muted">Common causes: missing scores for a judge, duplicates, or no bibs.</p></div>',
        )

    # Results table
    result_rows = ""
    for _, r in results_df.iterrows():
        result_rows += (
            f"<tr><td>{int(r['FinalRank'])}</td><td>{r['Bib']}</td>"
            f"<td>{int(r['TotalScore'])}</td><td>{int(r['SumPlacements'])}</td></tr>"
        )

    # Placement matrix table (admin sees placements only)
    mat_html = placements_df.copy()
    mat_html.insert(0, "Judge", mat_html.index)
    matrix_rows = ""
    for _, row in mat_html.iterrows():
        tds = "".join([f"<td>{row[c]}</td>" for c in mat_html.columns])
        matrix_rows += f"<tr>{tds}</tr>"

    matrix_head = "".join([f"<th>{c}</th>" for c in mat_html.columns])

    body = f"""
    <div class="card">
      <p><a href="/admin/event/{event_id}">← Back to Event</a></p>
      <h2>Results</h2>
      <p class="muted">FinalRank is computed by TotalScore (highest wins). Tie-breakers: lower SumPlacements, then Bib.</p>
      <p>
        <a href="/admin/event/{event_id}/download/results?admin_password={admin_password}">Download Results CSV</a>
        &nbsp;|&nbsp;
        <a href="/admin/event/{event_id}/download/placements?admin_password={admin_password}">Download Judge Placements CSV</a>
      </p>
      <table>
        <thead><tr><th>FinalRank</th><th>Bib</th><th>TotalScore</th><th>SumPlacements</th></tr></thead>
        <tbody>{result_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Judge Placements (scores hidden)</h3>
      <table>
        <thead><tr>{matrix_head}</tr></thead>
        <tbody>{matrix_rows}</tbody>
      </table>
    </div>
    """
    return page("Results", body)


@app.get("/admin/event/{event_id}/download/results")
def download_results(event_id: int, admin_password: str):
    require_admin(event_id, admin_password)
    judge_scores = load_scores_dataframe(event_id)
    results_df, _placements_df = compute_final_results(judge_scores)

    buf = StringIO()
    results_df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="event_{event_id}_results.csv"'},
    )


@app.get("/admin/event/{event_id}/download/placements")
def download_placements(event_id: int, admin_password: str):
    require_admin(event_id, admin_password)
    judge_scores = load_scores_dataframe(event_id)
    _results_df, placements_df = compute_final_results(judge_scores)

    out = placements_df.copy()
    out.insert(0, "Judge", out.index)
    buf = StringIO()
    out.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="event_{event_id}_judge_placements.csv"'},
    )


# -----------------------
# Routes: Judge
# -----------------------
@app.get("/judge", response_class=HTMLResponse)
def judge_home():
    body = """
    <div class="card">
      <form method="post" action="/judge/join">
        <div class="row">
          <input name="judge_name" placeholder="Your name" required />
          <input name="join_code" placeholder="Join code" required />
        </div>
        <button type="submit">Join Event</button>
      </form>
      <p class="muted">Ask the admin for the join code.</p>
    </div>
    """
    return page("Judge", body)


@app.post("/judge/join")
def judge_join(judge_name: str = Form(...), join_code: str = Form(...)):
    judge_name = judge_name.strip()
    join_code = join_code.strip().upper()

    with db() as conn:
        event = conn.execute("SELECT id, locked FROM events WHERE join_code=?", (join_code,)).fetchone()
        if not event:
            return page("Judge", '<div class="card"><p class="danger">Invalid join code.</p></div>')

        if event["locked"]:
            return page("Judge", '<div class="card"><p class="danger">This event is locked. No submissions allowed.</p></div>')

        token = secrets.token_urlsafe(16)

        existing = conn.execute(
            "SELECT id FROM judges WHERE event_id=? AND judge_name=?",
            (event["id"], judge_name),
        ).fetchone()

        if existing:
            judge_id = existing["id"]
            conn.execute("UPDATE judges SET judge_token=? WHERE id=?", (token, judge_id))
        else:
            conn.execute(
                "INSERT INTO judges(event_id, judge_name, judge_token) VALUES(?,?,?)",
                (event["id"], judge_name, token),
            )
            judge_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    return RedirectResponse(url=f"/judge/event/{event['id']}?judge_id={judge_id}&token={token}", status_code=303)


@app.get("/judge/event/{event_id}", response_class=HTMLResponse)
def judge_event(event_id: int, judge_id: int, token: str):
    judge = require_judge(event_id, judge_id, token)

    with db() as conn:
        event = conn.execute("SELECT name, locked FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        if event["locked"]:
            return page("Judge", '<div class="card"><p class="danger">This event is locked. Submissions are closed.</p></div>')

        bibs = [r["bib"] for r in conn.execute(
            "SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)
        ).fetchall()]

        if not bibs:
            return page("Judge", '<div class="card">No bibs posted yet. Check back soon.</div>')

        existing_scores: Dict[str, int] = {}
        for b in bibs:
            m = conn.execute(
                "SELECT score FROM marks WHERE event_id=? AND judge_id=? AND bib=?",
                (event_id, judge_id, b),
            ).fetchone()
            existing_scores[str(b)] = 0 if (m is None or m["score"] is None) else int(m["score"])

    rows = ""
    for b in bibs:
        val = int(existing_scores.get(str(b), 0))
        rows += f"""
        <tr data-bib="{b}">
          <td>{b}</td>
          <td style="min-width:280px;">
            <input type="range" min="0" max="100" step="1" name="s__{b}" value="{val}" oninput="syncVal(this)">
            <span class="score-pill">{val}</span>
          </td>
        </tr>
        """

    body = f"""
    <div class="card">
      <p><a href="/judge">← Back</a></p>
      <h2>{event["name"]}</h2>
      <p>Judge: <b>{judge["judge_name"]}</b></p>
      <p class="muted">Assign a unique score (0–100) to every competitor. Duplicates are blocked.</p>

      <div id="dupMsg" class="muted" style="margin-bottom:10px;"></div>

      <form method="post" action="/judge/event/{event_id}/submit" onsubmit="return validateBeforeSubmit();">
        <input type="hidden" name="judge_id" value="{judge_id}" />
        <input type="hidden" name="token" value="{token}" />

        <table id="scoreTable">
          <thead><tr><th>Bib</th><th>Score</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>

        <button id="submitBtn" type="submit" style="margin-top:12px;">Submit</button>
      </form>
    </div>

    <script>
      function syncVal(slider) {{
        const pill = slider.parentElement.querySelector('.score-pill');
        pill.textContent = slider.value;
        validateDuplicates();
      }}

      function validateDuplicates() {{
        const rows = document.querySelectorAll('#scoreTable tbody tr');
        const seen = new Map();
        let hasDup = false;

        rows.forEach(r => r.style.background = '');
        rows.forEach(r => {{
          const slider = r.querySelector('input[type="range"]');
          const v = slider.value;
          if (seen.has(v)) {{
            hasDup = true;
            r.style.background = '#ffe5e5';
            seen.get(v).style.background = '#ffe5e5';
          }} else {{
            seen.set(v, r);
          }}
        }});

        const msg = document.getElementById('dupMsg');
        const btn = document.getElementById('submitBtn');

        if (hasDup) {{
          msg.textContent = "Duplicate score detected. Adjust sliders so every score is unique.";
          msg.className = "danger";
          btn.disabled = true;
        }} else {{
          msg.textContent = "All scores unique ✅";
          msg.className = "ok";
          btn.disabled = false;
        }}
        return !hasDup;
      }}

      function validateBeforeSubmit() {{
        return validateDuplicates();
      }}

      validateDuplicates();
    </script>
    """
    return page("Judge Scoring", body)


@app.post("/judge/event/{event_id}/submit")
async def judge_submit(event_id: int, request: Request, judge_id: int = Form(...), token: str = Form(...)):
    _judge = require_judge(event_id, judge_id, token)
    form = await request.form()

    with db() as conn:
        event = conn.execute("SELECT locked FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")
        if event["locked"]:
            return page("Judge", '<div class="card"><p class="danger">This event is locked. Submission rejected.</p></div>')

        bibs = [r["bib"] for r in conn.execute(
            "SELECT bib FROM competitors WHERE event_id=?", (event_id,)
        ).fetchall()]

        if not bibs:
            return page("Judge", '<div class="card">No bibs posted yet.</div>')

        scores: Dict[str, int] = {}
        for b in bibs:
            key = f"s__{b}"
            if key not in form:
                return page("Judge Scoring", f'<div class="card"><p class="danger">Missing score for bib {b}.</p></div>')
            try:
                v = int(str(form[key]).strip())
            except ValueError:
                return page("Judge Scoring", f'<div class="card"><p class="danger">Invalid score for bib {b}.</p></div>')
            if v < 0 or v > 100:
                return page("Judge Scoring", f'<div class="card"><p class="danger">Score out of range for bib {b}: {v}.</p></div>')
            scores[str(b)] = v

        # Server-side uniqueness enforcement
        if len(set(scores.values())) != len(scores.values()):
            return page("Judge Scoring", '<div class="card"><p class="danger">Duplicate scores detected. All scores must be unique.</p></div>')

        # Save scores
        for bib, score in scores.items():
            conn.execute(
                """
                INSERT INTO marks(event_id, judge_id, bib, score)
                VALUES(?,?,?,?)
                ON CONFLICT(event_id, judge_id, bib) DO UPDATE SET score=excluded.score
                """,
                (event_id, judge_id, bib, score),
            )

        conn.execute(
            "UPDATE judges SET last_submit_at=? WHERE id=?",
            (datetime.utcnow().isoformat(timespec="seconds"), judge_id),
        )

    return RedirectResponse(url=f"/judge/event/{event_id}?judge_id={judge_id}&token={token}", status_code=303)
