from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response

DB_PATH = os.path.join(os.path.dirname(__file__), "judging.sqlite")

app = FastAPI()


# -----------------------
# Database helpers
# -----------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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

            CREATE TABLE IF NOT EXISTS marks (
                event_id INTEGER NOT NULL,
                judge_id INTEGER NOT NULL,
                bib TEXT NOT NULL,
                placement INTEGER,
                PRIMARY KEY (event_id, judge_id, bib)
            );
            """
        )


@app.on_event("startup")
def _startup():
    init_db()


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def require_admin(event_id: int, admin_password: str) -> None:
    with db() as conn:
        row = conn.execute("SELECT admin_pw_hash FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Event not found.")
        if sha256(admin_password) != row["admin_pw_hash"]:
            raise HTTPException(status_code=403, detail="Invalid admin password.")


# -----------------------
# Skating system (majority-based)
# -----------------------
def skating_rank(judge_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    judge_matrix:
      index = judge_name
      columns = bib
      values = placements (1=best)

    Returns DataFrame: FinalRank, Bib, TotalMarkSum
    """
    marks = judge_matrix.apply(pd.to_numeric, errors="coerce")
    marks = marks.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if marks.shape[0] == 0 or marks.shape[1] == 0:
        raise ValueError("No usable judge/competitor data after cleaning.")

    num_judges = marks.shape[0]
    majority_needed = num_judges // 2 + 1
    max_place = int(np.nanmax(marks.values))

    remaining_bibs = list(marks.columns)
    total_sum = marks.sum(axis=0, skipna=True)
    final_order: list[str] = []

    for _rank in range(1, len(remaining_bibs) + 1):
        if len(remaining_bibs) == 1:
            final_order.append(remaining_bibs[0])
            break

        tied = remaining_bibs.copy()
        picked = None

        for k in range(1, max_place + 1):
            stats = []
            for bib in tied:
                bib_marks = marks[bib].dropna().astype(int)
                count_le_k = int((bib_marks <= k).sum())
                sum_le_k = int(bib_marks[bib_marks <= k].sum()) if count_le_k > 0 else 10**9
                stats.append((bib, count_le_k, sum_le_k))

            majority_candidates = [s for s in stats if s[1] >= majority_needed]
            if not majority_candidates:
                continue

            majority_candidates.sort(key=lambda x: (-x[1], x[2]))
            best_count = majority_candidates[0][1]
            best_sum = majority_candidates[0][2]
            best = [s for s in majority_candidates if s[1] == best_count and s[2] == best_sum]

            if len(best) == 1:
                picked = best[0][0]
                break

            tied = [s[0] for s in best]

        if picked is None:
            picked = sorted(tied, key=lambda b: (float(total_sum.loc[b]), str(b)))[0]

        final_order.append(picked)
        remaining_bibs.remove(picked)

    return pd.DataFrame(
        {
            "FinalRank": range(1, len(final_order) + 1),
            "Bib": final_order,
            "TotalMarkSum": [float(total_sum.loc[b]) for b in final_order],
        }
    )


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
          body {{ font-family: system-ui, Arial; max-width: 900px; margin: 0 auto; padding: 24px; }}
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
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return page(
        "Judging App",
        """
        <div class="card">
          <p><a href="/admin">Admin</a> | <a href="/judge">Judge</a></p>
          <p class="muted">MVP: create an event, add bibs, let judges submit placements, compute final results.</p>
        </div>
        """,
    )


# ---------- Admin ----------
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
          <td>{'✅' if e["locked"] else '🟡'}</td>
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
      <p class="muted">Keep your admin password. You’ll need it to edit/lock/compute.</p>
    </div>

    <div class="card">
      <h2>Existing Events</h2>
      <table>
        <thead><tr><th>ID</th><th>Name</th><th>Join Code</th><th>Locked</th><th></th></tr></thead>
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

        bibs = conn.execute("SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)).fetchall()
        judges = conn.execute(
            """
            SELECT j.id, j.judge_name, j.last_submit_at
            FROM judges j
            WHERE j.event_id=?
            ORDER BY j.judge_name
            """,
            (event_id,),
        ).fetchall()

    bib_list = ", ".join([b["bib"] for b in bibs]) if bibs else "(none yet)"

    judges_rows = ""
    for j in judges:
        judges_rows += f"""
        <tr>
          <td>{j["judge_name"]}</td>
          <td>{j["last_submit_at"] or ''}</td>
        </tr>
        """
    if not judges_rows:
        judges_rows = '<tr><td colspan="2" class="muted">No judges yet.</td></tr>'

    body = f"""
    <div class="card">
      <p><a href="/admin">← Back to Admin</a></p>
      <h2>{event["name"]}</h2>
      <p>Join Code: <span class="pill">{event["join_code"]}</span> (Judges use this at <a href="/judge">/judge</a>)</p>
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
        <p class="muted">This replaces the current bib list (and clears existing marks for removed bibs).</p>
        <button type="submit">Save Bibs</button>
      </form>
    </div>

    <div class="card">
      <h3>Judges</h3>
      <table>
        <thead><tr><th>Judge</th><th>Last Submit</th></tr></thead>
        <tbody>{judges_rows}</tbody>
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
      <p class="muted">After computing, you’ll get a results page + CSV downloads.</p>
    </div>
    """
    return page(f"Admin Event #{event_id}", body)


@app.post("/admin/event/{event_id}/set_bibs")
def admin_set_bibs(event_id: int, admin_password: str = Form(...), bibs: str = Form("")):
    require_admin(event_id, admin_password)

    # Parse bibs: allow commas or newlines
    raw = bibs.replace(",", "\n")
    bib_list = [b.strip() for b in raw.splitlines() if b.strip()]
    bib_list = list(dict.fromkeys(bib_list))  # dedupe preserving order

    with db() as conn:
        # Ensure event exists
        event = conn.execute("SELECT id FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        # Replace competitors
        conn.execute("DELETE FROM competitors WHERE event_id=?", (event_id,))
        for b in bib_list:
            conn.execute("INSERT OR IGNORE INTO competitors(event_id, bib) VALUES(?,?)", (event_id, b))

        # Remove marks for bibs that no longer exist (marks table uses bib text)
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


@app.post("/admin/event/{event_id}/compute", response_class=HTMLResponse)
def admin_compute(event_id: int, admin_password: str = Form(...)):
    require_admin(event_id, admin_password)

    with db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        bibs = [r["bib"] for r in conn.execute("SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)).fetchall()]
        judges = conn.execute("SELECT id, judge_name FROM judges WHERE event_id=? ORDER BY judge_name", (event_id,)).fetchall()

        if not bibs:
            return page("Compute Results", '<div class="card">No bibs yet. Add bibs first.</div>')
        if not judges:
            return page("Compute Results", '<div class="card">No judges yet.</div>')

        # Build judge matrix
        data = {}
        for j in judges:
            placements = {}
            for b in bibs:
                m = conn.execute(
                    "SELECT placement FROM marks WHERE event_id=? AND judge_id=? AND bib=?",
                    (event_id, j["id"], b),
                ).fetchone()
                placements[b] = m["placement"] if m else None
            data[j["judge_name"]] = placements

    judge_matrix = pd.DataFrame.from_dict(data, orient="index")  # rows=judges, cols=bibs

    # Compute results
    try:
        final_df = skating_rank(judge_matrix)
    except Exception as e:
        return page("Compute Results", f'<div class="card">Error computing results: <pre>{e}</pre></div>')

    # Render results table and downloads
    result_rows = ""
    for _, r in final_df.iterrows():
        result_rows += f"<tr><td>{int(r['FinalRank'])}</td><td>{r['Bib']}</td><td>{r['TotalMarkSum']}</td></tr>"

    body = f"""
    <div class="card">
      <p><a href="/admin/event/{event_id}">← Back to Event</a></p>
      <h2>Results: {event["name"]}</h2>
      <p>
        <a href="/admin/event/{event_id}/download/results?admin_password={admin_password}">Download Results CSV</a>
        &nbsp;|&nbsp;
        <a href="/admin/event/{event_id}/download/matrix?admin_password={admin_password}">Download Judge Matrix CSV</a>
      </p>
      <table>
        <thead><tr><th>FinalRank</th><th>Bib</th><th>TotalMarkSum</th></tr></thead>
        <tbody>{result_rows}</tbody>
      </table>
    </div>
    """
    return page("Results", body)


@app.get("/admin/event/{event_id}/download/results")
def download_results(event_id: int, admin_password: str):
    require_admin(event_id, admin_password)

    with db() as conn:
        bibs = [r["bib"] for r in conn.execute("SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)).fetchall()]
        judges = conn.execute("SELECT id, judge_name FROM judges WHERE event_id=? ORDER BY judge_name", (event_id,)).fetchall()
        data = {}
        for j in judges:
            placements = {}
            for b in bibs:
                m = conn.execute(
                    "SELECT placement FROM marks WHERE event_id=? AND judge_id=? AND bib=?",
                    (event_id, j["id"], b),
                ).fetchone()
                placements[b] = m["placement"] if m else None
            data[j["judge_name"]] = placements

    judge_matrix = pd.DataFrame.from_dict(data, orient="index")
    final_df = skating_rank(judge_matrix)

    csv_buf = StringIO()
    final_df.to_csv(csv_buf, index=False)

    return Response(
        content=csv_buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="event_{event_id}_results.csv"'},
    )


@app.get("/admin/event/{event_id}/download/matrix")
def download_matrix(event_id: int, admin_password: str):
    require_admin(event_id, admin_password)

    with db() as conn:
        bibs = [r["bib"] for r in conn.execute("SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)).fetchall()]
        judges = conn.execute("SELECT id, judge_name FROM judges WHERE event_id=? ORDER BY judge_name", (event_id,)).fetchall()

        rows = []
        for j in judges:
            row = {"Judge": j["judge_name"]}
            for b in bibs:
                m = conn.execute(
                    "SELECT placement FROM marks WHERE event_id=? AND judge_id=? AND bib=?",
                    (event_id, j["id"], b),
                ).fetchone()
                row[str(b)] = m["placement"] if m else None
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)

    return Response(
        content=csv_buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="event_{event_id}_judge_matrix.csv"'},
    )


# ---------- Judge ----------
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
            return page("Judge", '<div class="card">Invalid join code.</div>')

        if event["locked"]:
            return page("Judge", '<div class="card">This event is locked. No more submissions.</div>')

        token = secrets.token_urlsafe(16)
        # Upsert judge by name
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

    # Redirect with token (simple MVP auth)
    return RedirectResponse(url=f"/judge/event/{event['id']}?judge_id={judge_id}&token={token}", status_code=303)


def require_judge(event_id: int, judge_id: int, token: str):
    with db() as conn:
        row = conn.execute(
            "SELECT id, judge_name FROM judges WHERE id=? AND event_id=? AND judge_token=?",
            (judge_id, event_id, token),
        ).fetchone()
        if not row:
            raise HTTPException(403, "Invalid judge session.")
        return row


@app.get("/judge/event/{event_id}", response_class=HTMLResponse)
def judge_event(event_id: int, judge_id: int, token: str):
    judge = require_judge(event_id, judge_id, token)

    with db() as conn:
        event = conn.execute("SELECT name, locked FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        if event["locked"]:
            return page("Judge", '<div class="card">This event is locked. Submissions are closed.</div>')

        bibs = [r["bib"] for r in conn.execute("SELECT bib FROM competitors WHERE event_id=? ORDER BY bib", (event_id,)).fetchall()]
        if not bibs:
            return page("Judge", '<div class="card">No bibs posted yet. Check back soon.</div>')

        # Existing marks
        existing = {}
        for b in bibs:
            m = conn.execute(
                "SELECT placement FROM marks WHERE event_id=? AND judge_id=? AND bib=?",
                (event_id, judge_id, b),
            ).fetchone()
            existing[b] = m["placement"] if m else ""

    # Build form rows
    rows = ""
    for b in bibs:
        rows += f"""
        <tr>
          <td>{b}</td>
          <td><input name="p__{b}" inputmode="numeric" pattern="[0-9]*" placeholder="1" value="{existing[b]}"/></td>
        </tr>
        """

    body = f"""
    <div class="card">
      <p><a href="/judge">← Back</a></p>
      <h2>{event["name"]}</h2>
      <p>Judge: <b>{judge["judge_name"]}</b></p>
      <p class="muted">Enter placements (1 = best). Use whole numbers.</p>
      <form method="post" action="/judge/event/{event_id}/submit">
        <input type="hidden" name="judge_id" value="{judge_id}" />
        <input type="hidden" name="token" value="{token}" />
        <table>
          <thead><tr><th>Bib</th><th>Placement</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <button type="submit" style="margin-top:12px;">Submit</button>
      </form>
      <p class="muted">MVP note: this doesn’t enforce “no duplicate placements” yet. We can add that next.</p>
    </div>
    """
    return page("Judge Entry", body)


@app.post("/judge/event/{event_id}/submit")
async def judge_submit(event_id: int, judge_id: int = Form(...), token: str = Form(...), **form_data):
    judge = require_judge(event_id, judge_id, token)

    with db() as conn:
        event = conn.execute("SELECT locked FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")
        if event["locked"]:
            return page("Judge", '<div class="card">This event is locked. Submission rejected.</div>')

        bibs = [r["bib"] for r in conn.execute("SELECT bib FROM competitors WHERE event_id=?", (event_id,)).fetchall()]

        # Extract placements from form fields p__{bib}
        updates = []
        for b in bibs:
            key = f"p__{b}"
            val = form_data.get(key)
            if val is None or str(val).strip() == "":
                placement = None
            else:
                try:
                    placement = int(str(val).strip())
                except ValueError:
                    return page("Judge Entry", f'<div class="card">Invalid placement for bib {b}: "{val}"</div>')
            updates.append((event_id, judge_id, b, placement))

        # Upsert marks
        for (eid, jid, bib, placement) in updates:
            conn.execute(
                """
                INSERT INTO marks(event_id, judge_id, bib, placement)
                VALUES(?,?,?,?)
                ON CONFLICT(event_id, judge_id, bib) DO UPDATE SET placement=excluded.placement
                """,
                (eid, jid, bib, placement),
            )

        conn.execute(
            "UPDATE judges SET last_submit_at=? WHERE id=?",
            (datetime.utcnow().isoformat(timespec="seconds"), judge_id),
        )

    return RedirectResponse(url=f"/judge/event/{event_id}?judge_id={judge_id}&token={token}", status_code=303)
