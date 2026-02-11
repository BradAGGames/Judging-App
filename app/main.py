from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

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


def short_code(n: int = 4) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no confusing I/1/O/0
    return "".join(secrets.choice(alphabet) for _ in range(n))


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

            CREATE TABLE IF NOT EXISTS admin_sessions (
                token TEXT PRIMARY KEY,
                event_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS competitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                bib TEXT NOT NULL,
                competitor_name TEXT,
                UNIQUE(event_id, bib)
            );

            CREATE TABLE IF NOT EXISTS round_competitors (
                round_id INTEGER NOT NULL,
                bib TEXT NOT NULL,
                PRIMARY KEY (round_id, bib)
            );

            CREATE TABLE IF NOT EXISTS rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                round_name TEXT NOT NULL,
                round_type TEXT NOT NULL, -- 'final' or 'prelim'
                yes_count INTEGER,        -- prelim only
                alt_count INTEGER,        -- prelim only
                locked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS judges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                judge_name TEXT NOT NULL,
                judge_token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(event_id, judge_name)
            );

            CREATE TABLE IF NOT EXISTS marks (
                round_id INTEGER NOT NULL,
                judge_id INTEGER NOT NULL,
                bib TEXT NOT NULL,
                score REAL,
                PRIMARY KEY (round_id, judge_id, bib)
            );

            CREATE TABLE IF NOT EXISTS judge_round_submissions (
                round_id INTEGER NOT NULL,
                judge_id INTEGER NOT NULL,
                submitted_at TEXT NOT NULL,
                PRIMARY KEY (round_id, judge_id)
            );
            """
        )


@app.on_event("startup")
def _startup():
    init_db()


def require_admin(event_id: int, admin_password: str) -> None:
    with db() as conn:
        row = conn.execute("SELECT admin_pw_hash FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Event not found.")
        if sha256(admin_password) != row["admin_pw_hash"]:
            raise HTTPException(403, "Invalid admin password.")


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def create_admin_session(event_id: int) -> str:
    """Create a browser-session admin token for a specific event."""
    token = secrets.token_urlsafe(24)
    created_at = _utc_now_iso()
    expires_at = (datetime.utcnow().replace(microsecond=0) + timedelta(hours=12)).isoformat()
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO admin_sessions(token, event_id, created_at, expires_at) VALUES(?,?,?,?)",
            (token, int(event_id), created_at, expires_at),
        )
    return token

def clear_admin_session(token: str) -> None:
    if not token:
        return
    with db() as conn:
        conn.execute("DELETE FROM admin_sessions WHERE token=?", (token,))

def require_admin_session(event_id: int, request: Request) -> str:
    """Require an existing admin session (cookie) for this event."""
    token = request.cookies.get("admin_session") or ""
    if not token:
        raise HTTPException(403, "Admin session required.")
    with db() as conn:
        row = conn.execute(
            "SELECT event_id, expires_at FROM admin_sessions WHERE token=?",
            (token,),
        ).fetchone()
    if not row or int(row["event_id"]) != int(event_id):
        raise HTTPException(403, "Admin session required.")
    try:
        exp = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > exp:
            clear_admin_session(token)
            raise HTTPException(403, "Admin session expired.")
    except ValueError:
        clear_admin_session(token)
        raise HTTPException(403, "Admin session expired.")
    return token


def require_judge(event_id: int, judge_id: int, token: str) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute(
            "SELECT id, judge_name FROM judges WHERE id=? AND event_id=? AND judge_token=?",
            (judge_id, event_id, token),
        ).fetchone()
    if not row:
        raise HTTPException(403, "Invalid judge session.")
    return row


def page(title: str, body: str) -> HTMLResponse:
    html = f"""
    <html>
      <head>
        <title>{title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body {{ font-family: system-ui, Arial; max-width: 1200px; margin: 0 auto; padding: 22px; }}
          input, textarea, button, select {{ font-size: 16px; padding: 10px; }}
          textarea {{ width: 100%; }}
          .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 16px 0; }}
          .row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
          .row > * {{ flex: 1; min-width: 220px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
          th {{ text-align: left; background: #f7f7f7; }}
          .muted {{ color: #666; }}
          .pill {{ display:inline-block; padding:4px 10px; border:1px solid #ddd; border-radius:999px; }}
          a {{ text-decoration: none; }}
          .danger {{ color: #b00020; }}
          .ok {{ color: #2e7d32; }}
          .score-pill {{ display:inline-block; min-width: 44px; text-align:center; border:1px solid #ddd; border-radius:999px; padding:2px 8px; margin-left: 10px; }}
          .bg-green {{ background: #e7f6ea; }}
          .bg-yellow {{ background: #fff7df; }}
          .bg-red {{ background: #ffe5e5; }}
          code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 6px; }}
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
# Skating (majority-based)
# -----------------------
def skating_rank(judge_placements: pd.DataFrame) -> pd.DataFrame:
    """
    judge_placements:
      index = judge_name
      columns = bib
      values = placement (1=best)

    Returns DataFrame: Place, Bib
    """
    marks = judge_placements.apply(pd.to_numeric, errors="coerce")
    marks = marks.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if marks.shape[0] == 0 or marks.shape[1] == 0:
        raise ValueError("No usable placement data.")

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

    return pd.DataFrame({"Place": range(1, len(final_order) + 1), "Bib": final_order})


# -----------------------
# Conversions
# -----------------------
def scores_to_placements(scores_by_bib: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(scores_by_bib.items(), key=lambda kv: (-kv[1], str(kv[0])))
    placements: Dict[str, int] = {}
    for idx, (bib, _score) in enumerate(ordered, start=1):
        placements[str(bib)] = idx
    return placements


def load_competitors(event_id: int) -> pd.DataFrame:
    with db() as conn:
        rows = conn.execute(
            "SELECT bib, competitor_name FROM competitors WHERE event_id=? ORDER BY bib",
            (event_id,),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Bib", "Competitor"])
    return pd.DataFrame(
        {"Bib": [r["bib"] for r in rows], "Competitor": [r["competitor_name"] or "" for r in rows]}
    )




def load_competitors_for_round(round_id: int) -> Tuple[int, pd.DataFrame]:
    """Return (event_id, competitors_df) for a round.

    If round_competitors has rows for this round, use only those bibs (names pulled from event competitors table).
    Otherwise, fall back to all event competitors.
    """
    with db() as conn:
        rnd = conn.execute("SELECT event_id FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise ValueError("Round not found.")
        event_id = int(rnd["event_id"])
        rc = conn.execute(
            "SELECT bib FROM round_competitors WHERE round_id=? ORDER BY bib",
            (round_id,),
        ).fetchall()

        if not rc:
            return event_id, load_competitors(event_id)

        bibs = [str(r["bib"]) for r in rc]
        # pull names from event competitors
        rows = conn.execute(
            "SELECT bib, competitor_name FROM competitors WHERE event_id=? AND bib IN (%s) ORDER BY bib"
            % (",".join(["?"] * len(bibs))),
            (event_id, *bibs),
        ).fetchall()

    # ensure we preserve the round-specific ordering and include blank names if missing
    name_map = {str(r["bib"]): (r["competitor_name"] or "") for r in rows}
    return event_id, pd.DataFrame({"Bib": bibs, "Competitor": [name_map.get(b, "") for b in bibs]})


def load_round_scores(round_id: int) -> Tuple[int, str, pd.DataFrame, pd.DataFrame]:
    """
    Returns: event_id, round_type, scores_df(rows=judge, cols=bib), competitors_df(Bib, Competitor)
    """
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise ValueError("Round not found.")
        event_id = int(rnd["event_id"])
        round_type = rnd["round_type"]

        event_id2, comps = load_competitors_for_round(round_id)
        event_id = event_id2
        bibs = comps["Bib"].astype(str).tolist()

        judges = conn.execute(
            "SELECT id, judge_name FROM judges WHERE event_id=? ORDER BY judge_name",
            (event_id,),
        ).fetchall()

        if not bibs:
            raise ValueError("No bibs found.")
        if not judges:
            raise ValueError("No judges yet.")

        data = {}
        for j in judges:
            row = {}
            for b in bibs:
                m = conn.execute(
                    "SELECT score FROM marks WHERE round_id=? AND judge_id=? AND bib=?",
                    (round_id, j["id"], b),
                ).fetchone()
                row[str(b)] = None if (m is None or m["score"] is None) else float(m["score"])
            data[j["judge_name"]] = row

    scores_df = pd.DataFrame.from_dict(data, orient="index")
    scores_df = scores_df[[str(b) for b in bibs]]
    return event_id, round_type, scores_df, comps


def judge_has_submitted(round_id: int, judge_id: int) -> bool:
    with db() as conn:
        row = conn.execute(
            "SELECT submitted_at FROM judge_round_submissions WHERE round_id=? AND judge_id=?",
            (round_id, judge_id),
        ).fetchone()
    return row is not None


# -----------------------
# Prelim callback math
# -----------------------
def prelim_points_for_rank(rank: int, y: int, a: int) -> float:
    if rank <= y:
        return 10.0
    if y < rank <= y + a:
        k = rank - y  # 1..a
        return round(4.5 - 0.1 * (k - 1), 3)
    return 0.0


def compute_prelim_callback(scores_df: pd.DataFrame, y: int, a: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      placements_df (judge x bib placements),
      points_df (judge x bib points),
      summary_df with columns: Bib, TotalPoints, Callback (Y/N/A1/A2...)
    """
    scores = scores_df.apply(pd.to_numeric, errors="coerce")
    if scores.isna().any().any():
        raise ValueError("Missing score(s) for one or more judges/bibs.")

    placements_rows = {}
    points_rows = {}

    for judge in scores.index:
        row = scores.loc[judge].to_dict()
        row_clean = {str(b): round(float(v), 1) for b, v in row.items()}
        if len(set(row_clean.values())) != len(row_clean.values()):
            raise ValueError(f"Judge '{judge}' has duplicate scores.")

        placements = scores_to_placements(row_clean)
        placements_rows[judge] = placements

        pts = {bib: prelim_points_for_rank(rank, y, a) for bib, rank in placements.items()}
        points_rows[judge] = pts

    placements_df = pd.DataFrame.from_dict(placements_rows, orient="index")[scores.columns]
    points_df = pd.DataFrame.from_dict(points_rows, orient="index")[scores.columns]

    totals = points_df.sum(axis=0)
    summary = pd.DataFrame({"Bib": scores.columns.astype(str), "TotalPoints": [float(totals[b]) for b in scores.columns]})
    summary = summary.sort_values(by=["TotalPoints", "Bib"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    # Callback rules: YES = top y, Alternates = next a
    yes_n = min(len(summary), y)
    alt_n = min(len(summary) - yes_n, a)

    labels = ["Y"] * yes_n + [f"A{i+1}" for i in range(alt_n)] + ["N"] * (len(summary) - yes_n - alt_n)
    summary["Callback"] = labels
    return placements_df, points_df, summary


# -----------------------
# Bib parsing (NEW)
# -----------------------
def parse_bib_entries(text: str) -> List[Tuple[str, str]]:
    """
    Accepts:
      - "101 Brad, 102 Jaden, 103 Ryan"
      - one per line
      - bib only
      - "101 | Brad"
    Strategy:
      Split into candidate chunks by newlines or commas.
      For each chunk:
        - if contains '|': bib | name
        - else: first token is bib, rest is name (optional)
    """
    raw = (text or "").strip()
    if not raw:
        return []

    # split by newline or comma
    chunks = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        chunks.extend(parts)

    parsed: List[Tuple[str, str]] = []
    for chunk in chunks:
        if not chunk:
            continue

        if "|" in chunk:
            left, right = chunk.split("|", 1)
            bib = left.strip()
            name = right.strip()
        else:
            # first token is bib (allow "219/210" too)
            m = re.match(r"^\s*([^\s]+)\s*(.*)$", chunk)
            if not m:
                continue
            bib = (m.group(1) or "").strip()
            name = (m.group(2) or "").strip()

        if bib:
            parsed.append((bib, name))

    # dedupe by bib, keep first name encountered
    seen: Dict[str, str] = {}
    for bib, name in parsed:
        if bib not in seen:
            seen[bib] = name
    return list(seen.items())


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
            Events contain rounds. Finals use Skating. Prelims use callback math (Y/A) with sliders and no duplicates.
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
          <input name="event_name" placeholder="Event name" required />
        </div>
        <button type="submit">Create</button>
      </form>
      <p class="muted">Join code is 4 characters. Admin password is required for edits/deletes.</p>
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
    join_code = short_code(4)
    with db() as conn:
        conn.execute(
            "INSERT INTO events(name, admin_pw_hash, join_code, locked, created_at) VALUES(?,?,?,?,?)",
            (event_name.strip(), sha256(admin_password), join_code, 0, datetime.utcnow().isoformat()),
        )
        event_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    token = create_admin_session(int(event_id))
    resp = RedirectResponse(url=f"/admin/event/{event_id}", status_code=303)
    resp.set_cookie("admin_session", token, httponly=True, samesite="lax")
    return resp


@app.get("/admin/event/{event_id}", response_class=HTMLResponse)
def admin_event(event_id: int, request: Request):
    # Require admin session for this event. If missing, prompt for password once.
    try:
        require_admin_session(event_id, request)
    except HTTPException:
        return page(
            "Admin Login",
            f'''
            <div class="card">
              <p><a href="/admin">← Back to Admin</a></p>
              <h2>Unlock Event #{event_id}</h2>
              <form method="post" action="/admin/event/{event_id}/login">
                <div class="row">
                  <input name="admin_password" placeholder="Admin password" type="password" required />
                </div>
                <button type="submit">Unlock</button>
              </form>
              <p class="muted">You’ll only need this again if you close your browser or revisit later.</p>
            </div>
            '''
        )

    with db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        rounds = conn.execute(
            "SELECT * FROM rounds WHERE event_id=? ORDER BY id DESC",
            (event_id,),
        ).fetchall()

        comps = conn.execute(
            "SELECT bib, competitor_name FROM competitors WHERE event_id=? ORDER BY bib",
            (event_id,),
        ).fetchall()

        judges = conn.execute(
            "SELECT judge_name FROM judges WHERE event_id=? ORDER BY judge_name",
            (event_id,),
        ).fetchall()

    rounds_rows = ""
    for r in rounds:
        typ = "Final" if r["round_type"] == "final" else f"Prelim (Y={r['yes_count']}, A={r['alt_count']})"
        rounds_rows += f"""
        <tr>
          <td>{r["id"]}</td>
          <td>{r["round_name"]}</td>
          <td>{typ}</td>
          <td>{'🔒' if r["locked"] else '🟢'}</td>
          <td><a href="/admin/round/{r['id']}">Open</a></td>
        </tr>
        """
    if not rounds_rows:
        rounds_rows = '<tr><td colspan="5" class="muted">No rounds yet.</td></tr>'

    comps_rows = ""
    for c in comps:
        comps_rows += f"<tr><td>{c['bib']}</td><td>{c['competitor_name'] or ''}</td></tr>"
    if not comps_rows:
        comps_rows = '<tr><td colspan="2" class="muted">No competitors yet.</td></tr>'

    judges_list = ", ".join([j["judge_name"] for j in judges]) if judges else "(none yet)"
    msg = request.query_params.get("msg") or ""
    msg_js = msg.replace("\\", "\\\\").replace("'", "\\'")

    body = f"""
    {f"<script>alert('{msg_js}');</script>" if msg_js else ""}
    <div class="card">
      <p><a href="/admin">← Back to Admin</a></p>
      <h2>{event["name"]}</h2>
      <p>Join Code: <span class="pill">{event["join_code"]}</span> (Judges go to <a href="/judge">/judge</a>)</p>
      <p class="muted">Judges: {judges_list}</p>
    </div>

    <div class="card">
      <h3>Rounds</h3>
      <table>
        <thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Status</th><th></th></tr></thead>
        <tbody>{rounds_rows}</tbody>
      </table>

      <h4 style="margin-top:16px;">Create Round</h4>
      <form method="post" action="/admin/event/{event_id}/create_round">
        <div class="row">
          <input name="admin_password" placeholder="Admin password" type="password" required />
          <input name="round_name" placeholder="Round name (e.g., Prelims / Finals)" required />
          <select id="roundType" name="round_type" required onchange="togglePrelimFields()">
            <option value="final">Final (Skating)</option>
            <option value="prelim">Prelim (Callback)</option>
          </select>
        </div>

        <div id="prelimFields" style="display:none; margin-top:12px;">
          <div class="row">
            <input name="yes_count" placeholder="Prelim YES count (Y) e.g., 7" type="number" min="1" />
            <input name="alt_count" placeholder="Prelim ALT count (A) e.g., 3" type="number" min="0" />
          </div>
          <p class="muted">These fields only apply to Prelim rounds.</p>
        </div>

        <button type="submit" style="margin-top:12px;">Create Round</button>
      </form>

      <script>
        function togglePrelimFields() {{
          const rt = document.getElementById('roundType').value;
          const box = document.getElementById('prelimFields');
          box.style.display = (rt === 'prelim') ? 'block' : 'none';
        }}
        togglePrelimFields();
      </script>
    </div>

    <div class="card">
      <h3>Competitors (bibs + optional names)</h3>
      <form method="post" action="/admin/event/{event_id}/set_bibs">
        <textarea name="bibs" rows="7" placeholder="Accepted formats:

101 Brad Gallow, 102 Jaden Pfeiffer, 103 Ryan Pflumm

or one per line:
101 Brad Gallow
102 Jaden Pfeiffer

or bib-only:
101
102
103

or pipe:
101 | Brad Gallow"></textarea>
        <button type="submit">Save Competitors</button>

      <div class="card" style="border:none; padding:0; margin:12px 0 0 0;">
        <h4 style="margin:10px 0 6px 0;">Add Competitors (append / update)</h4>
        <form method="post" action="/admin/event/{event_id}/add_bibs">
          <textarea name="bibs" rows="4" placeholder="Add new bibs (same formats). Existing bibs are kept; names update if provided."></textarea>
          <button type="submit" style="margin-top:8px;">Add</button>
        </form>
      </div>

      </form>

      <table style="margin-top:12px;">
        <thead><tr><th>Bib</th><th>Competitor</th></tr></thead>
        <tbody>{comps_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Delete Event</h3>
      <form method="post" action="/admin/event/{event_id}/delete" onsubmit="return confirm('Delete this event? This deletes rounds, judges, and marks.');">
        <button type="submit" style="background:#ffe5e5;">Delete Event</button>
      </form>
    </div>
    """
    return page(f"Admin Event #{event_id}", body)



@app.post("/admin/event/{event_id}/login")
def admin_event_login(event_id: int, admin_password: str = Form(...)):
    require_admin_session(event_id, request)
    token = create_admin_session(event_id)
    resp = RedirectResponse(url=f"/admin/event/{event_id}", status_code=303)
    resp.set_cookie("admin_session", token, httponly=True, samesite="lax")
    return resp


@app.post("/admin/event/{event_id}/delete")
def admin_delete_event(event_id: int, admin_password: str = Form(...)):
    require_admin_session(event_id, request)
    with db() as conn:
        round_ids = [r["id"] for r in conn.execute("SELECT id FROM rounds WHERE event_id=?", (event_id,)).fetchall()]
        for rid in round_ids:
            conn.execute("DELETE FROM marks WHERE round_id=?", (rid,))
            conn.execute("DELETE FROM judge_round_submissions WHERE round_id=?", (rid,))
        conn.execute("DELETE FROM rounds WHERE event_id=?", (event_id,))
        conn.execute("DELETE FROM competitors WHERE event_id=?", (event_id,))
        conn.execute("DELETE FROM judges WHERE event_id=?", (event_id,))
        conn.execute("DELETE FROM events WHERE id=?", (event_id,))
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/event/{event_id}/create_round")
def admin_create_round(
    event_id: int,
    request: Request,
    round_name: str = Form(...),
    round_type: str = Form(...),
    yes_count: Optional[int] = Form(None),
    alt_count: Optional[int] = Form(None),
):
    require_admin(event_id, admin_password)
    round_type = round_type.strip().lower()
    if round_type not in ("final", "prelim"):
        raise HTTPException(400, "Invalid round type.")

    y = int(yes_count) if yes_count is not None and str(yes_count) != "" else None
    a = int(alt_count) if alt_count is not None and str(alt_count) != "" else None

    if round_type == "prelim":
        if y is None or y < 1:
            raise HTTPException(400, "Prelim requires YES count (Y) >= 1.")
        if a is None or a < 0:
            raise HTTPException(400, "Prelim requires ALT count (A) >= 0.")

    with db() as conn:
        conn.execute(
            """
            INSERT INTO rounds(event_id, round_name, round_type, yes_count, alt_count, locked, created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (event_id, round_name.strip(), round_type, y, a, 0, datetime.utcnow().isoformat()),
        )
        rid = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return RedirectResponse(url=f"/admin/event/{event_id}?msg=Round%20{round_name.strip()}%20created", status_code=303)


@app.post("/admin/event/{event_id}/set_bibs")
def admin_set_bibs(event_id: int, request: Request, bibs: str = Form("")):
    require_admin(event_id, admin_password)

    bib_list = parse_bib_entries(bibs)

    with db() as conn:
        conn.execute("DELETE FROM competitors WHERE event_id=?", (event_id,))
        for bib, name in bib_list:
            conn.execute(
                "INSERT OR IGNORE INTO competitors(event_id, bib, competitor_name) VALUES(?,?,?)",
                (event_id, str(bib), (name or None)),
            )
    return RedirectResponse(url=f"/admin/event/{event_id}", status_code=303)



@app.post("/admin/event/{event_id}/add_bibs")
def admin_add_bibs(event_id: int, request: Request, bibs: str = Form("")):
    require_admin_session(event_id, request)
    bib_list = parse_bib_entries(bibs)

    with db() as conn:
        for bib, name in bib_list:
            conn.execute(
                """
                INSERT INTO competitors(event_id, bib, competitor_name)
                VALUES(?,?,?)
                ON CONFLICT(event_id, bib) DO UPDATE SET
                    competitor_name = CASE
                        WHEN excluded.competitor_name IS NULL OR excluded.competitor_name = ''
                        THEN competitors.competitor_name
                        ELSE excluded.competitor_name
                    END
                """,
                (event_id, str(bib), (name or None)),
            )
    return RedirectResponse(url=f"/admin/event/{event_id}?msg=Competitors%20added", status_code=303)


@app.get("/admin/round/{round_id}", response_class=HTMLResponse)
def admin_round(round_id: int, request: Request):
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        event = conn.execute("SELECT * FROM events WHERE id=?", (rnd["event_id"],)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        require_admin_session(int(event["id"]), request)

        judges = conn.execute(
            "SELECT id, judge_name FROM judges WHERE event_id=? ORDER BY judge_name",
            (event["id"],),
        ).fetchall()

        subs = conn.execute(
            "SELECT judge_id, submitted_at FROM judge_round_submissions WHERE round_id=?",
            (round_id,),
        ).fetchall()
        submitted_map = {s["judge_id"]: s["submitted_at"] for s in subs}

    typ = "Final (Skating)" if rnd["round_type"] == "final" else f"Prelim (Callback) Y={rnd['yes_count']} A={rnd['alt_count']}"
    judges_rows = ""
    for j in judges:
        judges_rows += f"<tr><td>{j['judge_name']}</td><td>{submitted_map.get(j['id'], '')}</td></tr>"
    if not judges_rows:
        judges_rows = '<tr><td colspan="2" class="muted">No judges yet.</td></tr>'

    body = f"""
    <div class="card">
      <p><a href="/admin/event/{event['id']}">← Back to Event</a></p>
      <h2>{event["name"]} | {rnd["round_name"]}</h2>
      <p>Type: <span class="pill">{typ}</span></p>
      <p>Status: {'🔒 Locked' if rnd["locked"] else '🟢 Open'}</p>
    </div>

    <div class="card">
      <h3>Judge Submissions</h3>
      <table>
        <thead><tr><th>Judge</th><th>Submitted At</th></tr></thead>
        <tbody>{judges_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Controls</h3>
      <form method="post" action="/admin/round/{round_id}/toggle_lock" style="margin-bottom:12px;">
        <button type="submit">{'Unlock' if rnd["locked"] else 'Lock'} Round</button>
      </form>

      <form method="post" action="/admin/round/{round_id}/compute">
        <button type="submit">Compute Results</button>
      </form>

      <p class="muted">
        Downloads:
        <a href="/admin/round/{round_id}/download/raw_scores">Raw scores CSV</a> |
        <a href="/admin/round/{round_id}/download/placements">Placements CSV</a>
      </p>
    </div>
    """
    return page("Admin Round", body)


@app.post("/admin/round/{round_id}/toggle_lock")
def admin_toggle_round_lock(round_id: int, request: Request):
    with db() as conn:
        rnd = conn.execute("SELECT event_id, locked FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        require_admin_session(int(rnd["event_id"]), request)
        new_val = 0 if rnd["locked"] else 1
        conn.execute("UPDATE rounds SET locked=? WHERE id=?", (new_val, round_id))
    return RedirectResponse(url=f"/admin/round/{round_id}", status_code=303)


@app.post("/admin/round/{round_id}/compute", response_class=HTMLResponse)
def admin_compute_round(
    round_id: int,
    request: Request,
    yes_override: Optional[int] = Form(None),
    alt_override: Optional[int] = Form(None),
):
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        require_admin_session(int(rnd["event_id"]), request)

    try:
        event_id, round_type, scores_df, comps_df = load_round_scores(round_id)
    except Exception as e:
        return page("Compute Results", f'<div class="card"><p class="danger">Error:</p><pre>{e}</pre></div>')

    scores = scores_df.apply(pd.to_numeric, errors="coerce")
    if scores.isna().any().any():
        return page("Compute Results", '<div class="card"><p class="danger">Missing score(s) detected.</p></div>')

    for j in scores.index:
        vals = [round(float(x), 1) for x in list(scores.loc[j].values)]
        if len(set(vals)) != len(vals):
            return page("Compute Results", f'<div class="card"><p class="danger">Duplicate score(s) for judge {j}.</p></div>')

    placements_rows = {}
    for j in scores.index:
        row = {str(b): round(float(v), 1) for b, v in scores.loc[j].to_dict().items()}
        placements_rows[j] = scores_to_placements(row)
    placements_df = pd.DataFrame.from_dict(placements_rows, orient="index")[scores.columns]

    bib_to_name = dict(zip(comps_df["Bib"].astype(str), comps_df["Competitor"].astype(str)))

    if round_type == "final":
        final_df = skating_rank(placements_df)
        judges = list(placements_df.index)

        rows_html = ""
        for _, r in final_df.iterrows():
            bib = str(r["Bib"])
            place = int(r["Place"])
            comp_name = bib_to_name.get(bib, "")
            marks = [int(placements_df.loc[j, bib]) for j in judges]
            marks_sorted = "-".join(str(x) for x in sorted(marks))
            judge_cells = "".join(f"<td>{int(placements_df.loc[j, bib])}</td>" for j in judges)

            rows_html += f"""
            <tr>
              <td>{place}</td>
              <td>{comp_name}</td>
              {judge_cells}
              <td>{bib}</td>
              <td>{marks_sorted}</td>
            </tr>
            """

        head_judges = "".join(f"<th>{j}</th>" for j in judges)

        raw_rows = ""
        for j in scores.index:
            raw_rows += "<tr>" + "".join([f"<td>{j}</td>"] + [f"<td>{float(scores.loc[j, b]):.1f}</td>" for b in scores.columns]) + "</tr>"
        raw_head = "".join([f"<th>{b}</th>" for b in scores.columns])

        body = f"""
        <div class="card">
          <p><a href="/admin/round/{round_id}">← Back to Round</a></p>
          <h2>Final Results (Skating)</h2>
          <p>
            <a href="/admin/round/{round_id}/download/final_results">Download Results CSV</a> |
            <a href="/admin/round/{round_id}/download/placements">Download Placements CSV</a> |
            <a href="/admin/round/{round_id}/download/raw_scores">Download Raw Scores CSV</a>
          </p>

          <table>
            <thead>
              <tr>
                <th>Place</th>
                <th>Competitor</th>
                {head_judges}
                <th>BIB</th>
                <th>Marks Sorted</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          
      </table>
    </div>

    <div class="card">
      <h3>Adjust Recall Cut (Admin)</h3>
      <form method="post" action="/admin/round/{round_id}/compute">
        <div class="row">
          <input name="admin_password" placeholder="Admin password" type="password" required />
          <input name="yes_override" placeholder="YES (Y)" type="number" min="1" value="{y}" required />
          <input name="alt_override" placeholder="Alternates (A)" type="number" min="0" value="{a}" required />
        </div>
        <button type="submit">Recompute with New Cut</button>
        <p class="muted">This updates the round’s Y/A cut and recomputes labels and points. Judges’ raw scores stay unchanged.</p>
      </form>
    </div>

    <div class="card">
      <h3>Create Finals from Promoted (Admin)</h3>
      <form method="post" action="/admin/round/{round_id}/create_final_from_yes" onsubmit="return confirm('Create a new Final round using only the Y-promoted bibs?');">
        <div class="row">
          <input name="admin_password" placeholder="Admin password" type="password" required />
          <input name="final_round_name" placeholder="Final round name" value="Finals (Auto)" />
        </div>
        <button type="submit">Create Final from Y</button>
        <p class="muted">Creates a new Final round and limits its competitor list to the Y-promoted dancers.</p>
      </form>
    </div>

    <div class="card">
      <h3>Judge Breakdown (Y / A / N)</h3>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Competitor</th>
            {detail_head_judges}
            <th>BIB</th>
            <th>Counts (Y-A-N)</th>
          </tr>
        </thead>
        <tbody>{detail_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Raw Scores (Admin Only)</h3>

          <table>
            <thead><tr><th>Judge</th>{raw_head}</tr></thead>
            <tbody>{raw_rows}</tbody>
          </table>
        </div>
        """
        return page("Final Results", body)

    # Prelim callback
    y = int(yes_override) if yes_override is not None and str(yes_override) != "" else int(rnd["yes_count"] or 0)
    a = int(alt_override) if alt_override is not None and str(alt_override) != "" else int(rnd["alt_count"] or 0)

    # If admin provided overrides, persist them to the round so future computes use the new cut
    if yes_override is not None or alt_override is not None:
        with db() as conn:
            conn.execute("UPDATE rounds SET yes_count=?, alt_count=? WHERE id=?", (y, a, round_id))
    try:
        placements_df2, points_df, summary_df = compute_prelim_callback(scores_df, y, a)
    except Exception as e:
        return page("Prelim Results", f'<div class="card"><p class="danger">Error:</p><pre>{e}</pre></div>')

    rows_html = ""
    for _, r in summary_df.iterrows():
        bib = str(r["Bib"])
        comp_name = bib_to_name.get(bib, "")
        rows_html += f"""
        <tr>
          <td>{comp_name}</td>
          <td>{bib}</td>
          <td>{r["Callback"]}</td>
          <td>{r["TotalPoints"]:.3f}</td>
        </tr>
        """

    judges = list(scores_df.index)
    # Judge Y/A/N breakdown (competitor rows, judge columns)
    def _rank_to_label(rank: int) -> str:
        if rank <= y:
            return "Y"
        if rank <= y + a:
            return f"A{rank - y}"
        return "N"

    detail_rows = ""
    # use summary order (best to worst) for rows
    for idx_row, r in summary_df.iterrows():
        bib = str(r["Bib"])
        comp_name = bib_to_name.get(bib, "")
        per_j = []
        y_count = a_count = n_count = 0
        for j in judges:
            rk = int(placements_df2.loc[j, bib])
            lab = _rank_to_label(rk)
            if lab == "Y":
                y_count += 1
            elif lab.startswith("A"):
                a_count += 1
            else:
                n_count += 1
            per_j.append(lab)
        counts_str = f"{y_count}-{a_count}-{n_count}"
        judge_cells = "".join([f"<td>{lab}</td>" for lab in per_j])
        detail_rows += f"""
        <tr>
          <td>{idx_row+1}</td>
          <td>{comp_name}</td>
          {judge_cells}
          <td>{bib}</td>
          <td>{counts_str}</td>
        </tr>
        """

    detail_head_judges = "".join([f"<th>{j}</th>" for j in judges])

    # Raw scores judge x bib

    raw_head = "".join([f"<th>{b}</th>" for b in scores_df.columns])
    raw_rows = ""
    for j in judges:
        raw_rows += "<tr><td>" + j + "</td>" + "".join([f"<td>{float(scores_df.loc[j, b]):.1f}</td>" for b in scores_df.columns]) + "</tr>"

    # (FIX) Optional: build a competitor column aligned to competitor rows if you later use placement_matrix
    # The previous crash was from inserting values sized to columns into an index-sized table.

    body = f"""
    <div class="card">
      <p><a href="/admin/round/{round_id}">← Back to Round</a></p>
      <h2>Prelim Callback Results</h2>
      <p class="muted">YES = top {y}; Alternates = next {a} (A1..A{a}); rest N.</p>
      <p>
        <a href="/admin/round/{round_id}/download/prelim_summary">Download Summary CSV</a> |
        <a href="/admin/round/{round_id}/download/placements">Download Placements CSV</a> |
        <a href="/admin/round/{round_id}/download/raw_scores">Download Raw Scores CSV</a>
      </p>

      <table>
        <thead><tr><th>Competitor</th><th>BIB</th><th>Callback</th><th>Total Points</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Raw Scores (Admin Only)</h3>
      <table>
        <thead><tr><th>Judge</th>{raw_head}</tr></thead>
        <tbody>{raw_rows}</tbody>
      </table>
    </div>
    """
    return page("Prelim Results", body)



@app.post("/admin/round/{round_id}/create_final_from_yes")
def admin_create_final_from_yes(
    round_id: int,
    admin_password: str = Form(...),
    final_round_name: str = Form("Finals (Auto)"),
):
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        if rnd["round_type"] != "prelim":
            raise HTTPException(400, "This action is only for prelim rounds.")
        event_id = int(rnd["event_id"])
        require_admin(event_id, admin_password)

    # Compute current prelim results to determine Y promotions
    _event_id, _rt, scores_df, _comps_df = load_round_scores(round_id)
    y = int(rnd["yes_count"] or 0)
    a = int(rnd["alt_count"] or 0)

    _placements_df, _points_df, summary_df = compute_prelim_callback(scores_df, y, a)
    promoted = [str(b) for b in summary_df.loc[summary_df["Callback"] == "Y", "Bib"].astype(str).tolist()]
    if not promoted:
        return page("Create Final", '<div class="card"><p class="danger">No promoted (Y) bibs found.</p></div>')

    with db() as conn:
        conn.execute(
            """
            INSERT INTO rounds(event_id, round_name, round_type, yes_count, alt_count, locked, created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                event_id,
                (final_round_name.strip() or "Finals (Auto)"),
                "final",
                None,
                None,
                0,
                datetime.utcnow().isoformat(),
            ),
        )
        new_round_id = int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])

        # Limit the new final to promoted competitors
        for bib in promoted:
            conn.execute(
                "INSERT OR IGNORE INTO round_competitors(round_id, bib) VALUES(?,?)",
                (new_round_id, bib),
            )

    return RedirectResponse(url=f"/admin/round/{new_round_id}", status_code=303)


# -----------------------
# Admin downloads
# -----------------------
@app.get("/admin/round/{round_id}/download/raw_scores")
def download_raw_scores(round_id: int, request: Request):
    with db() as conn:
        rnd = conn.execute("SELECT event_id FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        require_admin_session(int(rnd["event_id"]), request)

    _event_id, _round_type, scores_df, _comps = load_round_scores(round_id)
    out = scores_df.copy()
    out.insert(0, "Judge", out.index)
    buf = StringIO()
    out.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="round_{round_id}_raw_scores.csv"'},
    )


@app.get("/admin/round/{round_id}/download/placements")
def download_placements(round_id: int, request: Request):
    with db() as conn:
        rnd = conn.execute("SELECT event_id FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        require_admin_session(int(rnd["event_id"]), request)

    _event_id, _round_type, scores_df, _comps = load_round_scores(round_id)
    scores = scores_df.apply(pd.to_numeric, errors="coerce")
    placements_rows = {}
    for j in scores.index:
        row = {str(b): round(float(v), 1) for b, v in scores.loc[j].to_dict().items()}
        placements_rows[j] = scores_to_placements(row)
    placements_df = pd.DataFrame.from_dict(placements_rows, orient="index")[scores.columns]
    out = placements_df.copy()
    out.insert(0, "Judge", out.index)
    buf = StringIO()
    out.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="round_{round_id}_placements.csv"'},
    )


@app.get("/admin/round/{round_id}/download/final_results")
def download_final_results(round_id: int, request: Request):
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        require_admin_session(int(rnd["event_id"]), request)
        if rnd["round_type"] != "final":
            raise HTTPException(400, "Not a final round.")

    _event_id, _rt, scores_df, comps_df = load_round_scores(round_id)
    bib_to_name = dict(zip(comps_df["Bib"].astype(str), comps_df["Competitor"].astype(str)))

    scores = scores_df.apply(pd.to_numeric, errors="coerce")
    placements_rows = {}
    for j in scores.index:
        row = {str(b): round(float(v), 1) for b, v in scores.loc[j].to_dict().items()}
        placements_rows[j] = scores_to_placements(row)
    placements_df = pd.DataFrame.from_dict(placements_rows, orient="index")[scores.columns]

    final_df = skating_rank(placements_df)
    judges = list(placements_df.index)

    out_rows = []
    for _, r in final_df.iterrows():
        bib = str(r["Bib"])
        marks = [int(placements_df.loc[j, bib]) for j in judges]
        out_rows.append(
            {
                "Place": int(r["Place"]),
                "Competitor": bib_to_name.get(bib, ""),
                **{j: int(placements_df.loc[j, bib]) for j in judges},
                "BIB": bib,
                "Marks Sorted": "-".join(str(x) for x in sorted(marks)),
            }
        )

    out = pd.DataFrame(out_rows)
    buf = StringIO()
    out.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="round_{round_id}_final_results.csv"'},
    )


@app.get("/admin/round/{round_id}/download/prelim_summary")
def download_prelim_summary(round_id: int, request: Request):
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        require_admin_session(int(rnd["event_id"]), request)
        if rnd["round_type"] != "prelim":
            raise HTTPException(400, "Not a prelim round.")

    _event_id, _rt, scores_df, comps_df = load_round_scores(round_id)
    bib_to_name = dict(zip(comps_df["Bib"].astype(str), comps_df["Competitor"].astype(str)))
    y = int(rnd["yes_count"] or 0)
    a = int(rnd["alt_count"] or 0)

    _placements_df, _points_df, summary_df = compute_prelim_callback(scores_df, y, a)
    summary_df.insert(0, "Competitor", [bib_to_name.get(str(b), "") for b in summary_df["Bib"].astype(str)])
    buf = StringIO()
    summary_df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="round_{round_id}_prelim_summary.csv"'},
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
          <input name="join_code" placeholder="Join code (4 chars)" required />
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
        event = conn.execute("SELECT id FROM events WHERE join_code=?", (join_code,)).fetchone()
        if not event:
            return page("Judge", '<div class="card"><p class="danger">Invalid join code.</p></div>')

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
                "INSERT INTO judges(event_id, judge_name, judge_token, created_at) VALUES(?,?,?,?)",
                (event["id"], judge_name, token, datetime.utcnow().isoformat()),
            )
            judge_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    return RedirectResponse(url=f"/judge/event/{event['id']}?judge_id={judge_id}&token={token}", status_code=303)


@app.get("/judge/event/{event_id}", response_class=HTMLResponse)
def judge_event(event_id: int, judge_id: int, token: str):
    judge = require_judge(event_id, judge_id, token)

    with db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

        rounds = conn.execute(
            "SELECT * FROM rounds WHERE event_id=? ORDER BY id DESC",
            (event_id,),
        ).fetchall()

    rows = ""
    for r in rounds:
        submitted = judge_has_submitted(int(r["id"]), judge_id)
        disabled = "🔒" if r["locked"] else ("✅ Submitted" if submitted else "🟢 Open")
        link = "" if (r["locked"] or submitted) else f'<a href="/judge/round/{r["id"]}?judge_id={judge_id}&token={token}">Open</a>'
        typ = "Final" if r["round_type"] == "final" else f"Prelim (Y={r['yes_count']}, A={r['alt_count']})"
        rows += f"""
        <tr>
          <td>{r["round_name"]}</td>
          <td>{typ}</td>
          <td>{disabled}</td>
          <td>{link}</td>
        </tr>
        """
    if not rows:
        rows = '<tr><td colspan="4" class="muted">No rounds created yet.</td></tr>'

    body = f"""
    <div class="card">
      <p><a href="/judge">← Back</a></p>
      <h2>{event["name"]}</h2>
      <p>Judge: <b>{judge["judge_name"]}</b></p>
    </div>

    <div class="card">
      <h3>Select a Round</h3>
      <table>
        <thead><tr><th>Round</th><th>Type</th><th>Status</th><th></th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class="muted">After you submit, you will not be able to reopen your scores for that round.</p>
    </div>
    """
    return page("Judge Event", body)


@app.get("/judge/round/{round_id}", response_class=HTMLResponse)
def judge_round(round_id: int, judge_id: int, token: str):
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        event = conn.execute("SELECT * FROM events WHERE id=?", (rnd["event_id"],)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

    judge = require_judge(int(event["id"]), judge_id, token)

    if rnd["locked"]:
        return page("Judge", '<div class="card"><p class="danger">This round is locked.</p></div>')

    if judge_has_submitted(round_id, judge_id):
        return page(
            "Judge",
            '<div class="card"><p class="ok">Submission received. This round is now locked for you.</p></div>',
        )

    event_id2, comps = load_competitors_for_round(round_id)
    # event_id2 should match event['id']

    if comps.empty:
        return page("Judge", '<div class="card"><p class="danger">No competitors posted yet.</p></div>')

    with db() as conn:
        existing = {}
        for b in comps["Bib"].astype(str).tolist():
            m = conn.execute(
                "SELECT score FROM marks WHERE round_id=? AND judge_id=? AND bib=?",
                (round_id, judge_id, b),
            ).fetchone()
            existing[b] = None if (m is None or m["score"] is None) else float(m["score"])

    y = int(rnd["yes_count"] or 0)
    a = int(rnd["alt_count"] or 0)
    is_prelim = rnd["round_type"] == "prelim"

    rows = ""
    for i_row, (_idx, row) in enumerate(comps.iterrows()):
        bib = str(row["Bib"])
        name = str(row["Competitor"] or "")
        val = existing.get(bib)
        if val is None:
            val = max(70.0, 100.0 - 0.1 * i_row)
        val = round(float(val), 1)
        label = f"{bib}" if not name else f"{bib} | {name}"
        rows += f"""
        <tr data-bib="{bib}">
          <td>{label}</td>
          <td style="min-width:380px;">
            <button type="button" onclick="nudge(this,-0.1)" aria-label="Decrease score">-</button>
            <input type="range" min="70" max="100" step="0.1" name="s__{bib}" value="{val:.1f}" oninput="syncVal(this)">
            <button type="button" onclick="nudge(this,0.1)" aria-label="Increase score">+</button>
            <span class="score-pill">{val:.1f}</span>
          </td>
        </tr>
        """

    body = f"""
    <div class="card">
      <p><a href="/judge/event/{event["id"]}?judge_id={judge_id}&token={token}">← Back to Rounds</a></p>
      <h2>{event["name"]} | {rnd["round_name"]}</h2>
      <p>Judge: <b>{judge["judge_name"]}</b></p>
      <p class="muted">
        Sliders 70–100 (0.1 increments). Use +/- for fine control. Duplicate scores are not allowed.
        {"Prelim colors: top Y green, next A yellow, rest red." if is_prelim else ""}
      </p>

      <div id="dupMsg" class="muted" style="margin-bottom:10px;"></div>

      <form method="post" action="/judge/round/{round_id}/submit" onsubmit="return confirmSubmit();">
        <input type="hidden" name="judge_id" value="{judge_id}" />
        <input type="hidden" name="token" value="{token}" />
        <table id="scoreTable">
          <thead><tr><th>Competitor</th><th>Score</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <button id="submitBtn" type="submit" style="margin-top:12px;">Submit</button>
      </form>
    </div>

    <script>
      const IS_PRELIM = {str(is_prelim).lower()};
      const Y = {y};
      const A = {a};

      function syncVal(slider) {{
        const pill = slider.parentElement.querySelector('.score-pill');
        pill.textContent = parseFloat(slider.value).toFixed(1);
        validateAndColor();
      }}

      function nudge(btn, delta) {{
        const slider = btn.parentElement.querySelector('input[type="range"]');
        let v = parseFloat(slider.value) + delta;
        v = Math.min(100, Math.max(70, v));
        slider.value = v.toFixed(1);
        syncVal(slider);
        validateAndColor();
      }}

      function validateDuplicates() {{
        const rows = Array.from(document.querySelectorAll('#scoreTable tbody tr'));
        const seen = new Map();
        let hasDup = false;

        rows.forEach(r => {{
          r.classList.remove('bg-green','bg-yellow','bg-red');
          r.style.background = '';
        }});

        rows.forEach(r => {{
          const slider = r.querySelector('input[type="range"]');
          const v = parseFloat(slider.value).toFixed(1);
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
          msg.textContent = "Duplicate score detected. Make every score unique.";
          msg.className = "danger";
          btn.disabled = true;
        }} else {{
          msg.textContent = "All scores unique ✅";
          msg.className = "ok";
          btn.disabled = false;
        }}
        return !hasDup;
      }}

      function applyPrelimColors() {{
        if (!IS_PRELIM) return;

        const rows = Array.from(document.querySelectorAll('#scoreTable tbody tr'));
        const scored = rows.map(r => {{
          const v = parseFloat(r.querySelector('input[type="range"]').value);
          return {{ r, v }};
        }});

        scored.sort((a,b) => b.v - a.v);

        scored.forEach((obj, idx) => {{
          obj.r.classList.remove('bg-green','bg-yellow','bg-red');
          if (idx < Y) obj.r.classList.add('bg-green');
          else if (idx < Y + A) obj.r.classList.add('bg-yellow');
          else obj.r.classList.add('bg-red');
        }});
      }}

      function validateAndColor() {{
        const ok = validateDuplicates();
        if (ok) applyPrelimColors();
      }}

      function confirmSubmit() {{
        const ok = validateDuplicates();
        if (!ok) return false;
        return confirm("Are you sure you want to submit? This action cannot be undone.");
      }}

      validateAndColor();
    </script>
    """
    return page("Judge Round", body)


@app.post("/judge/round/{round_id}/submit")
async def judge_round_submit(round_id: int, request: Request, judge_id: int = Form(...), token: str = Form(...)):
    with db() as conn:
        rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not rnd:
            raise HTTPException(404, "Round not found.")
        event = conn.execute("SELECT * FROM events WHERE id=?", (rnd["event_id"],)).fetchone()
        if not event:
            raise HTTPException(404, "Event not found.")

    _judge = require_judge(int(event["id"]), judge_id, token)

    if rnd["locked"]:
        return page("Judge", '<div class="card"><p class="danger">Round locked. Submission rejected.</p></div>')

    if judge_has_submitted(round_id, judge_id):
        return page("Judge", '<div class="card"><p class="ok">Already submitted.</p></div>')

    event_id2, comps = load_competitors_for_round(round_id)
    
    bibs = comps["Bib"].astype(str).tolist()
    if not bibs:
        return page("Judge", '<div class="card"><p class="danger">No competitors posted.</p></div>')

    form = await request.form()

    scores: Dict[str, float] = {}
    for b in bibs:
        key = f"s__{b}"
        if key not in form:
            return page("Judge", f'<div class="card"><p class="danger">Missing score for bib {b}.</p></div>')
        try:
            v = float(str(form[key]).strip())
        except ValueError:
            return page("Judge", f'<div class="card"><p class="danger">Invalid score for bib {b}.</p></div>')
        v = round(float(v), 1)
        if v < 70.0 or v > 100.0:
            return page("Judge", f'<div class="card"><p class="danger">Score out of range for bib {b}.</p></div>')
        scores[b] = float(v)

    if len(set(scores.values())) != len(scores.values()):
        return page("Judge", '<div class="card"><p class="danger">Duplicate scores detected. Fix and resubmit.</p></div>')

    with db() as conn:
        for bib, score in scores.items():
            conn.execute(
                """
                INSERT INTO marks(round_id, judge_id, bib, score)
                VALUES(?,?,?,?)
                ON CONFLICT(round_id, judge_id, bib) DO UPDATE SET score=excluded.score
                """,
                (round_id, judge_id, bib, score),
            )

        conn.execute(
            """
            INSERT INTO judge_round_submissions(round_id, judge_id, submitted_at)
            VALUES(?,?,?)
            """,
            (round_id, judge_id, datetime.utcnow().isoformat(timespec="seconds")),
        )

    return page("Submitted", '<div class="card"><p class="ok">Submitted ✅</p><p class="muted">You can’t reopen this round now.</p></div>')
