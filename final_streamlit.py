import streamlit as st
import requests, os, time, random
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
EVK_URL    = "http://YOUR_EVK_IP:5000"  # put your Dragonwing EVK's local network IP here
OLLAMA_URL = "http://localhost:11434/api/generate"
ALL_PLAYERS = ["Alex", "Jordan", "Priya", "Morgan", "Riley", "Casey"]

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DragonPitch",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── PALETTE ───────────────────────────────────────────────────────────────────
LAVENDER  = "#a482b8"
ORANGE    = "#df6d45"
BLACK     = "#130e08"
WHITE     = "#ffffff"
BG        = "#0f0b09"
CARD_BG   = "#1c1410"
CARD_BG2  = "#221a15"
BORDER    = "#2e2118"
BORDER2   = "#3d2e22"
TEXT_MUT  = "#8a7060"
TEXT_SOFT = "#c4a882"

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {BG};
    color: {WHITE};
  }}
  .stApp {{ background: {BG}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
  section[data-testid="stSidebar"] {{ display: none !important; }}
  .block-container {{
    padding: 0 2.5rem 4rem 2.5rem !important;
    max-width: 100% !important;
  }}
  div[data-testid="stButton"] button {{
    background: linear-gradient(135deg, {ORANGE}, #c85a32) !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    padding: 10px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(223,109,69,0.25) !important;
  }}
  div[data-testid="stButton"] button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(223,109,69,0.4) !important;
  }}
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {CARD_BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 10px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUT}; }}
  .player-card {{ transition: all 0.25s ease; }}
  .player-card:hover {{
    border-color: {LAVENDER} !important;
    box-shadow: 0 8px 32px rgba(164,130,184,0.18) !important;
    transform: translateY(-3px);
  }}
  /* Pitch table */
  .pitch-scroll-wrap {{
    max-height: 320px;
    overflow-y: auto;
    border-radius: 0 0 14px 14px;
  }}
  .pitch-table {{
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
  }}
  .pitch-table thead th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: #2a1e16;
    padding: 11px 16px;
    text-align: left;
    font-size: 10px;
    color: {TEXT_MUT};
    font-weight: 700;
    letter-spacing: 1.5px;
    border-bottom: 1px solid {BORDER2};
    white-space: nowrap;
  }}
  .pitch-table tbody td {{
    padding: 12px 16px;
    vertical-align: middle;
    border-bottom: 1px solid {BORDER};
  }}
  .pitch-table tbody tr:last-child td {{ border-bottom: none; }}
  .pitch-table tbody tr:nth-child(even) {{ background: rgba(255,255,255,0.025); }}
  .pitch-table tbody tr:hover {{ background: rgba(164,130,184,0.06); }}
  .col-num   {{ width: 48px; }}
  .col-seq   {{ width: 200px; }}
  .col-fat   {{ width: 160px; }}
  .col-vid   {{ width: 130px; }}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("page", "dashboard"), ("sel_player", None), ("all_sessions", {}), ("last_status", "idle")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── FAKE DATA ─────────────────────────────────────────────────────────────────
random.seed(42)
FATIGUE_LEVELS = ["Low", "Moderate", "High"]
FATIGUE_COLORS = {
    "Low":      ("#22c55e", "#14532d"),
    "Moderate": ("#f59e0b", "#451a03"),
    "High":     ("#ef4444", "#450a0a"),
}
SEQ_GREAT = "Great Sequencing"
SEQ_ARM   = "Arm Dominant"

def _fake_pitches(n, seed=0):
    random.seed(seed)
    pitches = []
    for i in range(n):
        seq   = SEQ_GREAT if random.random() > 0.38 else SEQ_ARM
        fat_w = [0.45, 0.35, 0.20] if i < n // 2 else [0.20, 0.35, 0.45]
        fat   = random.choices(FATIGUE_LEVELS, weights=fat_w)[0]
        pitches.append({"seq": seq, "fatigue": fat, "video": f"pitch_{i+1:02d}.mp4"})
    return pitches

def _fake_feedback(player):
    feedback_map = {
        "Jordan": [
            "<strong>Strong rotational power</strong> — hip separation averaging 0.07s across session, top-tier for this group.",
            "Velocity dipped after pitch 6; suggests <strong>aerobic fatigue onset</strong> mid-session.",
            "2 arm-dominant pitches clustered at the end — <strong>core fatigue</strong> likely reducing hip drive efficiency.",
            "<strong>Focus drill:</strong> Hip-wall drill with resistance band to maintain separation under fatigue.",
            "Overall excellent session — recommend 36h rest before next high-intensity throw.",
        ],
        "Priya": [
            "<strong>Hip-shoulder separation</strong> solid on 5 of 8 pitches — best consistency on the team this week.",
            "Moderate fatigue onset detected around pitch 5 — stride length visibly shorter on later pitches.",
            "Arm-dominant pattern emerging late-session — classic sign of <strong>hip stabiliser fatigue</strong>.",
            "<strong>Focus area:</strong> Maintain stride consistency even when fatigued; practice slow-motion stride drills.",
            "Quality session overall — rest 48h before next high-intensity throwing.",
        ],
        "Morgan": [
            "<strong>Excellent sequencing</strong> on first 4 pitches — textbook hip-first delivery mechanics.",
            "High fatigue flags on pitches 7–9 suggest <strong>conditioning gap</strong>; consider shorter sessions for now.",
            "Arm-dominant pattern emerging late-session — classic sign of <strong>hip stabiliser fatigue</strong>.",
            "<strong>Focus drill:</strong> Single-leg balance squat series to build hip stabiliser endurance.",
            "Recommend adding a 10-min cool-down mobility routine post-session.",
        ],
        "Riley": [
            "<strong>Consistent delivery mechanics</strong> across all pitches — sequencing variance very low.",
            "Sequencing rate of 67% is solid; <strong>3 arm-dominant pitches</strong> all occurred after moderate-fatigue flags.",
            "Hip rotation timing slightly early on 2 pitches — may be <strong>over-rotating</strong> to compensate for fatigue.",
            "<strong>Focus area:</strong> Slow the hip initiation cue and focus on lead-leg block before rotation.",
            "Strong session. Ready for increased pitch count next session.",
        ],
        "Casey": [
            "<strong>Explosive hip drive</strong> on pitches 1–3 — outstanding kinetic chain activation.",
            "Significant fatigue escalation from pitch 5 onward; <strong>high fatigue</strong> on final 3 pitches.",
            "One of the steeper performance declines in the group — endurance is the limiting factor.",
            "<strong>Focus drill:</strong> Posterior chain conditioning (Romanian deadlifts, hip thrusts) for endurance.",
            "Excellent raw mechanics — fitness conditioning is the primary area to develop.",
        ],
    }
    return feedback_map.get(player, [
        "Session data analysed — biomechanical patterns within expected range.",
        "Continue monitoring fatigue indicators across sessions.",
        "Schedule follow-up session in 48h for comparative analysis.",
    ])

FAKE_SESSIONS = {
    "Jordan":      {"date": "2026-07-30", "session": 4, "pitches": _fake_pitches(9,  seed=1), "feedback": _fake_feedback("Jordan")},
    "Priya":    {"date": "2026-07-31", "session": 3, "pitches": _fake_pitches(8,  seed=2), "feedback": _fake_feedback("Priya")},
    "Morgan":   {"date": "2026-07-29", "session": 5, "pitches": _fake_pitches(10, seed=3), "feedback": _fake_feedback("Morgan")},
    "Riley": {"date": "2026-07-31", "session": 2, "pitches": _fake_pitches(9,  seed=4), "feedback": _fake_feedback("Riley")},
    "Casey": {"date": "2026-07-28", "session": 6, "pitches": _fake_pitches(11, seed=5), "feedback": _fake_feedback("Casey")},
}

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
AVATAR_GRADIENTS = {
    "Alex":     ("#df6d45", "#a82e0a"),
    "Jordan":      ("#a482b8", "#5e3080"),
    "Priya":    ("#22c55e", "#14532d"),
    "Morgan":   ("#f59e0b", "#7c3a00"),
    "Riley": ("#3b82f6", "#1e3a8a"),
    "Casey": ("#ec4899", "#7c1145"),
}
INITIALS = {
    "Alex": "WL", "Jordan": "SM", "Priya": "PV",
    "Morgan": "RC", "Riley": "TQ", "Casey": "JY",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def evk_status():
    try:
        return requests.get(f"{EVK_URL}/status", timeout=3).json()
    except:
        return {"status": "unreachable", "message": "Cannot reach EVK", "player": None}

def evk_results():
    try:
        r = requests.get(f"{EVK_URL}/results", timeout=10)
        return r.json() if r.ok else None
    except:
        return None

def ollama_generate(prompt):
    try:
        r = requests.post(OLLAMA_URL, json={"model": "gemma3", "prompt": prompt, "stream": False}, timeout=90)
        if r.ok:
            return r.json().get("response", "")
    except:
        pass
    return "Local Gemma model unavailable — ensure Ollama is running."

def seq_badge_html(seq):
    if seq == SEQ_GREAT:
        return (
            '<span style="display:inline-block;background:rgba(34,197,94,0.15);color:#22c55e;'
            'border:1px solid rgba(34,197,94,0.35);padding:4px 13px;border-radius:20px;'
            'font-size:11px;font-weight:700;white-space:nowrap;">&#10003; Great Sequencing</span>'
        )
    return (
        '<span style="display:inline-block;background:rgba(239,68,68,0.15);color:#ef4444;'
        'border:1px solid rgba(239,68,68,0.35);padding:4px 13px;border-radius:20px;'
        'font-size:11px;font-weight:700;white-space:nowrap;">&#10007; Arm Dominant</span>'
    )

def fatigue_badge_html(level):
    # Still used for both the fake players' "Fatigue" data and Alex's real
    # "Effort Trend" data — same Low/Moderate/High vocabulary, just a
    # different underlying signal for live data (see EFFORT_LEVELS note
    # below). Falls back to a neutral "—" badge if level is None/unknown
    # (e.g. a pitch the watch never got a matching reading for).
    if level not in FATIGUE_COLORS:
        return (
            f'<span style="display:inline-block;background:rgba(255,255,255,0.06);color:{TEXT_MUT};'
            f'border:1px solid rgba(255,255,255,0.1);padding:4px 13px;border-radius:20px;'
            f'font-size:11px;font-weight:700;white-space:nowrap;">&#8212; No data</span>'
        )
    c, _ = FATIGUE_COLORS[level]
    dots  = {"Low": "&#9679;", "Moderate": "&#9679;", "High": "&#9679;"}
    return (
        f'<span style="display:inline-block;background:rgba(255,255,255,0.06);color:{c};'
        f'border:1px solid rgba(255,255,255,0.1);padding:4px 13px;border-radius:20px;'
        f'font-size:11px;font-weight:700;white-space:nowrap;">{dots[level]} {level}</span>'
    )

def video_chip_html(video_name, video_url=None):
    if video_url:
        return (
            f'<a href="{video_url}" target="_blank" style="text-decoration:none;">'
            f'<span style="display:inline-block;background:rgba(164,130,184,0.12);color:{LAVENDER};'
            f'border:1px solid rgba(164,130,184,0.3);padding:4px 13px;border-radius:20px;'
            f'font-size:11px;font-weight:700;cursor:pointer;white-space:nowrap;" '
            f'title="Skeleton overlay: {video_name}">&#127910; View</span></a>'
        )
    # No URL available (fake/demo data) — non-clickable placeholder, same look
    return (
        f'<span style="display:inline-block;background:rgba(164,130,184,0.12);color:{LAVENDER};'
        f'border:1px solid rgba(164,130,184,0.3);padding:4px 13px;border-radius:20px;'
        f'font-size:11px;font-weight:700;cursor:pointer;white-space:nowrap;" '
        f'title="Skeleton overlay: {video_name}">&#127910; View</span>'
    )

# ── HEADER ────────────────────────────────────────────────────────────────────
def render_header():
    import base64
    logo_path = "logo.png"
    date_str  = time.strftime("%A, %B %d %Y")
    if os.path.isfile(logo_path):
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="height:70px;width:auto;filter:drop-shadow(0 0 16px rgba(164,130,184,0.6));" />'
        )
    else:
        logo_html = (
            '<span style="font-size:60px;line-height:1;'
            'filter:drop-shadow(0 0 20px rgba(223,109,69,0.65));">&#128009;</span>'
        )
    st.markdown(f"""
    <div style="
        display:flex;align-items:center;justify-content:space-between;
        padding:26px 0 22px 0;
        border-bottom:1px solid {BORDER2};
        margin-bottom:38px;
    ">
      <div style="display:flex;align-items:center;gap:22px;">
        {logo_html}
        <div>
          <div style="
            font-size:46px;font-weight:900;letter-spacing:-2px;
            background:linear-gradient(100deg,{WHITE} 0%,{LAVENDER} 55%,{ORANGE} 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;line-height:1.05;margin:0;
          ">DragonPitch</div>
          <div style="
            font-size:11px;font-weight:700;letter-spacing:4px;
            color:{TEXT_MUT};text-transform:uppercase;margin-top:5px;
          ">AI Pitching Coach</div>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:13px;color:{TEXT_SOFT};font-weight:500;">{date_str}</div>
        <div style="
          display:inline-block;margin-top:8px;
          background:linear-gradient(90deg,rgba(164,130,184,0.18),rgba(223,109,69,0.18));
          border:1px solid rgba(164,130,184,0.3);border-radius:20px;
          padding:4px 14px;font-size:10px;font-weight:700;
          letter-spacing:1.5px;color:{LAVENDER};text-transform:uppercase;
        ">&#9889; Powered by Qualcomm AI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: DASHBOARD ───────────────────────────────────────────────────────────
def page_dashboard():
    render_header()

    col_title, col_refresh, col_feedback = st.columns([4, 1, 1.4])
    with col_title:
        st.markdown(f"""
        <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:24px;">
          <div style="font-size:22px;font-weight:800;letter-spacing:-0.5px;color:{WHITE};">Player Roster</div>
          <div style="font-size:12px;color:{TEXT_MUT};font-weight:600;letter-spacing:1px;">— {len(ALL_PLAYERS)} ATHLETES</div>
        </div>
        """, unsafe_allow_html=True)
    with col_refresh:
        if st.button("🔄 Refresh", key="refresh_dashboard", use_container_width=True):
            st.rerun()
    with col_feedback:
        if st.button("🤖 Give Team Feedback", key="feedback_dashboard", use_container_width=True):
            with st.spinner("Analysing with local Gemma..."):
                prompt = _build_team_prompt()
                st.session_state.team_ai_feedback = ollama_generate(prompt)
            st.rerun()

    # Player grid
    cols = st.columns(3, gap="medium")
    for idx, player in enumerate(ALL_PLAYERS):
        grad_a, grad_b = AVATAR_GRADIENTS[player]
        initials       = INITIALS[player]
        if player == "Alex":
            sessions = st.session_state.all_sessions.get("Alex", [])
            total_p  = sum(len(s["videos"]) for s in sessions)
            good_s   = sum(1 for s in sessions for v in s["videos"] if v.get("sequencing", {}).get("verdict"))
            seq_rate = int(good_s / total_p * 100) if total_p else 0
            last_date = sessions[-1]["timestamp"][:10] if sessions else "No sessions yet"
            is_live   = True
        else:
            data      = FAKE_SESSIONS[player]
            pitches   = data["pitches"]
            total_p   = len(pitches)
            good_s    = sum(1 for p in pitches if p["seq"] == SEQ_GREAT)
            seq_rate  = int(good_s / total_p * 100) if total_p else 0
            last_date = data["date"]
            is_live   = False

        arm_dom  = total_p - good_s
        live_dot = (
            f'<span style="color:#22c55e;font-size:9px;vertical-align:middle;">&#9679;</span>'
            f' <span style="color:{TEXT_MUT};font-size:10px;font-weight:700;letter-spacing:1px;">LIVE DATA</span>'
            if is_live else
            f'<span style="color:{TEXT_MUT};font-size:10px;font-weight:600;letter-spacing:0.5px;">Last: {last_date}</span>'
        )

        with cols[idx % 3]:
            st.markdown(f"""
            <div class="player-card" style="
                background:linear-gradient(160deg,{CARD_BG2} 0%,{CARD_BG} 100%);
                border-radius:18px;border:1px solid {BORDER2};
                padding:22px 18px 18px 18px;margin-bottom:4px;
                cursor:pointer;box-shadow:0 4px 24px rgba(0,0,0,0.35);
            ">
              <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
                <div style="
                  width:52px;height:52px;border-radius:50%;
                  background:linear-gradient(135deg,{grad_a},{grad_b});
                  display:flex;align-items:center;justify-content:center;
                  font-size:17px;font-weight:900;color:{WHITE};
                  flex-shrink:0;box-shadow:0 4px 16px rgba(0,0,0,0.4);
                ">{initials}</div>
                <div style="flex:1;min-width:0;">
                  <div style="font-size:18px;font-weight:800;color:{WHITE};letter-spacing:-0.3px;line-height:1.2;">{player}</div>
                  <div style="margin-top:3px;">{live_dot}</div>
                </div>
              </div>
              <div style="height:1px;background:linear-gradient(90deg,{BORDER2},transparent);margin-bottom:14px;"></div>
              <div style="display:flex;justify-content:space-between;align-items:flex-end;">
                <div style="text-align:center;">
                  <div style="font-size:24px;font-weight:900;color:{ORANGE};line-height:1;">{total_p}</div>
                  <div style="font-size:9.5px;color:{TEXT_MUT};font-weight:700;letter-spacing:1.2px;margin-top:3px;">PITCHES</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:24px;font-weight:900;color:{LAVENDER};line-height:1;">{seq_rate}%</div>
                  <div style="font-size:9.5px;color:{TEXT_MUT};font-weight:700;letter-spacing:1.2px;margin-top:3px;">SEQ RATE</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:24px;font-weight:900;color:#22c55e;line-height:1;">{good_s}</div>
                  <div style="font-size:9.5px;color:{TEXT_MUT};font-weight:700;letter-spacing:1.2px;margin-top:3px;">GREAT SEQ</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:24px;font-weight:900;color:#ef4444;line-height:1;">{arm_dom}</div>
                  <div style="font-size:9.5px;color:{TEXT_MUT};font-weight:700;letter-spacing:1.2px;margin-top:3px;">ARM DOM</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"View {player}", key=f"view_{player}", use_container_width=True):
                st.session_state.sel_player = player
                st.session_state.page = "profile"
                st.rerun()

    # ── AI Team Insight ────────────────────────────────────────────────────────
    stored_team_feedback = st.session_state.get("team_ai_feedback")
    if stored_team_feedback:
        team_inner_html = f'<div style="font-size:13px;line-height:1.8;color:#d4c4b0;">{stored_team_feedback}</div>'
    else:
        team_inner_html = (
            f'<div style="font-size:13px;color:{TEXT_MUT};text-align:center;padding:10px 0;">'
            f'Click &#129302; Give Team Feedback above to generate a live team analysis from real roster data.</div>'
        )

    st.markdown(f"""
    <div style="margin-top:48px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
      <div style="font-size:22px;font-weight:800;letter-spacing:-0.5px;color:{WHITE};">Team AI Insight</div>
    </div>
    <div style="
        background:linear-gradient(135deg,{CARD_BG2} 0%,{CARD_BG} 100%);
        border-radius:18px;border:1px solid {BORDER2};border-left:3px solid {LAVENDER};
        padding:26px 28px;box-shadow:0 4px 24px rgba(0,0,0,0.3);
    ">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
        <span style="font-size:20px;">&#129302;</span>
        <div style="font-size:14px;font-weight:700;color:{WHITE};">Group Performance Overview</div>
        <div style="margin-left:auto;font-size:11px;color:{TEXT_MUT};font-style:italic;">Session: {time.strftime('%Y-%m-%d')}</div>
      </div>
      <div style="height:1px;background:linear-gradient(90deg,{BORDER2},transparent);margin-bottom:18px;"></div>
      {team_inner_html}
    </div>
    """, unsafe_allow_html=True)

# ── PITCH TABLE ───────────────────────────────────────────────────────────────
def _pitch_row_html(num, seq, fatigue, video_name, video_url=None):
    row_bg = "rgba(255,255,255,0.025)" if num % 2 == 0 else "transparent"
    return f"""
    <tr style="background:{row_bg};">
      <td class="col-num"  style="padding:12px 16px;color:{TEXT_MUT};font-size:12px;font-weight:600;">{num}</td>
      <td class="col-seq"  style="padding:12px 16px;">{seq_badge_html(seq)}</td>
      <td class="col-fat"  style="padding:12px 16px;">{fatigue_badge_html(fatigue)}</td>
      <td class="col-vid"  style="padding:12px 16px;">{video_chip_html(video_name, video_url)}</td>
    </tr>"""

def _render_pitch_table(rows_html, fatigue_col_label="FATIGUE"):
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,{CARD_BG2},{CARD_BG});
        border-radius:16px;border:1px solid {BORDER2};
        overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.3);
    ">
      <div class="pitch-scroll-wrap">
        <table class="pitch-table">
          <thead>
            <tr>
              <th class="col-num">#</th>
              <th class="col-seq">SEQUENCING</th>
              <th class="col-fat">{fatigue_col_label}</th>
              <th class="col-vid">SKELETON VIDEO</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── AI FEEDBACK BOX — single st.markdown call, no split HTML ─────────────────
def _build_team_prompt():
    lines = []
    for player in ALL_PLAYERS:
        if player == "Alex":
            sessions = st.session_state.all_sessions.get("Alex", [])
            pitches = [v for s in sessions for v in s["videos"]]
            total_p = len(pitches)
            good_s = sum(1 for v in pitches if v.get("sequencing", {}).get("verdict"))
            note = "live EVK data"
        else:
            data = FAKE_SESSIONS[player]
            pitches = data["pitches"]
            total_p = len(pitches)
            good_s = sum(1 for p in pitches if p["seq"] == SEQ_GREAT)
            note = f"session on {data['date']}"
        arm_dom = total_p - good_s
        seq_rate = int(good_s / total_p * 100) if total_p else 0
        lines.append(f"  {player} ({note}): {total_p} pitches, {seq_rate}% sequencing rate "
                     f"({good_s} great sequencing / {arm_dom} arm-dominant)")

    roster_detail = "\n".join(lines)

    return (
        "You are an expert baseball pitching coach giving a team-wide summary to a coaching "
        "staff. Here is each pitcher's most recent session data:\n\n"
        f"{roster_detail}\n\n"
        "Based on this real data, provide: (1) 2-3 bullets on overall team sequencing health, "
        "(2) any players who stand out (best mechanics, or needing attention), and "
        "(3) one recommended team drill for this week. Keep it concise and coach-friendly."
    )


def _render_onset_timing_chart(pitches):
    """Builds a horizontal timeline chart: one row per pitch, showing when
    hips vs shoulders started rotating (seconds into that pitch's capture
    clip). Green connector = good sequencing (hips led), red = arm-dominant
    (shoulders led). Returns None if no pitch has usable onset data, so the
    caller can skip rendering entirely rather than showing an empty chart."""
    valid_pitches = [
        p for p in pitches
        if p.get("sequencing", {}).get("hip_onset") is not None
        and p.get("sequencing", {}).get("shoulder_onset") is not None
    ]
    if not valid_pitches:
        return None

    fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(valid_pitches))))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for i, p in enumerate(valid_pitches):
        seq = p["sequencing"]
        hip = seq["hip_onset"]
        shoulder = seq["shoulder_onset"]
        verdict = seq.get("verdict")
        color = "#22c55e" if verdict else "#ef4444"

        y = len(valid_pitches) - i  # pitch 1 at top
        ax.plot([hip, shoulder], [y, y], color=color, linewidth=2, zorder=1)
        ax.scatter([hip], [y], color="#a482b8", s=70, zorder=2, label="Hip onset" if i == 0 else None)
        ax.scatter([shoulder], [y], color="#df6d45", s=70, zorder=2, label="Shoulder onset" if i == 0 else None)

    ax.set_yticks([len(valid_pitches) - i for i in range(len(valid_pitches))])
    ax.set_yticklabels([f"Pitch {i+1}" for i in range(len(valid_pitches))], color=TEXT_SOFT)
    ax.set_xlabel("Time into capture clip (seconds)", color=TEXT_SOFT)
    ax.tick_params(colors=TEXT_MUT)
    ax.set_title("Hip vs. Shoulder Rotation Onset", color=WHITE, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, facecolor=CARD_BG, edgecolor=BORDER2, labelcolor=TEXT_SOFT)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(BORDER2)
    fig.tight_layout()
    return fig


def _build_grounded_prompt(pitches):
    if not pitches:
        return (
            "You are an expert baseball pitching coach. No pitch data is available yet "
            "for this session. Say briefly that you're ready to analyse once pitches come in."
        )

    n = len(pitches)
    good = sum(1 for p in pitches if p.get("sequencing", {}).get("verdict") is True)
    arm_dom = sum(1 for p in pitches if p.get("sequencing", {}).get("verdict") is False)

    lines = []
    for i, p in enumerate(pitches, 1):
        verdict = p.get("sequencing", {}).get("verdict")
        seq_text = "great sequencing (hips leading)" if verdict else "arm-dominant (shoulders leading)" if verdict is False else "unscored"
        gap = p.get("sequencing", {}).get("gap_seconds")
        gap_text = f", gap={gap:.3f}s" if isinstance(gap, (int, float)) else ""
        effort = p.get("effort_level") or "no watch data"
        lines.append(f"  Pitch {i}: {seq_text}{gap_text}, effort trend: {effort}")

    pitch_detail = "\n".join(lines)

    return (
        "You are an expert baseball pitching coach reviewing a real bullpen session for a "
        "pitcher named Alex, captured live by a camera+pose-tracking system.\n\n"
        f"Session summary: {n} pitches thrown, {good} scored as great sequencing "
        f"(hips rotating before shoulders), {arm_dom} scored as arm-dominant "
        "(shoulders rotating too early, a common fatigue/mechanics red flag).\n\n"
        f"Per-pitch detail:\n{pitch_detail}\n\n"
        "IMPORTANT — what 'effort trend' means: it comes from a wrist-worn watch's "
        "accelerometer, measuring each throw's peak arm acceleration relative to this same "
        "pitcher's own first few throws in this session (not an absolute or clinical measure, "
        "and not comparable across different pitchers). 'Low' means comparable effort to their "
        "early throws; 'Moderate' or 'High' means a notable drop in peak arm acceleration versus "
        "their own baseline, which can suggest arm fatigue setting in. When you mention effort "
        "trend, describe it this way — as a relative within-session arm-effort signal — rather "
        "than implying it's a precise or medically validated fatigue score.\n\n"
        "Based on this actual data, give 3-5 specific, actionable coaching bullet points. "
        "Reference specific pitch numbers where relevant. Keep it concise and practical."
    )


def _render_ai_feedback_box(player, is_live, feedback, live_pitches=None):
    if is_live:
        stored = st.session_state.get("will_ai_feedback")
        if stored:
            inner_html = f'<div style="font-size:13px;line-height:1.8;color:#d4c4b0;">{stored}</div>'
        else:
            inner_html = (
                f'<div style="font-size:13px;color:{TEXT_MUT};text-align:center;padding:10px 0;">'
                f'Click &#129302; Give Feedback above to generate AI coaching analysis from Alex\'s live session data.</div>'
            )
    else:
        bullets = "".join([
            f'<li style="color:#d4c4b0;font-size:13px;line-height:1.7;margin-bottom:8px;">{b}</li>'
            for b in feedback
        ])
        inner_html = f'<ul style="margin:0;padding-left:18px;">{bullets}</ul>'

    st.markdown(f"""
    <div style="margin-top:28px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <div style="font-size:11px;color:{TEXT_MUT};font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">
          &#129302; AI Coach Feedback &mdash; {time.strftime('%Y-%m-%d')} &mdash; {
            ('Live Session' if is_live else ('Session ' + str(FAKE_SESSIONS[player]['session']) if player in FAKE_SESSIONS else 'Session'))
          }
        </div>
      </div>
      <div style="
          background:linear-gradient(135deg,{CARD_BG2} 0%,{CARD_BG} 100%);
          border-radius:18px;border:1px solid {BORDER2};border-left:3px solid {LAVENDER};
          padding:24px 26px;box-shadow:0 4px 24px rgba(0,0,0,0.3);
      ">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
          <span style="font-size:18px;">&#129302;</span>
          <div style="font-size:14px;font-weight:700;color:{WHITE};">AI Coach Feedback</div>
          <div style="
            margin-left:auto;background:rgba(164,130,184,0.15);
            border:1px solid rgba(164,130,184,0.3);border-radius:20px;
            padding:2px 10px;font-size:10px;font-weight:700;color:{LAVENDER};letter-spacing:1.5px;
          ">GEMMA &middot; LOCAL</div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,{BORDER2},transparent);margin-bottom:16px;"></div>
        {inner_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: PROFILE ─────────────────────────────────────────────────────────────
def page_profile():
    player         = st.session_state.sel_player
    grad_a, grad_b = AVATAR_GRADIENTS[player]
    initials       = INITIALS[player]

    # ── Stats ─────────────────────────────────────────────────────────────────
    # Computed BEFORE the refresh button below, since refreshing needs
    # all_pitches (Alex's real data) available to rebuild the AI prompt.
    if player == "Alex":
        sessions    = st.session_state.all_sessions.get("Alex", [])
        all_pitches = [v for s in sessions for v in s["videos"]]
        total_p     = len(all_pitches)
        good_s      = sum(1 for v in all_pitches if v.get("sequencing", {}).get("verdict"))
        arm_dom     = total_p - good_s
        seq_rate    = int(good_s / total_p * 100) if total_p else 0
        last_date   = sessions[-1]["timestamp"][:10] if sessions else "—"
        last_session_str = f"Session {len(sessions)}" if sessions else "No sessions yet"
        is_live     = True
    else:
        data     = FAKE_SESSIONS[player]
        pitches  = data["pitches"]
        total_p  = len(pitches)
        good_s   = sum(1 for p in pitches if p["seq"] == SEQ_GREAT)
        arm_dom  = total_p - good_s
        seq_rate = int(good_s / total_p * 100) if total_p else 0
        last_date        = data["date"]
        last_session_str = f"Session {data['session']}"
        is_live  = False

    if player == "Alex":
        col_back, col_refresh, col_feedback = st.columns([4, 1, 1.4])
    else:
        col_back, col_refresh = st.columns([5, 1])
        col_feedback = None

    with col_back:
        if st.button("← Back to Roster", key="back_btn"):
            st.session_state.page = "dashboard"
            st.rerun()
    with col_refresh:
        if st.button("🔄 Refresh", key="refresh_profile", use_container_width=True):
            st.rerun()
    if col_feedback is not None:
        with col_feedback:
            if st.button("🤖 Give Feedback", key="feedback_profile", use_container_width=True):
                with st.spinner("Analysing with local Gemma..."):
                    prompt = _build_grounded_prompt(all_pitches)
                    st.session_state.will_ai_feedback = ollama_generate(prompt)
                st.rerun()

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # ── Player header card ────────────────────────────────────────────────────
    c_name, c_s1, c_s2, c_s3, c_s4 = st.columns([3, 1, 1, 1, 1])
    with c_name:
        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,{CARD_BG2},{CARD_BG});
            border-radius:16px;border:1px solid {BORDER2};
            padding:20px 22px;display:flex;align-items:center;gap:16px;
            box-shadow:0 4px 20px rgba(0,0,0,0.3);
        ">
          <div style="
            width:58px;height:58px;border-radius:50%;
            background:linear-gradient(135deg,{grad_a},{grad_b});
            display:flex;align-items:center;justify-content:center;
            font-size:19px;font-weight:900;color:{WHITE};
            box-shadow:0 4px 16px rgba(0,0,0,0.4);flex-shrink:0;
          ">{initials}</div>
          <div>
            <div style="font-size:22px;font-weight:900;color:{WHITE};letter-spacing:-0.5px;">{player}</div>
            <div style="font-size:11px;color:{TEXT_MUT};font-weight:600;margin-top:2px;">
              Pitcher &middot; DragonPitch Academy &middot; {"&#9899; Live EVK Data" if is_live else f"Last session: {last_date}"}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    for col, (val, label, color) in zip(
        [c_s1, c_s2, c_s3, c_s4],
        [
            (str(total_p),   "TOTAL PITCHES", ORANGE),
            (f"{seq_rate}%", "SEQ RATE",      LAVENDER),
            (str(good_s),    "GREAT SEQ",     "#22c55e"),
            (str(arm_dom),   "ARM DOM",        "#ef4444"),
        ]
    ):
        with col:
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,{CARD_BG2},{CARD_BG});
                border-radius:16px;border:1px solid {BORDER2};
                padding:18px 14px;text-align:center;
                box-shadow:0 4px 20px rgba(0,0,0,0.3);
                display:flex;flex-direction:column;justify-content:center;align-items:center;
            ">
              <div style="font-size:28px;font-weight:900;color:{color};line-height:1;">{val}</div>
              <div style="font-size:9.5px;color:{TEXT_MUT};font-weight:700;letter-spacing:1.2px;margin-top:6px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:26px;'></div>", unsafe_allow_html=True)

    # ── Session label + pitch table ───────────────────────────────────────────
    if player == "Alex":
        if total_p == 0:
            st.markdown(f"""
            <div style="font-size:14px;font-weight:700;color:{TEXT_MUT};
                        letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">
                &#128197; Live Session &mdash; Alex
            </div>
            <div style="
                background:{CARD_BG2};border-radius:14px;
                border:1px dashed {BORDER2};padding:36px;text-align:center;
            ">
              <div style="font-size:30px;margin-bottom:10px;">&#127919;</div>
              <div style="font-size:14px;font-weight:700;color:{WHITE};margin-bottom:6px;">No live data yet</div>
              <div style="font-size:12px;color:{TEXT_MUT};">Start a session from the Pixel Watch to see Alex's pitch data here in real time.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
              <div style="font-size:14px;font-weight:700;color:{TEXT_MUT};letter-spacing:1.5px;text-transform:uppercase;">
                &#128197; {last_date} &mdash; {last_session_str}
              </div>
              <div style="font-size:12px;color:{TEXT_MUT};">{total_p} pitches &middot; {good_s} great seq &middot; {arm_dom} arm dominant</div>
            </div>
            """, unsafe_allow_html=True)
            rows_html = ""
            for vi, v in enumerate(all_pitches):
                verdict   = v.get("sequencing", {}).get("verdict", False)
                seq_label = SEQ_GREAT if verdict else SEQ_ARM
                # Real per-pitch effort trend from the watch (Low/Moderate/High),
                # falling back to None (rendered as "— No data") if this pitch
                # had no matching watch reading — e.g. watch wasn't worn/connected.
                effort_level = v.get("effort_level")
                video_url = f"{EVK_URL}/video/{v['video_path']}" if v.get("video_path") else None
                rows_html += _pitch_row_html(vi + 1, seq_label, effort_level, v.get("video_name", f"pitch_{vi+1:02d}.mp4"), video_url)
            _render_pitch_table(rows_html, fatigue_col_label="EFFORT TREND")
    else:
        data    = FAKE_SESSIONS[player]
        pitches = data["pitches"]
        n       = len(pitches)
        good_n  = sum(1 for p in pitches if p["seq"] == SEQ_GREAT)
        arm_n   = n - good_n
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
          <div style="font-size:14px;font-weight:700;color:{TEXT_MUT};letter-spacing:1.5px;text-transform:uppercase;">
            &#128197; {data['date']} &mdash; {last_session_str}
          </div>
          <div style="font-size:12px;color:{TEXT_MUT};">{n} pitches &middot; {good_n} great seq &middot; {arm_n} arm dominant</div>
        </div>
        """, unsafe_allow_html=True)
        rows_html = ""
        for i, p in enumerate(pitches):
            rows_html += _pitch_row_html(i + 1, p["seq"], p["fatigue"], p["video"])
        _render_pitch_table(rows_html, fatigue_col_label="FATIGUE")

    # ── AI feedback ───────────────────────────────────────────────────────────
    _render_ai_feedback_box(
        player,
        is_live=is_live,
        feedback=[] if player == "Alex" else FAKE_SESSIONS[player]["feedback"],
        live_pitches=all_pitches if player == "Alex" else None,
    )

# ── EVK POLLING ───────────────────────────────────────────────────────────────
def check_and_archive():
    # Session-ID based, not status-transition based: this works no matter
    # when Streamlit happens to poll (a missed transition — e.g. because
    # the tab sat idle through an entire session with no clicks — used to
    # mean the session was silently never archived at all).
    if "archived_session_ids" not in st.session_state:
        st.session_state.archived_session_ids = set()

    data = evk_results()
    if not data or not data.get("videos"):
        return
    session_id = data.get("session_id")
    if session_id is None or session_id in st.session_state.archived_session_ids:
        return  # already archived, or EVK hasn't sent a session_id (older /results shape)

    # Falls back to "Alex" — the only live-data player slot in this app by
    # design (everyone else is fake demo data). This matters when testing
    # sessions started directly on the EVK (terminal/keyboard), which never
    # goes through /session/start and so never sets a real player name.
    target = data.get("player") or "Alex"
    st.session_state.all_sessions.setdefault(target, []).append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "videos":    data["videos"],
    })
    st.session_state.archived_session_ids.add(session_id)

def poll_evk():
    s = evk_status()
    check_and_archive()
    return s

# ── MAIN ──────────────────────────────────────────────────────────────────────
poll_evk()
if   st.session_state.page == "dashboard": page_dashboard()
elif st.session_state.page == "profile":   page_profile()
