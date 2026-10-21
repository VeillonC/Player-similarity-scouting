import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Player Similarity Scouting", layout="wide")

PLOTLY_TEMPLATE = "plotly_white"
ACCENT_1 = "#0B6E4F"
ACCENT_2 = "#D9A440"
ACCENT_SEQUENCE = ["#0B6E4F", "#D9A440", "#9C3D54", "#3C6E91", "#6B4FA0"]
CLUSTER_COLORS = {0: "#0B6E4F", 1: "#D9A440", 2: "#9C3D54", 3: "#3C6E91"}
BG = "#F6F4EF"
TEXT = "#16211C"
MUTED = "#6B7268"
FONT_MONO = "'IBM Plex Mono', monospace"
LINKEDIN_URL = "https://www.linkedin.com/in/veilloncamille/"
GITHUB_URL = "https://github.com/VeillonC/Player-similarity-scouting"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Hanken+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] { font-family: 'Hanken Grotesk', sans-serif; font-size: 17px; color: #16211C; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: 0.01em; color: #16211C; }

.block-container { padding-top: 1.2rem; padding-left: 2.4rem; padding-right: 2.4rem; padding-bottom: 1rem; max-width: 100%; }

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border-top: 4px solid #0B6E4F;
    border-radius: 4px;
    padding: 0.8rem 0.9rem;
}
[data-testid="column"]:nth-of-type(even) [data-testid="stMetric"] { border-top-color: #D9A440; }

[data-testid="stMetricLabel"] { color: #6B7268; white-space: normal; overflow: visible; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.02em; }
[data-testid="stMetricValue"] { color: #16211C; font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.5rem; }

[data-testid="column"] { padding: 0.35rem; }

.stTabs [data-baseweb="tab-list"],
.stTabs [role="tablist"] {
    gap: 0.6rem !important;
    border-bottom: 3px solid #16211C !important;
    padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs button[role="tab"],
.stTabs [data-testid="stTab"],
.stTabs [data-baseweb="tab"] {
    box-sizing: border-box !important;
    height: auto !important;
    min-height: 0 !important;
    color: #6B7268 !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    padding: 0.7rem 1.3rem !important;
    margin: 0 0 -3px 0 !important;
    border-radius: 6px 6px 0 0 !important;
    background-color: transparent !important;
    transition: background-color 0.15s ease, color 0.15s ease !important;
}
.stTabs button[role="tab"] p,
.stTabs [data-testid="stTab"] p,
.stTabs [data-baseweb="tab"] p { padding: 0 !important; margin: 0 !important; }
.stTabs button[role="tab"]:hover,
.stTabs [data-testid="stTab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    color: #16211C !important;
    background-color: #EAE7DD !important;
}
.stTabs button[aria-selected="true"],
.stTabs [data-testid="stTab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    background-color: #0B6E4F !important;
}
.stTabs button[aria-selected="true"]:hover,
.stTabs [aria-selected="true"]:hover { background-color: #0B6E4F !important; color: #FFFFFF !important; }

[data-testid="stDataFrame"] { border: 1px solid #DEDACB; border-radius: 4px; font-size: 0.95rem; }

[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #DEDACB;
    border-radius: 6px;
    margin-top: 1.2rem;
}

.app-header {
    position: relative;
    overflow: hidden;
    border-bottom: 3px solid #16211C;
    background-image: repeating-linear-gradient(90deg, rgba(11,110,79,0.05) 0px, rgba(11,110,79,0.05) 1px, transparent 1px, transparent 46px);
    padding: 1.5rem 2.4rem 1.3rem;
    margin: -1.2rem -2.4rem 1.8rem;
}
.app-header::before {
    content: "";
    position: absolute; right: -70px; top: -70px;
    width: 220px; height: 220px; border-radius: 50%;
    border: 2px solid rgba(11,110,79,0.16);
    pointer-events: none;
}
.app-header::after {
    content: "";
    position: absolute; right: 60px; top: 30px;
    width: 90px; height: 90px; border-radius: 50%;
    border: 2px solid rgba(217,164,64,0.3);
    pointer-events: none;
}
.app-byline { font-size: 0.82rem; color: #6B7268; margin-top: 0.35rem; }
.app-byline a { color: #0B6E4F; font-weight: 600; text-decoration: none; border-bottom: 1px solid rgba(11,110,79,0.4); }
.app-byline a:hover { border-bottom-color: #0B6E4F; }

.intro-box {
    background-color: #FFFFFF;
    border: 1px solid #DEDACB;
    border-left: 3px solid #0B6E4F;
    border-radius: 6px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.4rem;
}
.intro-step { color: #16211C; margin-bottom: 0.3rem; }
.intro-step b { color: #D9A440; }

.section-title { margin-top: 1.6rem; margin-bottom: 0.5rem; color: #16211C; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 0.02em; }
.section-subtitle { color: #6B7268; margin-bottom: 0.8rem; font-size: 0.82rem; }
.stat-group-label { font-size: 0.76rem; font-weight: 600; color: #6B7268; text-transform: uppercase; letter-spacing: 0.03em; margin: 0.9rem 0 0.5rem; }

.player-banner { border-bottom: 1px solid #DEDACB; padding-bottom: 1rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 1rem; }
.player-tag { display: inline-block; background-color: #FFFFFF; border: 1px solid #DEDACB; color: #6B7268; padding: 0.25rem 0.7rem; border-radius: 16px; font-size: 0.72rem; margin-right: 0.5rem; }
.player-tag-accent { display: inline-block; background-color: #D9A440; color: #16211C; font-weight: 600; padding: 0.25rem 0.7rem; border-radius: 16px; font-size: 0.72rem; margin-right: 0.5rem; font-family: 'IBM Plex Mono', monospace; }

.player-photo {
    width: 92px; height: 110px; border-radius: 10px; object-fit: cover; object-position: top center;
    flex-shrink: 0; border: 2px solid #FFFFFF; box-shadow: 0 0 0 2px #16211C;
}
.player-photo-fallback {
    width: 92px; height: 110px; border-radius: 10px; flex-shrink: 0;
    background-color: #0B6E4F; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.8rem;
    box-shadow: 0 0 0 2px #16211C;
}

.compare-table { width: 100%; border-collapse: collapse; margin-top: 0.4rem; }
.compare-table th { padding: 0.75rem; font-size: 0.85rem; border-bottom: 3px solid #16211C; }
.compare-table td { padding: 0.75rem; text-align: center; border-bottom: 1px solid #EAE7DD; font-family: 'IBM Plex Mono', monospace; }
.compare-table tr:nth-child(even) td { background-color: #FFFFFF; }

.app-footer {
    border-top: 1px solid #DEDACB; margin-top: 2.2rem; padding-top: 1rem;
    color: #6B7268; font-size: 0.75rem; text-align: center; line-height: 1.5;
}
.app-footer b { color: #16211C; }
.app-footer a { color: #0B6E4F; font-weight: 600; text-decoration: none; border-bottom: 1px solid rgba(11,110,79,0.4); }
.app-footer a:hover { border-bottom-color: #0B6E4F; }

@media (max-width: 768px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; padding-top: 0.8rem !important; }
    html, body, [class*="css"] { font-size: 15.4px; }
    .app-header { margin: -0.8rem -1rem 1.2rem; padding: 1.1rem 1.2rem 1rem; }
    .app-header h1 { font-size: 1.5rem !important; }
    .app-header::before, .app-header::after { display: none; }
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    [data-testid="stHorizontalBlock"] [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
    .stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {
        overflow-x: auto !important; flex-wrap: nowrap !important; gap: 0.3rem !important;
        -webkit-overflow-scrolling: touch; scrollbar-width: thin;
    }
    .stTabs button[role="tab"], .stTabs [data-testid="stTab"], .stTabs [data-baseweb="tab"] {
        padding: 0.45rem 0.7rem !important; font-size: 0.68rem !important; letter-spacing: 0.01em !important;
        white-space: nowrap;
    }
    .compare-table { display: block; overflow-x: auto; white-space: nowrap; }
    [data-testid="stDataFrame"] { font-size: 0.72rem; }
    .player-banner { flex-wrap: wrap; }
    /* Plotly charts: let them scroll horizontally rather than squeeze their own text unreadably small */
    [data-testid="stPlotlyChart"] { overflow-x: auto; }
    .js-plotly-plot .plotly .legend text { font-size: 10px !important; }
    .js-plotly-plot .plotly .ytick text, .js-plotly-plot .plotly .xtick text { font-size: 9px !important; }
}
</style>
""", unsafe_allow_html=True)

with open("data/player_profiles.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
similarity_matrix = data["similarity_matrix"]
radar_cols = data["radar_cols"]  # the PCA/clustering feature set, not necessarily what we display

# richer stat set for the radar chart and the comparison table, independent of what PCA was trained on
DISPLAY_RADAR_COLS = [c for c in [
    "goals_p90", "xg_p90", "shots_p90", "progressive_passes_p90", "progressive_carries_p90",
    "passes_completed_p90", "Ball Recovery_p90", "Interception_p90", "Pressure_p90",
    "Block_p90", "Clearance_p90",
] if c in df.columns]

raw_cols = [c for c in [
    "minutes_played", "shots", "goals", "xg", "passes_completed",
    "progressive_passes", "progressive_carries", "Ball Recovery",
    "Interception", "Pressure", "Block", "Clearance",
] if c in df.columns]

discipline_cols = ["total_cards", "yellow_cards", "red_cards", "cards_p90", "fouls_committed_p90", "fouls_won"]
excluded_from_pct = ["cards_p90", "fouls_committed_p90"]

HAS_PHOTOS = "image_url" in df.columns

STAT_GROUPS = [
    ("Attacking", ["goals", "shots", "xg"]),
    ("Creation & progression", ["progressive_passes", "progressive_carries", "passes_completed"]),
    ("Defensive work", ["Interception", "Ball Recovery", "Pressure"]),
]

metric_labels = {
    "minutes_played": "Minutes played", "shots": "Shots", "goals": "Goals", "xg": "xG",
    "passes_completed": "Passes completed", "progressive_passes": "Progressive passes",
    "progressive_carries": "Progressive carries", "Ball Recovery": "Ball recoveries",
    "Interception": "Interceptions", "Pressure": "Pressures",
    "Block": "Blocks", "Clearance": "Clearances",
}

metric_help = {
    "minutes_played": "Total minutes played across both tournaments (World Cup 2022 and Euro 2024).",
    "shots": "Total shots taken, on or off target.",
    "goals": "Goals scored.",
    "xg": "Expected Goals: the likelihood of each shot resulting in a goal, based on location, angle and situation. A simple way to judge chance quality, not just the final score.",
    "passes_completed": "Total completed passes.",
    "progressive_passes": "Passes that move the ball meaningfully closer to the opponent's goal.",
    "progressive_carries": "Ball carries that move the ball meaningfully closer to the opponent's goal.",
    "Ball Recovery": "Number of times the player recovered a loose ball for their team.",
    "Interception": "Opponent passes intercepted by the player.",
    "Pressure": "Times the player pressed an opponent in possession, forcing a rushed decision.",
    "Block": "Shots or passes blocked by the player's body.",
    "Clearance": "The ball kicked or headed away from danger, clear of the player's own goal.",
    "total_cards": "Combined yellow and red cards across both tournaments.",
    "red_cards": "Red cards, including a second yellow card in the same match.",
    "cards_p90": "Cards received per 90 minutes played, a rate that accounts for playing time.",
}

p90_labels = {
    "shots_p90": "Shots", "goals_p90": "Goals", "xg_p90": "xG",
    "passes_completed_p90": "Passes completed", "progressive_passes_p90": "Progressive passes",
    "progressive_carries_p90": "Progressive carries", "Ball Recovery_p90": "Ball recoveries",
    "Block_p90": "Blocks", "Clearance_p90": "Clearances",
    "Interception_p90": "Interceptions", "Pressure_p90": "Pressures",
    "fouls_committed_p90": "Fouls committed", "cards_p90": "Cards",
}

def p90_label(col):
    return p90_labels.get(col, col.replace("_p90", "").replace("_", " ").capitalize())

def ranking_label(col):
    # the raw column name is what the selectbox needs internally, this is only for display
    if col.endswith("_p90"):
        return f"{p90_label(col)} per 90"
    return metric_labels.get(col, col.replace("_", " ").capitalize())

def age_tag(row):
    # true whole years lived, not rounded: a player who turned 27 last month shouldn't show as 28
    if "age" in row.index and pd.notna(row["age"]):
        return f'<span class="player-tag-accent">{int(row["age"])} yo</span>'
    return ""

def player_photo_html(row, width=92, height=110):
    # photos come through as rectangular press/ID-style shots, not square -> a circle crop just
    # chops off the top of the head, a rounded rectangle is a much better fit
    url = row.get("image_url") if "image_url" in row.index else None
    if pd.notna(url) and url:
        return f'<img class="player-photo" src="{url}" style="width:{width}px;height:{height}px;" alt="{row["player_name"]}">'
    initials = "".join(w[0] for w in str(row["player_name"]).split()[:2]).upper()
    font_size = round(height * 0.33, 1)
    return f'<div class="player-photo-fallback" style="width:{width}px;height:{height}px;font-size:{font_size}px;">{initials}</div>'

def clean_html(html):
    return "\n".join(line.strip() for line in html.strip().split("\n"))


def find_similar_players(name, n):
    idx = df.index[df["player_name"] == name][0]
    scores = similarity_matrix[df.index.get_loc(idx)]
    similar_idx = scores.argsort()[::-1][1:n + 1]

    highlight_cols = [c for c in ["goals_p90", "xg_p90", "progressive_passes_p90"] if c in df.columns]
    cols_to_show = [c for c in ["player_name", "position", "team", "age"] if c in df.columns] + highlight_cols
    result = df.iloc[similar_idx][cols_to_show].copy()
    result.insert(1, "similarity", scores[similar_idx].round(3))

    if "age" in result.columns:
        result["age"] = result["age"].apply(lambda a: int(a) if pd.notna(a) else a)
    for c in highlight_cols:
        result[c] = result[c].round(2)

    result = result.rename(columns={
        "goals_p90": "Goals/90", "xg_p90": "xG/90", "progressive_passes_p90": "Prog. passes/90",
        "player_name": "Player", "position": "Position", "team": "Team", "age": "Age", "similarity": "Similarity",
    })
    return result


def plot_radar(players_to_plot, cols=None, order=None):
    cols = cols or DISPLAY_RADAR_COLS
    if order:
        players_to_plot = players_to_plot.set_index("player_name").loc[order].reset_index()
    radar_df = players_to_plot.set_index("player_name")[cols]
    radar_norm = (radar_df - df[cols].min()) / (df[cols].max() - df[cols].min())
    theta_labels = [p90_label(c) for c in cols]
    # short definition per axis, shown on hover so the jargon doesn't need a tooltip elsewhere on the page
    theta_defs = [metric_help.get(c.replace("_p90", ""), "") for c in cols]
    theta_defs_wrapped = theta_defs + [theta_defs[0]]

    fig = go.Figure()
    for i, (name, row) in enumerate(radar_norm.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=theta_labels + [theta_labels[0]],
            fill="toself", name=name, opacity=0.5,
            line=dict(color=ACCENT_SEQUENCE[i % len(ACCENT_SEQUENCE)]),
            customdata=theta_defs_wrapped,
            hovertemplate="<b>%{theta}</b><br>" + name + ": %{r:.0%} of the sample's range<br><i>%{customdata}</i><extra></extra>",
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG,
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#EAE7DD", tickfont=dict(size=10)),
            # smaller axis labels so they don't overlap once the chart is squeezed into a narrow phone screen
            angularaxis=dict(gridcolor="#EAE7DD", tickfont=dict(size=11)),
        ),
        showlegend=True, height=580, margin=dict(t=40, b=30, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        font=dict(color=TEXT),
    )
    return fig


def render_kpi_row(row, cols_list, columns):
    kpi_cols = columns(len(cols_list), gap="medium")
    for col, stat_name in zip(kpi_cols, cols_list):
        value = row[stat_name]
        col.metric(
            metric_labels.get(stat_name, stat_name),
            f"{value:.2f}" if stat_name == "xg" else f"{value:.0f}",
            help=metric_help.get(stat_name),
        )


def render_stat_groups(row):
    for title, cols_list in STAT_GROUPS:
        st.markdown(f'<div class="stat-group-label">{title}</div>', unsafe_allow_html=True)
        render_kpi_row(row, cols_list, st.columns)


def render_player_banner(row):
    minutes_fmt = f"{row['minutes_played']:.0f} min played" if "minutes_played" in row.index else ""
    st.markdown(clean_html(f"""
    <div class="player-banner">
        {player_photo_html(row)}
        <div>
            <h2 style="margin-bottom: 0.4rem;">{row['player_name']}</h2>
            <span class="player-tag">{row['position']}</span>
            <span class="player-tag">{row['team']}</span>
            <span class="player-tag">{minutes_fmt}</span>
            {age_tag(row)}
        </div>
    </div>
    """), unsafe_allow_html=True)


st.markdown(f"""
<div class="app-header">
    <h1 style="margin-bottom: 0.25rem; font-size: 2.2rem;">Player Similarity Scouting</h1>
    <p style="color: #6B7268; margin: 0; font-size: 0.9rem;">World Cup 2022 and Euro 2024 &middot; StatsBomb Open Data</p>
    <p class="app-byline">A portfolio project by <a href="{LINKEDIN_URL}" target="_blank">Camille Veillon</a></p>
</div>
""", unsafe_allow_html=True)

tab_search, tab_compare, tab_rank, tab_about = st.tabs(
    ["Find a player", "Compare players", "Top players", "About"]
)

with tab_search:
    st.markdown("""
    <div class="intro-box">
        <div class="intro-step"><b>1.</b> Pick a player below.</div>
        <div class="intro-step"><b>2.</b> See his stats and the players who play most like him.</div>
        <div class="intro-step"><b>3.</b> Click any similar player to pull up their own stats, or use the chart to compare playing styles visually.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Narrow down the player list (optional)", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3, gap="large")
        with col_f1:
            comp_filter = st.multiselect("Competition", sorted(df["competition"].unique()))
        with col_f2:
            team_filter = st.multiselect("Team", sorted(df["team"].dropna().unique()))
        with col_f3:
            pos_filter = st.multiselect("Position", sorted(df["position"].dropna().unique()))

    filtered = df.copy()
    if comp_filter:
        filtered = filtered[filtered["competition"].apply(lambda x: any(c in x for c in comp_filter))]
    if team_filter:
        filtered = filtered[filtered["team"].isin(team_filter)]
    if pos_filter:
        filtered = filtered[filtered["position"].isin(pos_filter)]

    if filtered.empty:
        st.warning("No player matches these filters")
    else:
        col_search, col_n = st.columns([3, 1], gap="large")
        with col_search:
            main_name = st.selectbox("Search for a player", sorted(filtered["player_name"].unique()))
        with col_n:
            n_similar = st.slider(
                "How many similar players", 3, 8, 5,
                help="How many of the closest statistical profiles to show.",
            )

        similar = find_similar_players(main_name, n_similar)
        main_row = df[df["player_name"] == main_name].iloc[0]

        if st.session_state.get("last_searched_player") != main_name:
            st.session_state.selected_similar_player = None
            st.session_state.last_searched_player = main_name

        render_player_banner(main_row)
        render_stat_groups(main_row)

        st.markdown('<div class="section-title">Players with a similar playing style</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Click "View profile" on a card to pull up that player\'s own stats below.</div>',
            unsafe_allow_html=True,
        )

        if "selected_similar_player" not in st.session_state:
            st.session_state.selected_similar_player = None

        similar_rows = [df[df["player_name"] == pname].iloc[0] for pname in similar["Player"]]
        card_cols = st.columns(len(similar_rows), gap="small")
        for col, srow, sim_score in zip(card_cols, similar_rows, similar["Similarity"]):
            with col:
                st.markdown(clean_html(f"""
                <div style="background:#FFFFFF; border:1px solid #DEDACB; border-radius:8px; padding:0.6rem;
                            text-align:center; display:flex; flex-direction:column; align-items:center; gap:0.4rem;">
                    {player_photo_html(srow, width=68, height=82)}
                    <div style="font-weight:700; font-size:0.8rem; line-height:1.2;">{srow['player_name']}</div>
                    <div style="font-size:0.68rem; color:#6B7268;">{srow['position']}</div>
                    <div style="font-family:'IBM Plex Mono', monospace; font-size:0.8rem; color:#0B6E4F; font-weight:700;">{sim_score * 100:.0f}%<span style="font-family:'Hanken Grotesk'; font-weight:500; color:#6B7268; font-size:0.66rem;"> similar</span></div>
                </div>
                """), unsafe_allow_html=True)
                if st.button("View profile", key=f"view_similar_{srow['player_name']}", use_container_width=True):
                    st.session_state.selected_similar_player = srow["player_name"]

        with st.expander("See the full stats table"):
            st.dataframe(similar, use_container_width=True, hide_index=True)

        clicked_name = st.session_state.selected_similar_player
        if clicked_name:
            clicked_row = df[df["player_name"] == clicked_name].iloc[0]
            st.markdown('<div class="section-title">Selected profile</div>', unsafe_allow_html=True)
            render_player_banner(clicked_row)
            render_stat_groups(clicked_row)

        st.markdown('<div class="section-title">Visual comparison</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">The bigger the shape on an axis, the stronger the player is on that '
            'aspect of the game, compared to the rest of the sample.</div>',
            unsafe_allow_html=True,
        )

        all_candidates = [main_name] + similar["Player"].tolist()
        selected_players = st.multiselect(
            "Players shown on the chart",
            options=all_candidates,
            default=all_candidates[:4],
        )

        if selected_players:
            ordered_selection = [p for p in all_candidates if p in selected_players]
            players_to_plot = df[df["player_name"].isin(selected_players)].drop_duplicates(subset="player_name")
            st.plotly_chart(plot_radar(players_to_plot, order=ordered_selection), use_container_width=True)
            st.caption(f"Comparing on {len(DISPLAY_RADAR_COLS)} per-90 metrics: " + ", ".join(p90_label(c) for c in DISPLAY_RADAR_COLS))
        else:
            st.info("Select at least one player above to display the chart")

        st.markdown('<div class="section-title">How does he compare to his position?</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-subtitle">Ranked against other {main_row["position_group"].lower()}s in the '
            'sample. A score of 90 means he is in the top 10% at that aspect of the game. The dashed line is the '
            'middle of the pack.</div>',
            unsafe_allow_html=True,
        )

        pct_cols = [c for c in df.columns if c.endswith("_p90_pct") and c.replace("_pct", "") not in excluded_from_pct]
        pct_values = main_row[pct_cols].sort_values(ascending=True)
        pct_defs = [metric_help.get(c.replace("_p90_pct", ""), "") for c in pct_values.index]

        fig_pct = go.Figure(go.Bar(
            x=pct_values.values,
            y=[p90_label(c.replace("_pct", "")) for c in pct_values.index],
            orientation="h", marker_color=ACCENT_1,
            customdata=pct_defs,
            hovertemplate="<b>%{y}</b><br>Percentile: %{x:.0f}<br><i>%{customdata}</i><extra></extra>",
        ))
        fig_pct.add_vline(x=50, line_dash="dash", line_color="#6B7268")
        fig_pct.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor="#FFFFFF",
            xaxis=dict(range=[0, 100], title="Percentile", gridcolor="#EAE7DD"),
            # automargin keeps the metric labels on the left from getting clipped on a narrow screen,
            # and scaling height with the bar count stops them from getting cramped on top of each other
            yaxis=dict(gridcolor="#EAE7DD", automargin=True),
            height=max(320, len(pct_values) * 34), margin=dict(t=20, b=20, l=10),
            font=dict(color=TEXT),
        )
        st.plotly_chart(fig_pct, use_container_width=True)

        strengths = pct_values.sort_values(ascending=False).head(3)
        weaknesses = pct_values.sort_values(ascending=True).head(3)

        def render_cards(items, color):
            cards_html = ""
            for col, val in items.items():
                label = p90_label(col.replace("_pct", ""))
                cards_html += f"""
                <div style="background-color:#FFFFFF; border:1px solid #DEDACB; border-left:4px solid {color};
                            padding:0.75rem 1.1rem; margin-bottom:0.6rem; border-radius:4px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#16211C;">{label}</span>
                    <span style="color:{color}; font-family:'IBM Plex Mono', monospace; font-weight:600; font-size:0.95rem;">
                        {val:.0f}<span style="font-size:0.65rem; color:#6B7268;"> pct</span>
                    </span>
                </div>
                """
            return cards_html

        col_strengths, col_weaknesses = st.columns(2, gap="large")
        with col_strengths:
            st.markdown("**What he does best**")
            st.markdown(clean_html(render_cards(strengths, ACCENT_1)), unsafe_allow_html=True)
        with col_weaknesses:
            st.markdown("**Where he lags behind**")
            st.markdown(clean_html(render_cards(weaknesses, ACCENT_2)), unsafe_allow_html=True)

        st.markdown('<div class="section-title">Discipline</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">A risk indicator, separate from the playing style shown above.</div>',
            unsafe_allow_html=True,
        )
        col_disc1, col_disc2, col_disc3 = st.columns(3, gap="medium")
        col_disc1.metric("Total cards", f"{main_row['total_cards']:.0f}", help=metric_help["total_cards"])
        col_disc2.metric("Red cards", f"{main_row['red_cards']:.0f}", help=metric_help["red_cards"])
        col_disc3.metric("Cards per 90", f"{main_row['cards_p90']:.2f}", help=metric_help["cards_p90"])

with tab_compare:
    st.markdown("""
    <div class="intro-box">
        <div class="intro-step">Add 2 to 5 players below to see them side by side, no matter how similar their styles are.</div>
    </div>
    """, unsafe_allow_html=True)

    all_players = sorted(df["player_name"].unique())

    def _default_pick(keyword):
        hits = df[df["player_name"].str.contains(keyword, case=False, na=False)]
        return hits["player_name"].iloc[0] if not hits.empty else None

    default_compare = [p for p in (_default_pick("Yamal"), _default_pick("Kane")) if p]
    if len(default_compare) < 2:
        default_compare += [p for p in all_players if p not in default_compare][: 2 - len(default_compare)]

    selected_compare = st.multiselect(
        "Players to compare",
        options=all_players,
        default=default_compare,
        max_selections=5,
        help="Pick between 2 and 5 players. Remove one by clicking the cross on its tag.",
    )

    if len(selected_compare) < 2:
        st.info("Add at least two players to start comparing")
    else:
        compare_rows = [df[df["player_name"] == p].iloc[0] for p in selected_compare]
        compare_colors = {p: ACCENT_SEQUENCE[i % len(ACCENT_SEQUENCE)] for i, p in enumerate(selected_compare)}

        banner_cols = st.columns(len(compare_rows), gap="large")
        for col, row in zip(banner_cols, compare_rows):
            with col:
                st.markdown(clean_html(f"""
                <div style="border-top:4px solid {compare_colors[row['player_name']]}; padding-top:0.7rem; margin-bottom:0.8rem;
                            display:flex; flex-direction:column; align-items:center; text-align:center; gap:0.5rem; min-height:230px;">
                    {player_photo_html(row, width=76, height=92)}
                    <div>
                        <h3 style="margin-bottom: 0.4rem; font-size:1.1rem;">{row['player_name']}</h3>
                        <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:0.3rem;">
                            <span class="player-tag">{row['position']}</span>
                            <span class="player-tag">{row['team']}</span>
                            {age_tag(row)}
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

        players_to_plot = df[df["player_name"].isin(selected_compare)].drop_duplicates(subset="player_name")
        st.plotly_chart(plot_radar(players_to_plot, order=selected_compare), use_container_width=True)

        st.markdown('<div class="section-title">Side-by-side stats</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">The best value on each row is bolded and colored.</div>',
            unsafe_allow_html=True,
        )

        header_cells = "".join(
            f'<th style="color:{compare_colors[p]};">{p}</th>' for p in selected_compare
        )

        body_rows = ""
        for stat_name in raw_cols + ["total_cards", "cards_p90"]:
            label = metric_labels.get(stat_name, stat_name.replace("_", " ").capitalize())
            values = [row[stat_name] for row in compare_rows]
            best_value = max(values)
            value_cells = ""
            for row, value in zip(compare_rows, values):
                fmt = f"{value:.2f}" if stat_name in ("xg", "cards_p90") else f"{value:.0f}"
                color = compare_colors[row["player_name"]] if value == best_value else "#16211C"
                weight = 700 if value == best_value else 500
                value_cells += (
                    f'<td style="font-weight:{weight}; font-size:1rem; color:{color};">{fmt}</td>'
                )
            body_rows += f'<tr><td style="color:#6B7268; text-align:left; font-family:\'Hanken Grotesk\'; font-weight:500;">{label}</td>{value_cells}</tr>'

        st.markdown(clean_html(f"""
        <table class="compare-table">
            <thead>
                <tr><th style="text-align:left; color:#6B7268;">Metric</th>{header_cells}</tr>
            </thead>
            <tbody>
                {body_rows}
            </tbody>
        </table>
        """), unsafe_allow_html=True)

with tab_rank:
    st.markdown("""
    <div class="intro-box">
        <div class="intro-step">Pick a stat below to see who leads the sample on it.</div>
    </div>
    """, unsafe_allow_html=True)

    metric_options = radar_cols + raw_cols + discipline_cols
    selected_metric = st.selectbox("Rank players by", metric_options, format_func=ranking_label)
    top_n = st.slider("Number of players shown", 5, 30, 15)

    ranking = df[["player_name", "team", "position", selected_metric]].sort_values(
        selected_metric, ascending=False
    ).head(top_n)

    # single color instead of one per position: a 6-entry color legend ate too much space and
    # started overlapping the axis title on a narrow screen, position still shows up on hover
    fig_bar = go.Figure(go.Bar(
        x=ranking[selected_metric], y=ranking["player_name"], orientation="h",
        marker_color=ACCENT_1,
        customdata=ranking[["position", "team"]],
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}, %{customdata[1]}<br>" + ranking_label(selected_metric) + ": %{x}<extra></extra>",
    ))
    fig_bar.update_layout(
        template=PLOTLY_TEMPLATE, height=max(400, top_n * 32),
        # automargin stops player names from being clipped on a narrow screen
        yaxis=dict(autorange="reversed", automargin=True, title=None),
        xaxis=dict(title=ranking_label(selected_metric)),
        paper_bgcolor=BG, plot_bgcolor="#FFFFFF", font=dict(color=TEXT),
        margin=dict(l=10, r=10, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_about:
    st.markdown("""
    <div class="player-banner">
        <h2 style="margin-bottom: 0.3rem;">About this tool</h2>
        <p style="color: #6B7268; margin: 0;">How to read the results, and their limitations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
**Data**
StatsBomb Open Data event data for the 2022 World Cup and Euro 2024 (115 matches, 400,000+ events). Metrics like xG and progressive passes/carries are recalculated directly from each event's raw coordinates rather than reused from a pre-aggregated third-party source. Player age and photo are pulled from Wikidata and may be missing for a few less-documented profiles (shown with initials instead).

**Sample**
130 players with at least 450 minutes played across both competitions. A sample of this size, limited to two international tournaments, gives solid tendencies but not a multi-season club performance history.

**How players are grouped and compared**
Each player's per-90 stats are standardized, then reduced to a small set of components with PCA (Principal Component Analysis). Players are grouped into 4 style groups with K-means clustering on those components, with no position constraint: a center-back and a winger can land in the same group if they create or defend in a similar way. Pairwise similarity between any two players is the cosine similarity between their full standardized stat profiles, not just the 2 components used for the visual grouping. Percentile scores are computed against other players in the same broad position group within the sample.

**Limitations to keep in mind**
No contract or market value data (Transfermarkt, the only reliable source for this, has no public API and actively blocks automated scraping). Since the sample is limited to international tournaments, the playing styles observed may not fully reflect a full club season. Disciplinary metrics (cards, fouls) are a risk indicator, not a judgment on the player.
    """)

photo_credit = " Player photos: Wikimedia Commons." if HAS_PHOTOS else ""
st.markdown(f"""
<div class="app-footer">
    Built by <a href="{LINKEDIN_URL}" target="_blank"><b>Camille Veillon</b></a> &middot; Polytech Nice Sophia, Applied Mathematics & Data Science &middot; <a href="{GITHUB_URL}" target="_blank">View the code</a><br>
    Data: StatsBomb Open Data, used under StatsBomb's open data license (non-commercial use). Player age: Wikidata.{photo_credit}<br>
    This is an independent, non-commercial student portfolio project. Not affiliated with or endorsed by StatsBomb, FIFA, UEFA, or any club or federation mentioned.
</div>
""", unsafe_allow_html=True)