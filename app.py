import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Player Similarity Scouting", layout="wide")

PLOTLY_TEMPLATE = "plotly_white"
ACCENT_1 = "#1455C0"
ACCENT_2 = "#E8502A"
ACCENT_SEQUENCE = ["#1455C0", "#E8502A", "#00A86B", "#D81B60", "#8E6FC9"]
CLUSTER_COLORS = {0: "#1455C0", 1: "#FF7A00", 2: "#00A86B", 3: "#D81B60"}
BG = "#F3F5F8"
TEXT = "#0F1E3D"
MUTED = "#5B6878"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 18px; color: #0F1E3D; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; letter-spacing: 0.01em; color: #0F1E3D; }

.block-container { padding-top: 2rem; padding-left: 3rem; padding-right: 3rem; padding-bottom: 1rem; max-width: 100%; }

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border-top: 5px solid #1455C0;
    border-radius: 4px;
    padding: 1rem 1.1rem;
}
[data-testid="column"]:nth-of-type(even) [data-testid="stMetric"] { border-top-color: #E8502A; }

[data-testid="stMetricLabel"] { color: #5B6878; white-space: normal; overflow: visible; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.02em; }
[data-testid="stMetricValue"] { color: #0F1E3D; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 2rem; }

[data-testid="column"] { padding: 0.45rem; }

.stTabs [data-baseweb="tab-list"] { gap: 0.6rem; border-bottom: 3px solid #0F1E3D; padding-bottom: 0; }
.stTabs [data-baseweb="tab"] {
    height: auto;
    color: #5B6878; font-weight: 700; font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.03em;
    padding: 0.95rem 1.8rem; margin-bottom: -3px; border-radius: 6px 6px 0 0;
    transition: background-color 0.15s ease, color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #0F1E3D; }
.stTabs [aria-selected="true"] { color: #FFFFFF; background-color: #0F1E3D; }

[data-testid="stDataFrame"] { border: 1px solid #D7DCE4; border-radius: 4px; font-size: 0.95rem; }

[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #D7DCE4;
    border-radius: 6px;
    margin-top: 1.2rem;
}

.app-header { border-bottom: 4px solid #0F1E3D; padding-bottom: 1.2rem; margin-bottom: 1.8rem; }

.intro-box {
    background-color: #FFFFFF;
    border: 1px solid #D7DCE4;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.8rem;
}
.intro-step { color: #0F1E3D; margin-bottom: 0.35rem; }
.intro-step b { color: #E8502A; }

.section-title { margin-top: 2.1rem; margin-bottom: 0.6rem; color: #0F1E3D; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.35rem; text-transform: uppercase; letter-spacing: 0.02em; }
.section-subtitle { color: #5B6878; margin-bottom: 1rem; font-size: 0.95rem; }
.stat-group-label { font-size: 0.88rem; font-weight: 600; color: #5B6878; text-transform: uppercase; letter-spacing: 0.03em; margin: 1.1rem 0 0.6rem; }

.player-banner { border-bottom: 1px solid #D7DCE4; padding-bottom: 1.3rem; margin-bottom: 1.6rem; }
.player-tag { display: inline-block; background-color: #FFFFFF; border: 1px solid #D7DCE4; color: #5B6878; padding: 0.3rem 0.85rem; border-radius: 16px; font-size: 0.82rem; margin-right: 0.5rem; }
.player-tag-accent { display: inline-block; background-color: #E8502A; color: #FFFFFF; font-weight: 600; padding: 0.3rem 0.85rem; border-radius: 16px; font-size: 0.82rem; margin-right: 0.5rem; }

.compare-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
.compare-table th { padding: 1rem; font-size: 1rem; border-bottom: 3px solid #0F1E3D; }
.compare-table td { padding: 1rem; text-align: center; border-bottom: 1px solid #E3E7ED; }
.compare-table tr:nth-child(even) td { background-color: #FFFFFF; }

.app-footer {
    border-top: 1px solid #D7DCE4; margin-top: 3rem; padding-top: 1.2rem;
    color: #5B6878; font-size: 0.85rem; text-align: center; line-height: 1.6;
}
.app-footer b { color: #0F1E3D; }
</style>
""", unsafe_allow_html=True)

with open("data/player_profiles.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
similarity_matrix = data["similarity_matrix"]
radar_cols = data["radar_cols"]

raw_cols = ["minutes_played", "shots", "goals", "xg", "passes_completed",
            "progressive_passes", "progressive_carries", "Ball Recovery",
            "Interception", "Pressure"]

discipline_cols = ["total_cards", "yellow_cards", "red_cards", "cards_p90", "fouls_committed_p90", "fouls_won"]
excluded_from_pct = ["cards_p90", "fouls_committed_p90"]

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

def age_tag(row):
    if "age" in row.index and pd.notna(row["age"]):
        return f'<span class="player-tag-accent">{row["age"]:.0f} yo</span>'
    return ""

def clean_html(html):
    return "\n".join(line.strip() for line in html.strip().split("\n"))


def find_similar_players(name, n):
    idx = df.index[df["player_name"] == name][0]
    scores = similarity_matrix[df.index.get_loc(idx)]
    similar_idx = scores.argsort()[::-1][1:n + 1]
    cols_to_show = ["player_name", "position", "team", "cluster", "total_cards", "yellow_cards", "red_cards", "cards_p90"]
    result = df.iloc[similar_idx][cols_to_show].copy()
    result["similarity"] = scores[similar_idx].round(3)
    return result


def plot_radar(players_to_plot, order=None):
    if order:
        players_to_plot = players_to_plot.set_index("player_name").loc[order].reset_index()
    radar_df = players_to_plot.set_index("player_name")[radar_cols]
    radar_norm = (radar_df - df[radar_cols].min()) / (df[radar_cols].max() - df[radar_cols].min())

    fig = go.Figure()
    for i, (name, row) in enumerate(radar_norm.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=radar_cols + [radar_cols[0]],
            fill="toself", name=name, opacity=0.5,
            line=dict(color=ACCENT_SEQUENCE[i % len(ACCENT_SEQUENCE)]),
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG,
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E3E7ED"),
            angularaxis=dict(gridcolor="#E3E7ED"),
        ),
        showlegend=True, height=650, margin=dict(t=40, b=30),
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
    st.markdown(f"""
    <div class="player-banner">
        <h2 style="margin-bottom: 0.4rem;">{row['player_name']}</h2>
        <span class="player-tag">{row['position']}</span>
        <span class="player-tag">{row['team']}</span>
        <span class="player-tag">{minutes_fmt}</span>
        {age_tag(row)}
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="app-header">
    <h1 style="margin-bottom: 0.3rem; font-size: 3rem;">Player Similarity Scouting</h1>
    <p style="color: #5B6878; margin: 0; font-size: 1.05rem;">World Cup 2022 and Euro 2024 &middot; StatsBomb Open Data</p>
</div>
""", unsafe_allow_html=True)

tab_search, tab_compare, tab_clusters, tab_rank, tab_about = st.tabs(
    ["Find a player", "Compare players", "Style map", "Top players", "About"]
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

        render_player_banner(main_row)
        render_stat_groups(main_row)

        st.markdown('<div class="section-title">Players with a similar playing style</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Click any row in the table to pull up that player\'s own stats below.</div>',
            unsafe_allow_html=True,
        )

        clicked_name = None
        try:
            event = st.dataframe(
                similar, use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row",
            )
            if event.selection.rows:
                clicked_name = similar.iloc[event.selection.rows[0]]["player_name"]
        except TypeError:
            st.dataframe(similar, use_container_width=True, hide_index=True)
            st.caption("Upgrade streamlit (pip install --upgrade streamlit) to enable click-to-select rows.")

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

        all_candidates = [main_name] + similar["player_name"].tolist()
        selected_players = st.multiselect(
            "Players shown on the chart",
            options=all_candidates,
            default=all_candidates[:4],
        )

        if selected_players:
            ordered_selection = [p for p in all_candidates if p in selected_players]
            players_to_plot = df[df["player_name"].isin(selected_players)].drop_duplicates(subset="player_name")
            st.plotly_chart(plot_radar(players_to_plot, order=ordered_selection), use_container_width=True)
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

        fig_pct = go.Figure(go.Bar(
            x=pct_values.values,
            y=[p90_label(c.replace("_pct", "")) for c in pct_values.index],
            orientation="h", marker_color=ACCENT_1,
        ))
        fig_pct.add_vline(x=50, line_dash="dash", line_color="#5B6878")
        fig_pct.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor="#FFFFFF",
            xaxis=dict(range=[0, 100], title="Percentile", gridcolor="#E3E7ED"),
            yaxis=dict(gridcolor="#E3E7ED"),
            height=420, margin=dict(t=20, b=20),
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
                <div style="background-color:#FFFFFF; border:1px solid #D7DCE4; border-left:4px solid {color};
                            padding:0.75rem 1.1rem; margin-bottom:0.6rem; border-radius:4px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#0F1E3D;">{label}</span>
                    <span style="color:{color}; font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:1.15rem;">
                        {val:.0f}<span style="font-size:0.75rem; color:#5B6878;"> pct</span>
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
    default_compare = all_players[:2]

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
                st.markdown(f"""
                <div style="border-top:4px solid {compare_colors[row['player_name']]}; padding-top:0.6rem; margin-bottom:0.8rem;">
                    <h3 style="margin-bottom: 0.3rem;">{row['player_name']}</h3>
                    <span class="player-tag">{row['position']}</span>
                    <span class="player-tag">{row['team']}</span>
                    {age_tag(row)}
                </div>
                """, unsafe_allow_html=True)

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
                color = compare_colors[row["player_name"]] if value == best_value else "#0F1E3D"
                weight = 700 if value == best_value else 400
                value_cells += (
                    f'<td style="font-family:\'Barlow Condensed\', sans-serif; font-weight:{weight}; '
                    f'font-size:1.2rem; color:{color};">{fmt}</td>'
                )
            body_rows += f'<tr><td style="color:#5B6878; text-align:left; font-family:Inter; font-weight:500;">{label}</td>{value_cells}</tr>'

        st.markdown(clean_html(f"""
        <table class="compare-table">
            <thead>
                <tr><th style="text-align:left; color:#5B6878;">Metric</th>{header_cells}</tr>
            </thead>
            <tbody>
                {body_rows}
            </tbody>
        </table>
        """), unsafe_allow_html=True)

with tab_clusters:
    st.markdown("""
    <div class="intro-box">
        <div class="intro-step">Every dot is a player. Players close together, and sharing a color, play in a similar style, whatever their position on the pitch.</div>
    </div>
    """, unsafe_allow_html=True)

    df_plot = df.copy()
    df_plot["cluster_label"] = df_plot["cluster"].map(lambda c: f"Group {c}")

    fig_scatter = go.Figure()
    for cluster_id, color in CLUSTER_COLORS.items():
        subset = df_plot[df_plot["cluster"] == cluster_id]
        if subset.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=subset["pca_x"], y=subset["pca_y"],
            mode="markers", name=f"Group {cluster_id}",
            marker=dict(size=10, color=color, opacity=0.85, line=dict(width=1, color="#FFFFFF")),
            text=subset["player_name"],
            customdata=subset[["position", "team"]],
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}, %{customdata[1]}<extra></extra>",
        ))
    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT), height=600, legend_title_text="Style group",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab_rank:
    st.markdown("""
    <div class="intro-box">
        <div class="intro-step">Pick a stat below to see who leads the sample on it.</div>
    </div>
    """, unsafe_allow_html=True)

    metric_options = radar_cols + raw_cols + discipline_cols
    selected_metric = st.selectbox("Rank players by", metric_options)
    top_n = st.slider("Number of players shown", 5, 30, 15)

    ranking = df[["player_name", "team", "position", selected_metric]].sort_values(
        selected_metric, ascending=False
    ).head(top_n)

    fig_bar = px.bar(
        ranking, x=selected_metric, y="player_name", orientation="h",
        color="position", height=max(400, top_n * 28),
        template=PLOTLY_TEMPLATE, color_discrete_sequence=list(CLUSTER_COLORS.values()) + ACCENT_SEQUENCE,
    )
    fig_bar.update_layout(
        yaxis=dict(autorange="reversed"), paper_bgcolor=BG, plot_bgcolor="#FFFFFF", font=dict(color=TEXT),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_about:
    st.markdown("""
    <div class="player-banner">
        <h2 style="margin-bottom: 0.3rem;">About this tool</h2>
        <p style="color: #5B6878; margin: 0;">How to read the results, and their limitations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
**Data**
StatsBomb Open Data event data for the 2022 World Cup and Euro 2024 (115 matches, 400,000+ events). Metrics like xG and progressive passes/carries are recalculated directly from each event's raw coordinates rather than reused from a pre-aggregated third-party source. Player age is pulled from Wikidata and may be missing for a few less-documented profiles.

**Sample**
130 players with at least 450 minutes played across both competitions. A sample of this size, limited to two international tournaments, gives solid tendencies but not a multi-season club performance history.

**How players are grouped and compared**
Players are grouped by playing style (a statistical clustering method, with no position constraint), then compared to each other by how close their overall style is. Percentile scores are computed against other players in the same broad position group within the sample.

**Limitations to keep in mind**
No contract or market value data (Transfermarkt, the only reliable source for this, has no public API and actively blocks automated scraping). Since the sample is limited to international tournaments, the playing styles observed may not fully reflect a full club season. Disciplinary metrics (cards, fouls) are a risk indicator, not a judgment on the player.
    """)

st.markdown("""
<div class="app-footer">
    Built by <b>Camille Veillon</b> &middot; Polytech Nice Sophia, Applied Mathematics & Data Science<br>
    Data: StatsBomb Open Data, used under StatsBomb's open data license (non-commercial use). Player age: Wikidata.<br>
    This is an independent, non-commercial student portfolio project. Not affiliated with or endorsed by StatsBomb, FIFA, UEFA, or any club or federation mentioned.
</div>
""", unsafe_allow_html=True)