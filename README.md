# Player Similarity Scouting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://camille-veillon-player-similarity.streamlit.app/)

A data-driven scouting tool that finds players with a similar playing style, built on StatsBomb Open Data event data (World Cup 2022 and Euro 2024).

## What it does
- Recomputes xG, progressive passes and progressive carries directly from raw event coordinates
- Groups 130 players into playing-style clusters (KMeans on a PCA-reduced feature space), independent of their listed position
- Finds the closest statistical profiles to any given player (cosine similarity)
- Compares players side by side, with a percentile breakdown against their position group
- Flags disciplinary risk (cards, fouls) as a separate indicator

## Live app
**Live app:** [camille-veillon-player-similarity.streamlit.app](https://camille-veillon-player-similarity.streamlit.app/)

## Stack
Python, pandas, scikit-learn, Plotly, Streamlit. Data pulled via `statsbombpy`.

## Run locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Data source
[StatsBomb Open Data](https://github.com/statsbomb/open-data), used under StatsBomb's open data license (non-commercial use).

## Limitations
No contract or market value data available. Sample limited to two international tournaments (130 players, 450+ minutes played), so results reflect tournament form rather than a full club season.