🏈 Fantasy Draft War Room

A Streamlit-powered Fantasy Football Draft Dashboard that combines Sleeper API live draft data, FantasyPros rankings, and custom analytics to give you a real-time edge during your draft.
Track live picks, monitor positional needs across the league, and identify value players based on ADP vs. projections.

Open your browser to:
http://localhost:8501](https://sleeper-draft-buddy.streamlit.app/

🚀 Features

✅ Live Draft Tracking – Connect to your Sleeper league and follow the draft in real time

✅ Custom Fantasy Points – Half-PPR scoring with automatic calculations

✅ ADP vs. Draft Trends – Visualize how players are being drafted compared to their ADP

✅ Team Needs Analysis – Monitor which positions other teams still need

✅ Tiered Player Rankings – Integrated FantasyPros tier data with Sleeper projections

✅ Interactive Filters & Charts – Altair-based charts to explore drop-offs and positional value


⚙️ Configuration

Inside the sidebar:

League ID – Your Sleeper league ID

Draft ID – Your Sleeper draft ID

Draft Slot – Your draft position

Toggle Snake Draft (on/off)


📊 Data Sources

Sleeper API:

Player info (team, position, bye weeks)

Live draft picks

Seasonal stats & projections

FantasyPros CSVs:

Tier rankings

Strength of schedule (SOS) metrics


⚡️ Tech Stack

Python
 – Data processing & API integration
 
Streamlit
 – Interactive UI
 
Pandas
 – Data cleaning & manipulation
 
Altair
 – Interactive charts


