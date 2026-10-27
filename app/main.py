"""Streamlit dashboard: crime hotspot predictor."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src import config as C
from src.data_preprocessor import load_features, get_feature_columns
from src.hotspot_predictor import HotspotPredictor

st.set_page_config(page_title="City of Cape Town Crime Analysis", layout="wide")


# ---------- Cached loaders ----------
@st.cache_data
def load_data():
    return load_features()


@st.cache_resource
def load_predictor():
    return HotspotPredictor()


@st.cache_data
def get_forecast():
    if C.FORECAST_PARQUET.exists():
        fc = pd.read_parquet(C.FORECAST_PARQUET)
        return fc, int(fc["year_start"].iloc[0])
    df = load_data()
    p = load_predictor()
    fc, yr = p.forecast_next_year(df)
    fc.to_parquet(C.FORECAST_PARQUET, index=False)
    return fc, yr


@st.cache_data
def compute_importance():
    df = load_data()
    p = load_predictor()
    fcols = get_feature_columns(df)
    imps = []
    for c, m in p.models.items():
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
        elif hasattr(m, "named_steps"):
            last = list(m.named_steps.values())[-1]
            imp = getattr(last, "feature_importances_", None)
            if imp is None and hasattr(last, "coef_"):
                imp = abs(last.coef_).ravel()
        elif hasattr(m, "coef_"):
            imp = abs(m.coef_).ravel()
        else:
            continue
        if len(imp) == len(fcols):
            imps.append(pd.Series(imp, name=c, index=fcols))
    if not imps:
        return None, None
    imp_df = pd.DataFrame(imps)
    return imp_df, imp_df.mean(axis=0).sort_values(ascending=False)


# ---------- Guards ----------
if not C.FEATURES_PARQUET.exists() or not C.MODELS_FILE.exists():
    st.error("Run `python train_models.py` first.")
    st.stop()

df = load_data()
predictor = load_predictor()

available_crimes = sorted(predictor.models.keys())
available_stations = sorted(df["station"].dropna().unique().tolist())
years = sorted(df["year_start"].dropna().unique().tolist(), reverse=True)
latest_year = int(years[0])
next_year = latest_year + 1
available_years = [next_year] + years


# ---------- Helpers ----------
def pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def gauge_html(prob, thr, crime):
    pct = prob * 100
    if prob >= thr + 0.2:
        color, label = "#dc2626", "VERY HIGH RISK"
    elif prob >= thr:
        color, label = "#f59e0b", "HIGH RISK"
    elif prob >= thr - 0.2:
        color, label = "#facc15", "MODERATE"
    else:
        color, label = "#16a34a", "LOW RISK"
    return f"""
    <div style="text-align:center;padding:24px;border-radius:14px;
                background:{color}22;border:2px solid {color};">
        <div style="font-size:14px;letter-spacing:2px;color:{color};
                    font-weight:600;">{label}</div>
        <div style="font-size:64px;font-weight:800;color:{color};
                    line-height:1.1;margin:8px 0;">{pct:.1f}%</div>
        <div style="font-size:14px;color:#888;">
            probability of being a <b>{crime}</b> hotspot
        </div>
    </div>
    """


def osm_map(lat, lon, zoom=12):
    return folium.Map(location=[lat, lon], zoom_start=zoom,
                      tiles="OpenStreetMap")


CHART_LAYOUT = dict(
    margin=dict(l=10, r=10, t=40, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13),
    title_font=dict(size=15),
    xaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
    yaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
)

HEAT_SCALE = "YlOrRd"
COOL_SCALE = "Blues"


def download_row(fig, data_df, base_name, key_suffix):
    """PNG download button under a Plotly chart."""
    try:
        st.download_button(
            "Download PNG",
            data=fig.to_image(format="png", width=1200, height=700, scale=2),
            file_name=f"{base_name}.png",
            mime="image/png",
            key=f"png_{key_suffix}",
            use_container_width=True,
        )
    except Exception:
        pass


# ---------- Sidebar ----------
st.sidebar.header("Make a prediction")
station = st.sidebar.selectbox("Station", available_stations)
crime = st.sidebar.selectbox("Crime type", available_crimes,
                             format_func=pretty)
year = st.sidebar.selectbox("Year", available_years)
predict_clicked = st.sidebar.button("Predict hotspot", type="primary",
                                    use_container_width=True)
st.sidebar.divider()
threshold = st.sidebar.slider("Hotspot threshold", 0.0, 1.0, 0.5, 0.05)


# ---------- Tabs ----------
tab_predict, tab_charts = st.tabs(
    ["Predict a station", "City of Cape Town analysis"]
)


# =====================================================================
# TAB 1 — Predict a station
# =====================================================================
with tab_predict:
    st.title("City of Cape Town Crime Hotspot Predictor")

    if predict_clicked:
        is_forecast = (year == next_year)
        if is_forecast:
            fc, forecast_year = get_forecast()
            row = fc[fc["station"] == station]
            effective_year = forecast_year
        else:
            row = df[(df["station"] == station) & (df["year_start"] == year)]
            effective_year = year

        if row.empty:
            st.error(f"No data for {station} in {effective_year}.")
        else:
            cache_key = (station, crime, effective_year)
            if st.session_state.get("cache_key") != cache_key:
                X = predictor._prep(row)
                prob = float(predictor.models[crime].predict_proba(X)[:, 1][0])
                if is_forecast:
                    all_probs = {c: float(row[f"{c}_prob"].iloc[0])
                                 for c in predictor.models
                                 if f"{c}_prob" in row.columns}
                else:
                    all_probs = predictor.predict_station_year(
                        df, station, effective_year)
                st.session_state.update(cache_key=cache_key, prob=prob,
                                        all_probs=all_probs,
                                        meta=row.iloc[0].to_dict())

            prob = st.session_state["prob"]
            all_probs = st.session_state["all_probs"]
            meta = st.session_state["meta"]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(gauge_html(prob, threshold, crime),
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f"### {station.title()} - {effective_year}")
                st.markdown(f"**Municipality:** {meta.get('loc_mn', '')}")
                st.markdown(f"**District:** {meta.get('dc_mn', '')}")
                lat, lon = meta.get("latitude"), meta.get("longitude")
                if lat is not None and lon is not None:
                    st.markdown(f"**Coordinates:** ({lat:.4f}, {lon:.4f})")
                st.markdown(f"**Crime:** `{crime}`")
                if prob >= threshold:
                    st.success(f"Classified as a {crime} hotspot "
                               f"at {threshold:.0%}.")
                else:
                    st.info(f"Not classified as a hotspot at {threshold:.0%}.")

            if lat is not None and lon is not None:
                m = osm_map(lat, lon, 12)
                color = "red" if prob >= threshold else "green"
                folium.CircleMarker(
                    location=[lat, lon], radius=18, color=color,
                    fill=True, fill_opacity=0.7,
                    popup=f"{station} - P({crime}) = {prob:.1%}",
                ).add_to(m)
                st.subheader("Where is this station?")
                st_folium(m, width=None, height=420, returned_objects=[])

            st.subheader("Which crimes are most likely at this station?")
            if all_probs:
                s = (pd.Series(all_probs).sort_values(ascending=False)
                       .rename("probability").to_frame())
                s["probability_pct"] = (s["probability"] * 100).round(1).astype(str) + "%"
                s["hotspot"] = s["probability"].apply(
                    lambda p: "yes" if p >= threshold else "no")
                s = s[["probability_pct", "hotspot"]].rename(
                    columns={"probability_pct": "probability"})
                st.dataframe(s, use_container_width=True)
    else:
        st.info("Pick a station, crime type, and year in the sidebar, "
                "then click Predict hotspot.")
        m = osm_map(-33.95, 18.6, 10)
        for _, r in df[df["year_start"] == latest_year].drop_duplicates("station").iterrows():
            if pd.notna(r["latitude"]) and pd.notna(r["longitude"]):
                folium.CircleMarker(
                    location=[r["latitude"], r["longitude"]], radius=5,
                    color="#3b82f6", fill=True, fill_opacity=0.7,
                    popup=r["station"]).add_to(m)
        st.subheader("Stations included in the model")
        st_folium(m, width=None, height=520, returned_objects=[])


# =====================================================================
# TAB 2 — City of Cape Town analysis
# =====================================================================
with tab_charts:
    fc, forecast_year = get_forecast()
    imp_df, mean_imp = compute_importance()

    st.header(f"City of Cape Town crime analysis — {forecast_year} forecast")

    focus_crimes = [c for c in ["murder", "rape", "carjacking",
                                "drug_crime", "common_assault", "robbery_res"]
                    if c in available_crimes]

    # ---------- Chart 1 ----------
    st.subheader("Which stations are most at risk?")
    chart_crime = st.selectbox("Pick a crime", focus_crimes,
                               format_func=pretty, key="c1c")
    col = f"{chart_crime}_prob"
    top15 = fc[["station", col]].sort_values(col, ascending=True).tail(15)
    top15["station"] = top15["station"].str.title()

    fig1 = px.bar(
        top15, x=col, y="station", orientation="h",
        color=col, color_continuous_scale=HEAT_SCALE,
        labels={col: "Chance of being a hotspot", "station": ""},
        text=top15[col].apply(lambda v: f"{v:.0%}"),
    )
    fig1.update_traces(textposition="outside", marker_line_width=0,
                       textfont_size=11)
    fig1.update_layout(height=520, coloraxis_showscale=False,
                       title=f"Top 15 stations for {pretty(chart_crime)}",
                       **CHART_LAYOUT)
    fig1.update_xaxes(range=[0, 1.05], tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)
    download_row(fig1, top15, f"top15_{chart_crime}", "c1")

    # ---------- Chart 2 ----------
    st.subheader("Where does this crime happen in Cape Town?")
    geo_df = fc.dropna(subset=["latitude", "longitude"]).copy()

    fig2 = px.scatter_map(
        geo_df, lat="latitude", lon="longitude",
        color=col, size=col, size_max=22,
        color_continuous_scale=HEAT_SCALE,
        hover_name="station",
        hover_data={"loc_mn": True, "latitude": False, "longitude": False,
                    col: ":.1%"},
        zoom=9.3, height=560, map_style="open-street-map",
        center={"lat": -33.95, "lon": 18.6},
    )
    fig2.update_layout(
        height=560, margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Chance", tickformat=".0%",
                                thickness=12, len=0.5),
    )
    st.plotly_chart(fig2, use_container_width=True)
    download_row(fig2, geo_df[["station", "latitude", "longitude", col]],
                 f"geo_{chart_crime}", "c2")

        # ---------- Chart 3 ----------
    st.subheader("Which stations are risky for which crimes?")

    n_stations = min(30, len(fc))   # fixed — no slider

    prob_cols = [f"{c}_prob" for c in available_crimes]
    matrix_full = fc.set_index("station")[prob_cols].copy()
    matrix_full.columns = [pretty(c) for c in available_crimes]

    station_mean = matrix_full.mean(axis=1).sort_values(ascending=False)
    top_n_stations = station_mean.head(n_stations).index.tolist()
    heat = matrix_full.loc[top_n_stations]

    crime_order = heat.mean(axis=0).sort_values(ascending=False).index.tolist()
    heat = heat[crime_order]

    fig3 = px.imshow(
        heat.values,
        x=heat.columns,
        y=[s.title() for s in heat.index],
        color_continuous_scale=HEAT_SCALE,
        aspect="auto",
        labels=dict(x="Crime", y="Station", color="Chance"),
        zmin=0, zmax=1,
    )
    fig3.update_layout(
        height=max(500, n_stations * 22),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13),
        coloraxis_colorbar=dict(title="Chance", tickformat=".0%",
                                thickness=12, len=0.6),
    )
    fig3.update_xaxes(side="top", tickangle=-45, tickfont=dict(size=10))
    fig3.update_yaxes(tickfont=dict(size=10), autorange="reversed")
    st.plotly_chart(fig3, use_container_width=True)

    heat_export = heat.reset_index()
    download_row(fig3, heat_export, "heatmap_station_crime", "c3")

    # ---------- Chart 4 ----------
    st.subheader("What crimes happen most often?")

    c_left, c_right = st.columns([1, 1.2])

    with c_left:
        totals_df = (df[available_crimes].sum().sort_values()
                     .rename("incidents").reset_index())
        totals_df.columns = ["crime", "incidents"]
        totals_df["crime_pretty"] = totals_df["crime"].apply(pretty)

        fig4a = px.bar(
            totals_df, x="incidents", y="crime_pretty", orientation="h",
            color="incidents", color_continuous_scale=HEAT_SCALE,
            labels={"incidents": "Reported incidents (2005–2024)",
                    "crime_pretty": ""},
        )
        fig4a.update_layout(height=780, coloraxis_showscale=False,
                            title="Total reports per crime in Cape Town",
                            **CHART_LAYOUT)
        fig4a.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig4a, use_container_width=True)
        download_row(fig4a, totals_df, "popular_crimes_citywide", "c4a")

    with c_right:
        station_totals = df.groupby("station")[available_crimes].sum().sum(axis=1)
        top_stations = station_totals.nlargest(10).index.tolist()
        top_crimes = df[available_crimes].sum().nlargest(8).index.tolist()
        matrix = (df[df["station"].isin(top_stations)]
                  .groupby("station")[top_crimes].sum().loc[top_stations])
        long_df = matrix.reset_index().melt(id_vars="station",
                                            var_name="crime",
                                            value_name="incidents")
        long_df["crime"] = long_df["crime"].apply(pretty)
        matrix.columns = [pretty(c) for c in matrix.columns]
        matrix.index = [s.title() for s in matrix.index]

        palette = px.colors.sequential.Oranges[2:] + px.colors.sequential.Greys[3:]
        fig4b = go.Figure()
        for i, c in enumerate(matrix.columns):
            fig4b.add_trace(go.Bar(
                y=matrix.index, x=matrix[c], name=c, orientation="h",
                marker_color=palette[i % len(palette)],
            ))
        fig4b.update_layout(
            barmode="stack", height=780,
           title="Top crimes at the busiest stations",
            legend=dict(orientation="v", x=1.02, y=1,
                        font=dict(size=10), title=None),
            **CHART_LAYOUT,
        )
        fig4b.update_yaxes(autorange="reversed", title=None)
        fig4b.update_xaxes(title=None)
        st.plotly_chart(fig4b, use_container_width=True)
        download_row(fig4b, long_df, "popular_crimes_top_stations", "c4b")

    # ---------- Chart 5 ----------
    st.subheader("How has crime in Cape Town changed over time?")
    trend_crimes = [c for c in ["murder", "rape", "carjacking",
                                "drug_crime", "common_assault"]
                    if c in available_crimes]
    hist_long = (df.groupby("year_start")[trend_crimes].sum()
                   .reset_index()
                   .melt(id_vars="year_start", var_name="crime",
                         value_name="incidents"))
    hist_long["crime"] = hist_long["crime"].apply(pretty)

    fig5 = px.line(
        hist_long, x="year_start", y="incidents", color="crime",
        markers=True,
        labels={"year_start": "Year",
                "incidents": "Reported incidents",
                "crime": ""},
        color_discrete_sequence=[
            "#b91c1c", "#ea580c", "#ca8a04", "#0891b2", "#4f46e5"],
    )
    fig5.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig5.update_layout(
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig5, use_container_width=True)
    download_row(fig5, hist_long, "historical_trends", "c5")

    # ---------- Chart 6 ----------
    st.subheader("What information drives the predictions?")
    if mean_imp is not None:
        c_left, c_right = st.columns(2)

        with c_left:
            top20_df = (mean_imp.head(20).iloc[::-1]
                        .rename("importance").reset_index())
            top20_df.columns = ["feature", "importance"]
            top20_df["label"] = top20_df["feature"].str.replace("_", " ")

            fig6a = px.bar(
                top20_df, x="importance", y="label", orientation="h",
                color="importance", color_continuous_scale=COOL_SCALE,
                labels={"importance": "Importance", "label": ""},
                title="Across all crime types",
            )
            fig6a.update_layout(height=640, coloraxis_showscale=False,
                                **CHART_LAYOUT)
            fig6a.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(fig6a, use_container_width=True)
            download_row(fig6a, top20_df, "feature_importance_top20", "c6a")

        with c_right:
            fi_crime = st.selectbox(
                "Pick a crime",
                [c for c in focus_crimes if c in imp_df.index],
                format_func=pretty, key="fic",
            )
            if fi_crime:
                top10_df = (imp_df.loc[fi_crime].sort_values(ascending=False)
                            .head(10).iloc[::-1]
                            .rename("importance").reset_index())
                top10_df.columns = ["feature", "importance"]
                top10_df["label"] = top10_df["feature"].str.replace("_", " ")

                fig6b = px.bar(
                    top10_df, x="importance", y="label", orientation="h",
                    color="importance", color_continuous_scale=COOL_SCALE,
                    labels={"importance": "Importance", "label": ""},
                    title=f"For {pretty(fi_crime)}",
                )
                fig6b.update_layout(height=640, coloraxis_showscale=False,
                                    **CHART_LAYOUT)
                fig6b.update_yaxes(tickfont=dict(size=10))
                st.plotly_chart(fig6b, use_container_width=True)
                download_row(fig6b, top10_df,
                             f"feature_importance_{fi_crime}", "c6b")

    # ---------- Master export ----------
    st.divider()
    full = fc[["station", "loc_mn", "dc_mn", "latitude", "longitude"] +
              [f"{c}_prob" for c in available_crimes]].copy()
    full.columns = (["station", "municipality", "district",
                     "latitude", "longitude"] +
                    [f"P_{c}" for c in available_crimes])
    st.download_button(
        "Download the full Cape Town forecast (CSV)",
        data=full.to_csv(index=False).encode("utf-8"),
        file_name=f"cape_town_forecast_{forecast_year}.csv",
        mime="text/csv",
        key="download_all",
        use_container_width=True,
    )