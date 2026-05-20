"""Funciones de gráficos reutilizables con el tema del dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import theme as T


def _apply(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(template=T.plotly_template(), height=height)
    return fig


# --- Heatmap por estado -----------------------------------------------------
def bar_estado_retraso(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    d = df.head(top_n).copy()
    d["pct"] = d["tasa_retraso"] * 100
    fig = go.Figure(go.Bar(
        x=d["pct"], y=d["customer_state"], orientation="h",
        marker=dict(
            color=d["pct"],
            colorscale=[[0, T.EMERALD], [0.5, T.AMBER], [1, T.ROSE]],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in d["pct"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", color=T.TEXT, size=11),
        hovertemplate="<b>%{y}</b><br>Pedidos: %{customdata:,}<br>Con retraso: %{x:.2f}%<extra></extra>",
        customdata=d["pedidos"],
    ))
    fig.update_xaxes(title="% de pedidos con retraso", ticksuffix="%")
    fig.update_yaxes(title="", autorange="reversed")
    return _apply(fig, height=max(280, 22 * len(d)))


# --- Tendencia mensual ------------------------------------------------------
def line_tendencia_mensual(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["aniomes"], y=df["pedidos"], name="Pedidos",
        marker_color="rgba(56,189,248,0.35)", marker_line_width=0,
        hovertemplate="<b>%{x|%b %Y}</b><br>Pedidos: %{y:,}<extra></extra>",
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=df["aniomes"], y=df["tasa_retraso"] * 100, name="% con retraso",
        mode="lines+markers",
        line=dict(color=T.AMBER, width=2.2),
        marker=dict(size=7, color=T.AMBER, line=dict(color=T.BG, width=1.5)),
        hovertemplate="<b>%{x|%b %Y}</b><br>Con retraso: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=""),
        yaxis=dict(title="% de pedidos con retraso", ticksuffix="%", side="left",
                    rangemode="tozero"),
        yaxis2=dict(title="Pedidos", overlaying="y", side="right",
                     showgrid=False, rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return _apply(fig, height=380)


# --- Ventas por mes (estacionalidad) ---------------------------------------
def bar_ventas_mensuales(df: pd.DataFrame) -> go.Figure:
    """Barras de ingresos mes a mes (espera columnas `aniomes`, `ingresos`)."""
    d = df.sort_values("aniomes").copy()
    fig = go.Figure(go.Bar(
        x=d["aniomes"], y=d["ingresos"],
        marker=dict(color=T.AMBER, line=dict(width=0)),
        text=[f"R$ {v/1e3:.0f}K" for v in d["ingresos"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", color=T.TEXT_DIM, size=10),
        hovertemplate="<b>%{x|%b %Y}</b><br>Ingresos: R$ %{y:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Ingresos (R$)")
    return _apply(fig, height=360)


# --- Top categorías ---------------------------------------------------------
def bar_categorias(df: pd.DataFrame, metric: str = "ingresos") -> go.Figure:
    d = df.sort_values(metric, ascending=True).copy()
    label = "Ingresos (R$)" if metric == "ingresos" else "Productos vendidos"
    txt = ([f"R$ {v/1e6:.2f}M" for v in d[metric]] if metric == "ingresos"
           else [f"{int(v):,}" for v in d[metric]])
    fig = go.Figure(go.Bar(
        x=d[metric], y=d["product_category_name_english"], orientation="h",
        marker=dict(color=T.AMBER, line=dict(width=0)),
        text=txt, textposition="outside",
        textfont=dict(family="JetBrains Mono", color=T.TEXT_DIM, size=11),
        hovertemplate="<b>%{y}</b><br>"+label+": %{x:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="")
    return _apply(fig, height=max(280, 28 * len(d)))


# --- SARIMA forecast + alertas ---------------------------------------------
def forecast_chart(historico: pd.DataFrame, alertas: pd.DataFrame,
                    categoria: str) -> go.Figure:
    """`historico` con columnas (fecha, demanda); `alertas` con columnas
    (fecha, tipo, lim_inf, lim_sup, p10_hist, p90_hist).
    El pronóstico puntual se aproxima con el centro del IC 90 %."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historico["fecha"], y=historico["demanda"],
        name="Ventas reales (lo que ya pasó)", mode="lines",
        line=dict(color=T.SKY, width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Ventas: %{y}<extra></extra>",
    ))

    if not alertas.empty:
        a = alertas.sort_values("fecha").copy()
        a["pronostico"] = (a["lim_inf"] + a["lim_sup"]) / 2

        # Bandas P10 / P90 históricas
        fig.add_trace(go.Scatter(
            x=a["fecha"], y=a["p90_hist"], mode="lines",
            line=dict(color=T.ROSE, width=1, dash="dot"),
            name="Máximo histórico",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Máximo histórico: %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=a["fecha"], y=a["p10_hist"], mode="lines",
            line=dict(color=T.EMERALD, width=1, dash="dot"),
            name="Mínimo histórico",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Mínimo histórico: %{y:.1f}<extra></extra>",
        ))

        # Banda IC 90 %
        fig.add_trace(go.Scatter(
            x=a["fecha"], y=a["lim_sup"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=a["fecha"], y=a["lim_inf"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(245,158,11,0.18)", name="Rango esperado",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Mínimo esperado: %{y:.1f}<extra></extra>",
        ))

        # Pronóstico puntual
        fig.add_trace(go.Scatter(
            x=a["fecha"], y=a["pronostico"], mode="lines",
            line=dict(color=T.AMBER, width=2.4),
            name="Lo que se espera vender",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Esperado: %{y:.1f}<extra></extra>",
        ))

        # Marcadores de alertas
        labels = {"STOCKOUT": "Riesgo de quedarse sin stock",
                  "SOBRE-STOCK": "Riesgo de exceso de inventario",
                  "OK": "Día normal"}
        for tipo, color, sym in [("STOCKOUT", T.ROSE, "triangle-up"),
                                   ("SOBRE-STOCK", T.AMBER, "triangle-down"),
                                   ("OK", T.EMERALD, "circle")]:
            sub = a[a["tipo"] == tipo]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["fecha"], y=sub["pronostico"],
                mode="markers", name=labels[tipo],
                marker=dict(color=color, size=10, symbol=sym,
                             line=dict(color=T.BG, width=1.2)),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>" + labels[tipo]
                              + "<br>Esperado: %{y:.1f}<extra></extra>",
            ))

    fig.update_layout(
        title=f"Ventas por día — {categoria}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Productos vendidos por día", rangemode="tozero")
    return _apply(fig, height=460)


# --- Picos de ventas por evento del calendario ------------------------------
def bar_uplift_eventos(df: pd.DataFrame) -> go.Figure:
    """Barras horizontales del uplift de revenue (%) por evento del calendario.
    Verdes los eventos que suben las ventas, ámbar los que las bajan."""
    d = df.sort_values("uplift_revenue_pct", ascending=True).copy()
    colors = [T.EMERALD if v > 0 else T.AMBER for v in d["uplift_revenue_pct"]]
    texto = [f"{v:+.0f}%" for v in d["uplift_revenue_pct"]]
    fig = go.Figure(go.Bar(
        x=d["uplift_revenue_pct"], y=d["nombre_feriado"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=texto, textposition="outside",
        textfont=dict(family="JetBrains Mono", color=T.TEXT_DIM, size=11),
        customdata=np.stack([
            d["pedidos_prom"], d["revenue_prom"], d["uplift_pedidos_pct"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pedidos promedio: %{customdata[0]:,.0f}<br>"
            "Revenue promedio: R$ %{customdata[1]:,.0f}<br>"
            "Uplift pedidos: %{customdata[2]:+.1f}%<br>"
            "Uplift revenue: %{x:+.1f}%<extra></extra>"
        ),
    ))
    fig.update_xaxes(title="Diferencia vs día normal (% en revenue)",
                     ticksuffix="%", zeroline=True, zerolinecolor=T.BORDER,
                     zerolinewidth=1)
    fig.update_yaxes(title="")
    return _apply(fig, height=max(280, 26 * len(d)))


# --- Scatter de sellers (clusters) ------------------------------------------
def scatter_sellers(df: pd.DataFrame, color_palette: list[str]) -> go.Figure:
    d = df.copy()
    d["ingresos_log"] = np.log10(d["ingresos_totales"].clip(lower=1))
    fig = px.scatter(
        d, x="tasa_retraso", y="ingresos_log",
        color="etiqueta", color_discrete_sequence=color_palette,
        size="n_pedidos", size_max=22,
        hover_data={"seller_id": True, "n_pedidos": True,
                    "review_promedio": ":.2f",
                    "ingresos_totales": ":,.0f",
                    "tasa_retraso": ":.2%",
                    "ingresos_log": False},
        labels={"tasa_retraso": "% de pedidos con retraso",
                "ingresos_log": "Ingresos (escala comprimida)",
                "etiqueta": "Grupo"},
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color=T.BG)))
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                   y=1.02, xanchor="right", x=1))
    return _apply(fig, height=460)


# --- Importancias -----------------------------------------------------------
def bar_importancias(df: pd.DataFrame, top: int = 12) -> go.Figure:
    d = df.head(top).copy().iloc[::-1]
    d["pct"] = d["importancia"] * 100
    fig = go.Figure(go.Bar(
        x=d["pct"], y=d["feature"], orientation="h",
        marker=dict(color=T.AMBER, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in d["pct"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", color=T.TEXT_DIM, size=11),
        hovertemplate="<b>%{y}</b><br>Influencia: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(title="Influencia del factor en la predicción", ticksuffix="%")
    fig.update_yaxes(title="")
    return _apply(fig, height=max(280, 26 * len(d)))


# --- Gauge de probabilidad (modelo) -----------------------------------------
def gauge_prob(prob: float, base: float) -> go.Figure:
    color = T.EMERALD if prob < 0.15 else T.AMBER if prob < 0.30 else T.ROSE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 32, "color": T.TEXT,
                                          "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                      "tickcolor": T.BORDER,
                      "tickfont": {"color": T.TEXT_MUTED, "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1, "bordercolor": T.BORDER,
            "steps": [
                {"range": [0, 15], "color": "rgba(16,185,129,0.12)"},
                {"range": [15, 30], "color": "rgba(245,158,11,0.12)"},
                {"range": [30, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": T.TEXT_DIM, "width": 2},
                "thickness": 0.7,
                "value": base * 100,
            },
        },
    ))
    fig.update_layout(template=T.plotly_template(), height=240,
                       margin=dict(l=20, r=20, t=20, b=10))
    return fig


# --- Comparación v1 vs v2 ---------------------------------------------------
def bar_compare_v1v2(df: pd.DataFrame) -> go.Figure:
    """`df` debe tener columnas: modelo, f1_1, roc_auc, pr_auc, recall_1."""
    metrics = ["f1_1", "roc_auc", "pr_auc", "recall_1"]
    names = {"f1_1": "Calidad general",
             "roc_auc": "Capacidad de distinguir",
             "pr_auc": "Precisión en casos difíciles",
             "recall_1": "Retrasos detectados"}
    long = df.melt(id_vars="modelo", value_vars=metrics,
                    var_name="metrica", value_name="valor")
    long["metrica"] = long["metrica"].map(names)
    fig = px.bar(long, x="modelo", y="valor", color="metrica",
                  barmode="group", text=long["valor"].round(2),
                  color_discrete_sequence=[T.AMBER, T.SKY, T.EMERALD, T.VIOLET])
    fig.update_traces(textposition="outside",
                       textfont=dict(family="JetBrains Mono",
                                     color=T.TEXT_DIM, size=10))
    fig.update_yaxes(title="Puntaje (mayor es mejor)", rangemode="tozero")
    fig.update_xaxes(title="", tickangle=-15)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                   y=1.02, xanchor="right", x=1))
    return _apply(fig, height=420)
