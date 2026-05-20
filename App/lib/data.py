"""Carga cacheada de los datasets producidos por los notebooks."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = Path(os.environ.get("CSV_DIR", PROJECT_ROOT / "CSV")).resolve()
MODEL_DIR = PROJECT_ROOT / "Modelado" / "_modelo"


@st.cache_data(show_spinner=False)
def load_pedidos() -> pd.DataFrame:
    """`04_tad_pedidos_enriquecido.csv` (a nivel pedido con banderas exógenas)."""
    df = pd.read_csv(CSV_DIR / "04_tad_pedidos_enriquecido.csv",
                     parse_dates=["order_purchase_timestamp",
                                  "order_approved_at",
                                  "order_estimated_delivery_date"])
    return df


@st.cache_data(show_spinner=False)
def load_ventas_min() -> pd.DataFrame:
    """`02_tad_ventas.csv` con un subconjunto de columnas para tarjetas y joins."""
    cols = ["order_id", "order_item_id", "seller_id", "product_id",
            "product_category_name_english",
            "price", "freight_value", "total_item_value",
            "seller_state", "seller_city",
            "seller_geo_lat", "seller_geo_lng",
            "is_late_delivery", "is_delivered",
            "anio", "mes"]
    return pd.read_csv(CSV_DIR / "02_tad_ventas.csv", usecols=cols)


@st.cache_data(show_spinner=False)
def load_clusters() -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / "06_seller_agg_clusters.csv")


@st.cache_data(show_spinner=False)
def load_perfil_clusters() -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / "06_perfil_clusters.csv")


@st.cache_data(show_spinner=False)
def cluster_stats() -> pd.DataFrame:
    """Estadísticas por cluster derivadas en vivo de `06_seller_agg_clusters.csv`.

    Devuelve la media real (sobre los sellers de cada grupo) de la tasa de
    retraso y la calificación. La calculadora de retraso las usa como features
    `seller_tasa_retraso_hist` y `seller_review_promedio`; calcularlas aquí
    evita constantes hardcodeadas que se desincronizan si cambia el clustering.
    """
    clu = load_clusters()
    return (clu.groupby(["cluster", "etiqueta"], as_index=False)
               .agg(tasa_retraso=("tasa_retraso", "mean"),
                    review_promedio=("review_promedio", "mean")))


@st.cache_data(show_spinner=False)
def load_series_diaria() -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / "07_series_demanda_diaria.csv",
                     parse_dates=["fecha"])
    return df


@st.cache_data(show_spinner=False)
def load_alertas() -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / "07_alertas_inventario.csv",
                     parse_dates=["fecha"])
    return df


@st.cache_data(show_spinner=False)
def load_metricas_series() -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / "07_metricas_series_tiempo.csv")


@st.cache_data(show_spinner=False)
def load_calendario() -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / "04_dim_calendario.csv", parse_dates=["fecha"])
    return df


@st.cache_data(show_spinner=False)
def load_serie_diaria_calendario() -> pd.DataFrame:
    """`04_serie_diaria_calendario.csv` (notebook 04): serie diaria de pedidos
    y revenue con las banderas del calendario brasileño ya unidas."""
    df = pd.read_csv(CSV_DIR / "04_serie_diaria_calendario.csv",
                     parse_dates=["fecha"])
    return df


@st.cache_data(show_spinner=False)
def eventos_uplift_ventas() -> pd.DataFrame:
    """Para cada evento del calendario (Black Friday, Carnaval, feriados, etc.)
    calcula los pedidos y revenue promedio del día del evento contra la línea
    base de días sin evento, y reporta el uplift porcentual.

    Devuelve columnas: nombre_feriado, dias_observados, pedidos_prom,
    revenue_prom, uplift_pedidos_pct, uplift_revenue_pct.
    """
    df = load_serie_diaria_calendario()
    if df.empty:
        return pd.DataFrame()
    base_pedidos = df.loc[df["tipo"].isna(), "pedidos"].mean()
    base_revenue = df.loc[df["tipo"].isna(), "revenue"].mean()

    eventos = (
        df[df["nombre_feriado"].notna()]
          .groupby("nombre_feriado")
          .agg(dias_observados=("fecha", "count"),
               pedidos_prom=("pedidos", "mean"),
               revenue_prom=("revenue", "mean"))
          .reset_index()
    )
    eventos["uplift_pedidos_pct"] = (
        (eventos["pedidos_prom"] - base_pedidos) / base_pedidos * 100
    )
    eventos["uplift_revenue_pct"] = (
        (eventos["revenue_prom"] - base_revenue) / base_revenue * 100
    )
    return eventos.sort_values("uplift_revenue_pct", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def eventos_uplift_base() -> dict:
    """Base diaria (sin evento) usada como referencia del uplift."""
    df = load_serie_diaria_calendario()
    if df.empty:
        return {"pedidos": 0.0, "revenue": 0.0}
    return {
        "pedidos": float(df.loc[df["tipo"].isna(), "pedidos"].mean()),
        "revenue": float(df.loc[df["tipo"].isna(), "revenue"].mean()),
    }


@st.cache_data(show_spinner=False)
def alertas_resumen_calendario() -> dict:
    """Cuenta cuántas alertas críticas son esperadas por calendario vs anómalas.

    El notebook 07 escribe la columna `categoria_alerta` con valores
    NORMAL / ESPERADA_POR_CALENDARIO / ANOMALA_INVESTIGAR. Aquí las contamos
    para que las vistas y el LLM puedan hacer foco en las anómalas.
    """
    al = load_alertas()
    if al.empty or "categoria_alerta" not in al.columns:
        return {"esperadas": 0, "anomalas": 0, "criticas_totales": 0}
    criticas = al[al["tipo"] != "OK"]
    return {
        "esperadas": int((criticas["categoria_alerta"] == "ESPERADA_POR_CALENDARIO").sum()),
        "anomalas":  int((criticas["categoria_alerta"] == "ANOMALA_INVESTIGAR").sum()),
        "criticas_totales": int(len(criticas)),
    }


@st.cache_data(show_spinner=False)
def load_comparacion_v1v2() -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / "08_comparacion_v1_v2.csv")


@st.cache_data(show_spinner=False)
def load_importancias() -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / "08_importancias_rf_tuneado.csv")


@st.cache_resource(show_spinner=False)
def load_model():
    """Pipeline serializado del modelo v2 (Random Forest tuneado)."""
    pipeline = joblib.load(MODEL_DIR / "rf_v2_pipeline.joblib")
    schema = joblib.load(MODEL_DIR / "rf_v2_schema.joblib")
    return pipeline, schema


# --- Combinaciones derivadas (cacheadas) ------------------------------------

@st.cache_data(show_spinner=False)
def kpi_globales() -> dict:
    """Métricas agregadas globales para el hero/header."""
    ped = load_pedidos()
    ven = load_ventas_min()
    return {
        "n_pedidos":   int(len(ped)),
        "n_items":     int(len(ven)),
        "n_sellers":   int(ven["seller_id"].nunique()),
        "n_productos": int(ven["product_id"].nunique()),
        "gmv":         float(ven["price"].sum() + ven["freight_value"].sum()),
        "gmv_price":   float(ven["price"].sum()),
        "tasa_retraso":float(ped["is_late_delivery"].mean()),
        "tasa_retraso_items": float(ven["is_late_delivery"].mean()),
        "n_alertas":   int(len(load_alertas())),
        "rango_fechas": (ped["order_purchase_timestamp"].min(),
                         ped["order_purchase_timestamp"].max()),
    }


@st.cache_data(show_spinner=False)
def retraso_por_estado() -> pd.DataFrame:
    ped = load_pedidos()
    out = (ped.groupby("customer_state", as_index=False)
              .agg(pedidos=("order_id", "count"),
                   tasa_retraso=("is_late_delivery", "mean")))
    return out.sort_values("tasa_retraso", ascending=False)


@st.cache_data(show_spinner=False)
def retraso_por_mes() -> pd.DataFrame:
    ped = load_pedidos()
    ped = ped.dropna(subset=["order_purchase_timestamp"]).copy()
    ped["aniomes"] = ped["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    out = (ped.groupby("aniomes", as_index=False)
              .agg(pedidos=("order_id", "count"),
                   tasa_retraso=("is_late_delivery", "mean")))
    return out.sort_values("aniomes")


@st.cache_data(show_spinner=False)
def ventas_por_mes() -> pd.DataFrame:
    """Agrega ventas mensuales: pedidos, items, ingresos y ticket promedio."""
    ven = load_ventas_min()
    if ven.empty:
        return pd.DataFrame(columns=["aniomes", "anio", "mes", "items",
                                       "pedidos", "ingresos", "ticket_promedio"])
    df = ven.copy()
    df["aniomes"] = pd.to_datetime(
        df["anio"].astype(int).astype(str) + "-" +
        df["mes"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    out = (df.groupby(["aniomes", "anio", "mes"], as_index=False)
              .agg(items=("order_item_id", "count"),
                   pedidos=("order_id", "nunique"),
                   ingresos=("price", "sum")))
    out["ticket_promedio"] = out["ingresos"] / out["pedidos"].clip(lower=1)
    return out.sort_values("aniomes")


@st.cache_data(show_spinner=False)
def ventas_por_mes_calendario() -> pd.DataFrame:
    """Promedio de ingresos por número de mes (1-12), sumando todos los años."""
    vm = ventas_por_mes()
    if vm.empty:
        return pd.DataFrame(columns=["mes", "ingresos_promedio",
                                       "pedidos_promedio", "n_anios"])
    out = (vm.groupby("mes", as_index=False)
              .agg(ingresos_promedio=("ingresos", "mean"),
                   pedidos_promedio=("pedidos", "mean"),
                   n_anios=("anio", "nunique")))
    return out.sort_values("ingresos_promedio", ascending=False)


@st.cache_data(show_spinner=False)
def ventas_por_estado_cliente() -> pd.DataFrame:
    """Ingresos por estado del cliente (a nivel pedido). Útil para 'dónde se vende más'."""
    ped = load_pedidos()
    if ped.empty:
        return pd.DataFrame(columns=["customer_state", "pedidos", "ingresos"])
    if "payment_value" in ped.columns:
        out = ped.groupby("customer_state", as_index=False).agg(
            pedidos=("order_id", "count"),
            ingresos=("payment_value", "sum"),
        )
    else:
        out = ped.groupby("customer_state", as_index=False).agg(
            pedidos=("order_id", "count"),
        )
        out["ingresos"] = 0.0
    return out.sort_values("ingresos", ascending=False)


@st.cache_data(show_spinner=False)
def ventas_cat_por_mes() -> pd.DataFrame:
    """Ingresos por (categoría × año-mes). Base para preguntas categoría↔mes."""
    ven = load_ventas_min()
    if ven.empty:
        return pd.DataFrame(columns=["aniomes", "anio", "mes",
                                       "product_category_name_english",
                                       "items", "ingresos"])
    df = ven.dropna(subset=["product_category_name_english"]).copy()
    df["aniomes"] = pd.to_datetime(
        df["anio"].astype(int).astype(str) + "-" +
        df["mes"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    out = (df.groupby(["aniomes", "anio", "mes",
                         "product_category_name_english"],
                        as_index=False)
              .agg(items=("order_item_id", "count"),
                   ingresos=("price", "sum")))
    return out


@st.cache_data(show_spinner=False)
def mes_pico_por_categoria(top_n: int = 5) -> pd.DataFrame:
    """Para las top_n categorías por ingresos, mes-calendario más fuerte
    (promedio entre años) e ingresos asociados."""
    base = ventas_cat_por_mes()
    if base.empty:
        return pd.DataFrame()
    top_cats = (base.groupby("product_category_name_english", as_index=False)
                       .agg(ingresos_totales=("ingresos", "sum"))
                       .sort_values("ingresos_totales", ascending=False)
                       .head(top_n)["product_category_name_english"]
                       .tolist())
    sub = base[base["product_category_name_english"].isin(top_cats)]
    cal = (sub.groupby(["product_category_name_english", "mes"], as_index=False)
              .agg(ingresos_promedio=("ingresos", "mean")))
    # Para cada cat: mes con mayor promedio.
    idx = cal.groupby("product_category_name_english")["ingresos_promedio"].idxmax()
    pico = cal.loc[idx].reset_index(drop=True)
    pico["orden"] = pico["product_category_name_english"].map(
        {c: i for i, c in enumerate(top_cats)})
    return pico.sort_values("orden").drop(columns="orden")


@st.cache_data(show_spinner=False)
def top_cats_por_estado(top_estados: int = 5, top_cats: int = 3) -> pd.DataFrame:
    """Top categorías por ingresos para cada estado top (del cliente).

    Requiere unir ventas (categoría, ingresos, order_id) con pedidos
    (order_id → customer_state).
    """
    ven = load_ventas_min()
    ped = load_pedidos()
    if ven.empty or ped.empty:
        return pd.DataFrame()
    cross = ven.merge(ped[["order_id", "customer_state"]],
                       on="order_id", how="inner")
    cross = cross.dropna(subset=["product_category_name_english"])
    # Top estados por ingresos.
    estados_top = (cross.groupby("customer_state", as_index=False)
                          .agg(ingresos=("price", "sum"))
                          .sort_values("ingresos", ascending=False)
                          .head(top_estados)["customer_state"]
                          .tolist())
    sub = cross[cross["customer_state"].isin(estados_top)]
    agg = (sub.groupby(["customer_state", "product_category_name_english"],
                          as_index=False)
              .agg(ingresos=("price", "sum")))
    # Top N categorías por estado.
    agg = (agg.sort_values(["customer_state", "ingresos"],
                              ascending=[True, False])
              .groupby("customer_state", as_index=False)
              .head(top_cats))
    # Ranking para presentación.
    agg["rank"] = (agg.groupby("customer_state").cumcount() + 1)
    agg["orden_estado"] = agg["customer_state"].map(
        {e: i for i, e in enumerate(estados_top)})
    return (agg.sort_values(["orden_estado", "rank"])
                .drop(columns="orden_estado")
                .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def clusters_por_seller_state(top_estados: int = 6) -> pd.DataFrame:
    """Distribución de clusters (etiquetas) por estado donde vive el vendedor."""
    clu = load_clusters()
    ven = load_ventas_min()
    if clu.empty or ven.empty:
        return pd.DataFrame()
    sellers_state = (ven.dropna(subset=["seller_state"])
                          .groupby("seller_id", as_index=False)["seller_state"]
                          .agg(lambda s: s.mode().iloc[0] if not s.mode().empty
                                else s.iloc[0]))
    merged = clu.merge(sellers_state, on="seller_id", how="inner")
    # Top estados por # sellers.
    estados_top = (merged.groupby("seller_state", as_index=False)
                            .agg(n=("seller_id", "count"))
                            .sort_values("n", ascending=False)
                            .head(top_estados)["seller_state"].tolist())
    sub = merged[merged["seller_state"].isin(estados_top)]
    out = (sub.groupby(["seller_state", "etiqueta"], as_index=False)
              .agg(n_sellers=("seller_id", "count")))
    out["orden"] = out["seller_state"].map(
        {e: i for i, e in enumerate(estados_top)})
    return (out.sort_values(["orden", "n_sellers"], ascending=[True, False])
                .drop(columns="orden")
                .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def top_categorias(n: int = 12) -> pd.DataFrame:
    ven = load_ventas_min()
    out = (ven.groupby("product_category_name_english", as_index=False)
              .agg(items=("order_item_id", "count"),
                   ingresos=("price", "sum"),
                   tasa_retraso=("is_late_delivery", "mean"))
              .sort_values("ingresos", ascending=False)
              .head(n))
    return out


@st.cache_data(show_spinner=False)
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia entre dos puntos (vectorizable). Pública para la calculadora."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


@st.cache_data(show_spinner=False)
def coords_por_estado() -> pd.DataFrame:
    """Centroide aproximado por estado (cliente)."""
    ped = load_pedidos()
    return (ped.dropna(subset=["geo_lat", "geo_lng"])
              .groupby("customer_state", as_index=False)
              .agg(geo_lat=("geo_lat", "mean"),
                   geo_lng=("geo_lng", "mean")))


@st.cache_data(show_spinner=False)
def coords_seller_por_estado() -> pd.DataFrame:
    ven = load_ventas_min()
    return (ven.dropna(subset=["seller_geo_lat", "seller_geo_lng"])
              .groupby("seller_state", as_index=False)
              .agg(seller_geo_lat=("seller_geo_lat", "mean"),
                   seller_geo_lng=("seller_geo_lng", "mean")))
