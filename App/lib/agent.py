"""Agente conversacional — motor de intents sobre las sábanas del proyecto.

Estrategia
----------
Pipeline clásico de minería de texto (visto en clase): normalización
(lowercase + strip de acentos), extracción de entidades por diccionario
(estados, categorías, clusters) y matching por puntaje contra un conjunto
finito de intents. Cada intent tiene un handler que consulta los DataFrames
ya cacheados en ``lib.data`` y devuelve una respuesta estructurada
(texto en lenguaje de negocio + tabla y/o gráfica opcionales).

No requiere modelos externos ni LLM: queda dentro del cinturón de tecnologías
revisadas en el curso.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
import plotly.graph_objects as go

from . import charts as C
from . import data as D


# ---------------------------------------------------------------------------
#  Respuesta del agente
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Lo que el agente devuelve por turno."""
    text: str
    intent: str = ""
    chips: list[tuple[str, str]] = field(default_factory=list)  # (label, kind)
    table: Optional[pd.DataFrame] = None
    table_config: Optional[dict] = None
    chart: Optional[go.Figure] = None
    followups: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Normalización y entidades
# ---------------------------------------------------------------------------

_ACCENT_MAP = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")

def normalize(text: str) -> str:
    """Lowercase + strip de acentos + colapso de espacios."""
    text = text.translate(_ACCENT_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


# Estados de Brasil (códigos UF)
BR_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

# Nombres amigables de las 5 categorías que tienen pronóstico
CATEGORIA_SINONIMOS: dict[str, list[str]] = {
    "bed_bath_table":        ["bed", "bath", "cama", "bano", "mesa", "tabla", "ropa de cama"],
    "health_beauty":         ["health", "beauty", "salud", "belleza", "cosmetic"],
    "sports_leisure":        ["sport", "leisure", "deport", "ocio", "fitness"],
    "furniture_decor":       ["furniture", "decor", "mueble", "decoracion", "hogar"],
    "computers_accessories": ["computer", "accessor", "computadora", "tecnologia", "tech", "pc"],
}

CLUSTER_SINONIMOS: dict[str, list[str]] = {
    "Vendedores grandes y confiables": ["grande", "power", "confiable", "top",
                                         "estrella", "mejor", "premium"],
    "Vendedores medianos regionales":  ["mediano", "medio", "regional"],
    "Vendedores pequeños en riesgo":   ["pequeno", "chico", "riesgo", "cola larga",
                                         "inestable", "long tail", "peor"],
}


def detect_state(text_norm: str) -> Optional[str]:
    """Busca un código de estado de Brasil escrito en mayúscula original."""
    # Buscamos en el texto original (con mayúsculas) para no confundir con palabras
    return None  # se delega a detect_state_raw()


def detect_state_raw(text: str) -> Optional[str]:
    """Detecta un código de estado en el texto original (mayúsculas)."""
    for match in re.finditer(r"\b([A-Z]{2})\b", text):
        code = match.group(1)
        if code in BR_STATES:
            return code
    # también permitir "sao paulo" → SP en minúscula
    norm = normalize(text)
    mapping = {
        "sao paulo": "SP", "rio de janeiro": "RJ", "minas gerais": "MG",
        "bahia": "BA", "ceara": "CE", "parana": "PR", "santa catarina": "SC",
        "rio grande do sul": "RS", "pernambuco": "PE", "para": "PA",
        "distrito federal": "DF", "espirito santo": "ES",
    }
    for name, code in mapping.items():
        if name in norm:
            return code
    return None


def _contains_word(text_norm: str, term: str) -> bool:
    """`term` aparece como raíz: empieza en límite de palabra y admite hasta
    3 letras de sufijo (plurales, género). Evita falsos positivos como
    "ocio" dentro de "negocio" porque exigimos límite al inicio.
    """
    return re.search(
        rf"(?<![a-z0-9]){re.escape(term)}[a-z]{{0,3}}(?![a-z0-9])",
        text_norm,
    ) is not None


def detect_categoria(text_norm: str) -> Optional[str]:
    """Detecta una de las 5 categorías con pronóstico."""
    for cat, sinonimos in CATEGORIA_SINONIMOS.items():
        for syn in sinonimos:
            if _contains_word(text_norm, syn):
                return cat
    return None


def detect_cluster(text_norm: str) -> Optional[str]:
    """Detecta una etiqueta de cluster."""
    for label, sinonimos in CLUSTER_SINONIMOS.items():
        for syn in sinonimos:
            if _contains_word(text_norm, syn):
                return label
    return None


# ---------------------------------------------------------------------------
#  Formateadores
# ---------------------------------------------------------------------------

def fmt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", " ")

def fmt_money(v: float) -> str:
    if v >= 1e9: return f"R$ {v/1e9:.2f} B"
    if v >= 1e6: return f"R$ {v/1e6:.2f} M"
    if v >= 1e3: return f"R$ {v/1e3:.1f} K"
    return f"R$ {v:.0f}"

def fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%"

def categoria_humana(c: str) -> str:
    return c.replace("_", " ").title()


# ---------------------------------------------------------------------------
#  Handlers — cada uno arma una AgentResponse
# ---------------------------------------------------------------------------

SUGERENCIAS_BASE = [
    "¿Cómo va el negocio?",
    "¿Qué estados tienen más entregas tardías?",
    "¿Qué tan certero es el sistema para predecir retrasos?",
    "¿En qué categorías hay riesgo de quedarnos sin stock?",
    "¿Qué tipos de vendedores tenemos?",
    "¿Qué se espera vender en salud y belleza?",
]


def handle_saludo(_text: str) -> AgentResponse:
    return AgentResponse(
        intent="saludo",
        text=(
            "Hola, soy el **asistente del tablero de operaciones**. "
            "Puedo conversar contigo sobre las ventas, los retrasos en entregas, "
            "los tipos de vendedores, los avisos de inventario y la certeza de "
            "las predicciones. Hazme una pregunta o usa una de las sugerencias."
        ),
        followups=SUGERENCIAS_BASE,
    )


def handle_ayuda(_text: str) -> AgentResponse:
    return AgentResponse(
        intent="ayuda",
        text=(
            "Estas son las cosas que puedo responder sobre el negocio:\n\n"
            "**Sobre las ventas y los retrasos**\n"
            "- Cómo va el negocio en general\n"
            "- Qué estados tienen más entregas tardías\n"
            "- Qué categorías generan más ingresos\n"
            "- Si los feriados u ofertas afectan los retrasos\n\n"
            "**Sobre los vendedores**\n"
            "- Cuántos vendedores hay y qué tipos\n"
            "- Cuáles son los vendedores con peor cumplimiento\n"
            "- Qué hacer con cada grupo de vendedores\n\n"
            "**Sobre el inventario y el pronóstico**\n"
            "- Qué categorías están en riesgo de quedarse sin producto\n"
            "- Cuánto se espera vender los próximos días\n"
            "- Cuáles son las recomendaciones más urgentes\n\n"
            "**Sobre la certeza del sistema**\n"
            "- Qué tan certero es el modelo de retraso\n"
            "- Qué tan certero es el pronóstico de ventas\n"
            "- Cuánto mejoró respecto a la versión anterior"
        ),
        followups=SUGERENCIAS_BASE,
    )


def handle_resumen_negocio(_text: str) -> AgentResponse:
    k = D.kpi_globales()
    desde, hasta = k["rango_fechas"]
    text = (
        f"Entre **{desde:%b %Y}** y **{hasta:%b %Y}** la operación movió "
        f"**{fmt_int(k['n_pedidos'])} pedidos**, vendidos por "
        f"**{fmt_int(k['n_sellers'])} vendedores** distintos. "
        f"En total se cobraron **{fmt_money(k['gmv'])}** "
        f"(producto + envío). "
        f"De cada 100 pedidos, **{k['tasa_retraso']*100:.2f}** llegaron tarde. "
        f"Hoy hay **{fmt_int(k['n_alertas'])} avisos de inventario** activos "
        f"para los próximos 14 días."
    )
    chips = [
        (f"Pedidos: {fmt_int(k['n_pedidos'])}", "info"),
        (f"Vendedores: {fmt_int(k['n_sellers'])}", "info"),
        (f"Retraso global: {k['tasa_retraso']*100:.2f}%", "warn"),
        (f"Avisos de inventario: {k['n_alertas']}", "alert"),
    ]
    return AgentResponse(
        intent="resumen_negocio",
        text=text,
        chips=chips,
        followups=[
            "¿Qué estados tienen más entregas tardías?",
            "¿Qué categorías generan más ingresos?",
            "¿Qué avisos de inventario tengo hoy?",
        ],
    )


def handle_retraso_global(_text: str) -> AgentResponse:
    k = D.kpi_globales()
    mes_df = D.retraso_por_mes()
    # mes más alto y más bajo
    mes_alto = mes_df.loc[mes_df["tasa_retraso"].idxmax()]
    mes_bajo = mes_df.loc[mes_df["tasa_retraso"].idxmin()]
    text = (
        f"En promedio, el **{k['tasa_retraso']*100:.2f}%** de los pedidos llega tarde. "
        f"El peor mes fue **{mes_alto['aniomes']:%b %Y}** con "
        f"{mes_alto['tasa_retraso']*100:.2f}% de retrasos, y el mejor fue "
        f"**{mes_bajo['aniomes']:%b %Y}** con {mes_bajo['tasa_retraso']*100:.2f}%."
    )
    return AgentResponse(
        intent="retraso_global",
        text=text,
        chart=C.line_tendencia_mensual(mes_df),
        chips=[(f"Promedio: {k['tasa_retraso']*100:.2f}%", "warn")],
        followups=[
            "¿Qué estados tienen más entregas tardías?",
            "¿Los feriados afectan los retrasos?",
            "¿Qué vendedores cumplen peor?",
        ],
    )


def handle_retraso_por_estado(text: str) -> AgentResponse:
    df = D.retraso_por_estado()
    estado = detect_state_raw(text)
    if estado and (df["customer_state"] == estado).any():
        row = df[df["customer_state"] == estado].iloc[0]
        rank = (df.sort_values("tasa_retraso", ascending=False)
                  .reset_index(drop=True))
        pos = rank.index[rank["customer_state"] == estado].tolist()[0] + 1
        kind = "alert" if row["tasa_retraso"] > 0.10 else "warn" if row["tasa_retraso"] > 0.05 else "ok"
        text_out = (
            f"En **{estado}** el {row['tasa_retraso']*100:.2f}% de los pedidos "
            f"llega tarde (sobre {fmt_int(row['pedidos'])} pedidos). "
            f"Ocupa el lugar **#{pos} de {len(rank)}** estados, ordenando del peor al mejor."
        )
        return AgentResponse(
            intent="retraso_por_estado",
            text=text_out,
            chips=[(f"{estado}: {row['tasa_retraso']*100:.2f}%", kind)],
            chart=C.bar_estado_retraso(df, top_n=12),
            followups=[
                f"¿Qué tipos de vendedores hay en {estado}?",
                "¿Qué categorías retrasan más?",
                "Compara con el promedio nacional",
            ],
        )
    top = df.head(5)
    bullets = "\n".join(
        f"- **{r['customer_state']}** — {r['tasa_retraso']*100:.2f}% "
        f"(sobre {fmt_int(r['pedidos'])} pedidos)"
        for _, r in top.iterrows()
    )
    return AgentResponse(
        intent="retraso_por_estado",
        text=(
            "Los estados con **más entregas tardías** son:\n\n"
            f"{bullets}\n\n"
            "Pregúntame por un estado en particular (por ejemplo *¿cómo está SP?*) "
            "para ver su detalle."
        ),
        chart=C.bar_estado_retraso(df, top_n=12),
        followups=[
            "¿Cómo está SP?",
            "¿Cómo está RJ?",
            "¿Cuáles son los estados que mejor entregan?",
        ],
    )


def handle_mejores_estados(_text: str) -> AgentResponse:
    df = D.retraso_por_estado().sort_values("tasa_retraso", ascending=True).head(5)
    bullets = "\n".join(
        f"- **{r['customer_state']}** — {r['tasa_retraso']*100:.2f}% "
        f"(sobre {fmt_int(r['pedidos'])} pedidos)"
        for _, r in df.iterrows()
    )
    return AgentResponse(
        intent="mejores_estados",
        text=(
            "Los estados que **mejor entregan** (menos retrasos) son:\n\n"
            f"{bullets}"
        ),
        followups=[
            "¿Cuáles son los peores estados?",
            "¿A qué estados llega cada tipo de vendedor?",
        ],
    )


def handle_top_categorias(text: str) -> AgentResponse:
    df = D.top_categorias(10)
    text_norm = normalize(text)
    quiere_retraso = any(k in text_norm for k in ["retras", "tarde", "demora"])
    if quiere_retraso:
        df = df.sort_values("tasa_retraso", ascending=False)
        bullets = "\n".join(
            f"- **{categoria_humana(r['product_category_name_english'])}** — "
            f"{r['tasa_retraso']*100:.2f}% de retraso "
            f"({fmt_int(r['items'])} productos vendidos)"
            for _, r in df.head(5).iterrows()
        )
        return AgentResponse(
            intent="top_categorias_retraso",
            text=(
                "Entre las categorías que más venden, estas son las que "
                "**más se retrasan**:\n\n" + bullets
            ),
            followups=[
                "¿Cuáles categorías generan más ingresos?",
                "¿Hay riesgo de stockout en alguna categoría?",
            ],
        )
    bullets = "\n".join(
        f"- **{categoria_humana(r['product_category_name_english'])}** — "
        f"{fmt_money(r['ingresos'])} en ventas"
        for _, r in df.head(5).iterrows()
    )
    return AgentResponse(
        intent="top_categorias",
        text=(
            "Las categorías que **más dinero generan** son:\n\n" + bullets
        ),
        chart=C.bar_categorias(df, "ingresos"),
        followups=[
            "¿Cuáles tienen más retrasos?",
            "¿Qué se espera vender de salud y belleza?",
        ],
    )


_MES_NOMBRE_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}


def handle_estacionalidad_ventas(text: str) -> AgentResponse:
    """Responde 'qué mes/año/temporada se vendió más', con tabla y gráfica."""
    vm = D.ventas_por_mes().copy()
    if vm.empty:
        return handle_fallback(text)
    text_norm = normalize(text)

    # `normalize` quita acentos, así que "año" llega como "ano".
    quiere_anio = (("anio" in text_norm or " ano" in f" {text_norm}")
                    and "mes" not in text_norm)

    if quiere_anio:
        anual = (vm.groupby("anio", as_index=False)
                   .agg(pedidos=("pedidos", "sum"),
                         items=("items", "sum"),
                         ingresos=("ingresos", "sum")))
        mejor = anual.loc[anual["ingresos"].idxmax()]
        peor  = anual.loc[anual["ingresos"].idxmin()]
        bullets = "\n".join(
            f"- **{int(r['anio'])}** — {fmt_money(r['ingresos'])} "
            f"({fmt_int(r['pedidos'])} pedidos)"
            for _, r in anual.sort_values("ingresos", ascending=False).iterrows()
        )
        text_out = (
            f"El año más fuerte fue **{int(mejor['anio'])}** con "
            f"{fmt_money(mejor['ingresos'])} en ventas y "
            f"{fmt_int(mejor['pedidos'])} pedidos. El más bajo fue "
            f"**{int(peor['anio'])}** con {fmt_money(peor['ingresos'])}.\n\n"
            f"Detalle por año:\n\n{bullets}\n\n"
            "Nota: el dataset solo cubre fragmentos de 2016 y septiembre 2018, "
            "por lo que **2017** es el año con cobertura completa."
        )
        return AgentResponse(
            intent="ventas_por_anio",
            text=text_out,
            chips=[(f"Mejor: {int(mejor['anio'])}", "ok"),
                    (f"Peor: {int(peor['anio'])}", "warn")],
            followups=[
                "¿Qué mes se vendió más?",
                "¿Qué categorías generan más ingresos?",
                "¿Cómo va el negocio?",
            ],
        )

    # Default: respuesta mes a mes.
    mejor = vm.loc[vm["ingresos"].idxmax()]
    peor  = vm.loc[vm["ingresos"].idxmin()]
    vmc = D.ventas_por_mes_calendario()
    top_mes_cal = vmc.iloc[0] if not vmc.empty else None
    top5 = vm.sort_values("ingresos", ascending=False).head(5)
    bullets = "\n".join(
        f"- **{r['aniomes']:%b %Y}** — {fmt_money(r['ingresos'])} "
        f"({fmt_int(r['pedidos'])} pedidos)"
        for _, r in top5.iterrows()
    )
    text_out = (
        f"El mes con **más ventas** fue **{mejor['aniomes']:%B %Y}** con "
        f"{fmt_money(mejor['ingresos'])} en ingresos sobre "
        f"{fmt_int(mejor['pedidos'])} pedidos. El más bajo fue "
        f"**{peor['aniomes']:%B %Y}** ({fmt_money(peor['ingresos'])}).\n\n"
        "**Top 5 meses por ventas:**\n\n" + bullets
    )
    if top_mes_cal is not None:
        nombre = _MES_NOMBRE_ES.get(int(top_mes_cal["mes"]), str(int(top_mes_cal["mes"])))
        text_out += (
            f"\n\n**Estacionalidad:** en promedio entre años, **{nombre}** "
            f"es el mes más fuerte ({fmt_money(top_mes_cal['ingresos_promedio'])} "
            "de ingresos promedio)."
        )
    chips = [
        (f"Pico: {mejor['aniomes']:%b %Y}", "ok"),
        (f"Valle: {peor['aniomes']:%b %Y}", "warn"),
    ]
    return AgentResponse(
        intent="estacionalidad_ventas",
        text=text_out,
        chips=chips,
        chart=C.bar_ventas_mensuales(vm),
        followups=[
            "¿Qué año se vendió más?",
            "¿Qué categorías generan más ingresos?",
            "¿Los días de oferta afectan los retrasos?",
        ],
    )


def handle_ventas_por_estado(text: str) -> AgentResponse:
    """Responde 'dónde se vende más' / 'qué estados compran más'."""
    df = D.ventas_por_estado_cliente()
    if df.empty:
        return handle_fallback(text)
    estado = detect_state_raw(text)
    if estado and (df["customer_state"] == estado).any():
        row = df[df["customer_state"] == estado].iloc[0]
        rank = (df.sort_values("ingresos", ascending=False)
                  .reset_index(drop=True))
        pos = rank.index[rank["customer_state"] == estado].tolist()[0] + 1
        text_out = (
            f"En **{estado}** se vendieron {fmt_money(row['ingresos'])} "
            f"sobre {fmt_int(row['pedidos'])} pedidos. Ocupa el lugar "
            f"**#{pos} de {len(rank)}** estados por ingresos."
        )
        return AgentResponse(
            intent="ventas_por_estado",
            text=text_out,
            chips=[(f"{estado}: {fmt_money(row['ingresos'])}", "info")],
            followups=[
                f"¿Cómo entrega {estado}?",
                "¿Cuáles son los top estados en ventas?",
            ],
        )
    top = df.head(5)
    bullets = "\n".join(
        f"- **{r['customer_state']}** — {fmt_money(r['ingresos'])} "
        f"({fmt_int(r['pedidos'])} pedidos)"
        for _, r in top.iterrows()
    )
    text_out = (
        "Los **5 estados que más compran** son:\n\n" + bullets + "\n\n"
        f"**SP** concentra cerca del {df.iloc[0]['ingresos']/df['ingresos'].sum()*100:.0f}% "
        "de los ingresos del país."
    )
    return AgentResponse(
        intent="ventas_por_estado",
        text=text_out,
        followups=[
            "¿Qué estados entregan peor?",
            "¿Qué categorías se venden más?",
            "¿Qué mes se vendió más?",
        ],
    )


def handle_efecto_fechas(_text: str) -> AgentResponse:
    ped = D.load_pedidos()
    media = ped["is_late_delivery"].mean()
    cohortes = [
        ("Promedio general",    media),
        ("En feriado nacional", ped.loc[ped["es_feriado_nacional"] == 1, "is_late_delivery"].mean()),
        ("En Carnaval",         ped.loc[ped["es_carnaval"] == 1, "is_late_delivery"].mean()),
        ("En día de oferta",    ped.loc[ped["es_evento_retail"] == 1, "is_late_delivery"].mean()),
        ("En fin de semana",    ped.loc[ped["es_fin_semana"] == 1, "is_late_delivery"].mean()),
    ]
    bullets = []
    chips = []
    for name, rate in cohortes:
        ratio = rate / media if media else 1.0
        bullets.append(f"- **{name}** — {rate*100:.2f}% (×{ratio:.2f} vs el promedio)")
        kind = "alert" if ratio > 1.2 else "warn" if ratio > 1.1 else "ok" if ratio < 0.95 else "neutral"
        chips.append((f"{name}: ×{ratio:.2f}", kind))
    text = (
        "Así se comportan los retrasos según el tipo de día:\n\n"
        + "\n".join(bullets) + "\n\n"
        "**Lectura:** los días de oferta (Black Friday, Cyber Monday) son los "
        "que más golpean la operación; los feriados y Carnaval también "
        "elevan el retraso por encima del promedio."
    )
    return AgentResponse(
        intent="efecto_fechas",
        text=text,
        chips=chips[1:],  # quitar el "Promedio general"
        followups=[
            "¿Cuáles categorías retrasan más?",
            "¿Qué estados aguantan mejor las ofertas?",
        ],
    )


def handle_tipos_vendedores(_text: str) -> AgentResponse:
    perf = D.load_perfil_clusters().sort_values("n_sellers", ascending=False)
    bullets = []
    for _, r in perf.iterrows():
        bullets.append(
            f"- **{r['etiqueta']}** ({fmt_int(r['n_sellers'])} vendedores) — "
            f"retraso {r['tasa_retraso']*100:.2f}%, "
            f"calificación {r['review_promedio']:.2f}/5, "
            f"llega a {int(r['n_estados_clientes'])} estados en promedio."
        )
    text = (
        f"Los **{fmt_int(int(perf['n_sellers'].sum()))} vendedores** se agrupan "
        f"naturalmente en tres tipos:\n\n" + "\n".join(bullets) + "\n\n"
        "Cada grupo necesita una estrategia distinta de inventario. "
        "Pregúntame por uno en particular para ver la recomendación."
    )
    return AgentResponse(
        intent="tipos_vendedores",
        text=text,
        followups=[
            "¿Qué hago con los vendedores grandes y confiables?",
            "¿Qué hago con los vendedores pequeños en riesgo?",
            "¿Cuáles son los peores vendedores?",
        ],
    )


def handle_recomendacion_cluster(text: str) -> AgentResponse:
    label = detect_cluster(normalize(text))
    perf = D.load_perfil_clusters()
    if label is None:
        return handle_tipos_vendedores(text)
    row = perf[perf["etiqueta"] == label].iloc[0]
    chip_kind = ("ok" if "grandes" in label
                 else "info" if "medianos" in label else "alert")
    text_out = (
        f"**{label}** — {fmt_int(row['n_sellers'])} vendedores en este grupo.\n\n"
        f"- Pedidos en promedio: **{row['n_pedidos']:.0f}**\n"
        f"- Ingresos en promedio: **{fmt_money(row['ingresos_totales'])}**\n"
        f"- Entregas con retraso: **{row['tasa_retraso']*100:.2f}%**\n"
        f"- Calificación: **{row['review_promedio']:.2f}/5**\n"
        f"- Llega a **{int(row['n_estados_clientes'])} estados** en promedio\n\n"
        f"**Estrategia recomendada:** {row['estrategia_reabastecimiento']}"
    )
    return AgentResponse(
        intent="recomendacion_cluster",
        text=text_out,
        chips=[(label, chip_kind)],
        followups=[
            "¿Qué hago con los otros grupos?",
            "¿Quiénes son los vendedores con peor cumplimiento?",
        ],
    )


def handle_peores_vendedores(_text: str) -> AgentResponse:
    clu = D.load_clusters()
    flt = clu[clu["n_pedidos"] >= 20].sort_values("tasa_retraso", ascending=False).head(10)
    tabla = flt[["seller_id", "etiqueta", "n_pedidos", "tasa_retraso",
                 "review_promedio", "ingresos_totales"]].copy()
    tabla["tasa_retraso"] = tabla["tasa_retraso"] * 100
    tabla = tabla.rename(columns={
        "seller_id": "Vendedor",
        "etiqueta":  "Tipo",
        "n_pedidos": "Pedidos",
        "tasa_retraso": "% con retraso",
        "review_promedio": "Calificación",
        "ingresos_totales": "Ingresos (R$)",
    })
    return AgentResponse(
        intent="peores_vendedores",
        text=(
            "Estos son los **10 vendedores con peor cumplimiento** (de quienes "
            "tienen al menos 20 pedidos para ser comparables):"
        ),
        table=tabla,
        table_config={
            "% con retraso":  {"format": "%.2f"},
            "Calificación":   {"format": "%.2f"},
            "Ingresos (R$)": {"format": "%.0f"},
        },
        followups=[
            "¿Qué tipos de vendedores tenemos?",
            "¿En qué categorías retrasan más?",
        ],
    )


def handle_top_vendedores(_text: str) -> AgentResponse:
    clu = D.load_clusters()
    top = clu.sort_values("ingresos_totales", ascending=False).head(10)
    tabla = top[["seller_id", "etiqueta", "n_pedidos", "ingresos_totales",
                 "tasa_retraso", "review_promedio"]].copy()
    tabla["tasa_retraso"] = tabla["tasa_retraso"] * 100
    tabla = tabla.rename(columns={
        "seller_id": "Vendedor",
        "etiqueta":  "Tipo",
        "n_pedidos": "Pedidos",
        "ingresos_totales": "Ingresos (R$)",
        "tasa_retraso": "% con retraso",
        "review_promedio": "Calificación",
    })
    return AgentResponse(
        intent="top_vendedores",
        text="Estos son los **10 vendedores que más facturan**:",
        table=tabla,
        table_config={
            "% con retraso":  {"format": "%.2f"},
            "Calificación":   {"format": "%.2f"},
            "Ingresos (R$)": {"format": "%.0f"},
        },
        followups=[
            "¿Y los peores en cumplimiento?",
            "¿Qué tipos de vendedores hay?",
        ],
    )


def handle_alertas_globales(_text: str) -> AgentResponse:
    al = D.load_alertas()
    resumen = al.groupby(["categoria", "tipo"]).size().unstack(fill_value=0)
    for col in ["STOCKOUT", "SOBRE-STOCK", "OK"]:
        if col not in resumen.columns:
            resumen[col] = 0
    resumen = resumen[["STOCKOUT", "SOBRE-STOCK", "OK"]]
    resumen["Categoría"] = resumen.index.map(categoria_humana)
    resumen = resumen.reset_index(drop=True)
    resumen = resumen.rename(columns={
        "STOCKOUT":    "Riesgo de quedarse sin stock",
        "SOBRE-STOCK": "Riesgo de exceso de stock",
        "OK":          "Días normales",
    })[["Categoría", "Riesgo de quedarse sin stock",
        "Riesgo de exceso de stock", "Días normales"]]

    n_stock = int((al["tipo"] == "STOCKOUT").sum())
    n_sobre = int((al["tipo"] == "SOBRE-STOCK").sum())
    resumen_cal = D.alertas_resumen_calendario()
    n_esperadas = resumen_cal["esperadas"]
    n_anomalas  = resumen_cal["anomalas"]
    text = (
        f"Para los próximos **14 días** hay **{n_stock} avisos de quiebre de stock** "
        f"y **{n_sobre} avisos de exceso**. "
        f"De esos avisos críticos, **{n_esperadas} caen en un evento del calendario** "
        f"(Black Friday, Día de la Madre, Carnaval…) y **{n_anomalas} son anómalos** "
        f"— estos últimos son los que conviene investigar primero. "
        f"El detalle por categoría:"
    )
    chips = [
        (f"Stockout: {n_stock}", "alert"),
        (f"Sobre-stock: {n_sobre}", "warn"),
    ]
    if n_esperadas:
        chips.append((f"Esperados por calendario: {n_esperadas}", "ok"))
    if n_anomalas:
        chips.append((f"Anómalos: {n_anomalas}", "alert"))
    return AgentResponse(
        intent="alertas_globales",
        text=text,
        chips=chips,
        table=resumen,
        followups=[
            "¿Qué tan urgente es bed bath table?",
            "¿Qué avisos son anómalos?",
            "¿Cómo afectan los feriados a las ventas?",
        ],
    )


def handle_alertas_categoria(text: str) -> AgentResponse:
    cat = detect_categoria(normalize(text))
    if cat is None:
        return handle_alertas_globales(text)
    al = D.load_alertas()
    al_cat = al[al["categoria"] == cat].sort_values("fecha")
    if al_cat.empty:
        return AgentResponse(
            intent="alertas_categoria",
            text=f"No hay avisos registrados para **{categoria_humana(cat)}** en los próximos 14 días.",
            followups=["¿Cuáles categorías sí tienen avisos?"],
        )
    n_stock = int((al_cat["tipo"] == "STOCKOUT").sum())
    n_sobre = int((al_cat["tipo"] == "SOBRE-STOCK").sum())
    n_ok    = int((al_cat["tipo"] == "OK").sum())
    criticas = al_cat[al_cat["tipo"] != "OK"]
    n_esperadas = (
        int((criticas["categoria_alerta"] == "ESPERADA_POR_CALENDARIO").sum())
        if "categoria_alerta" in criticas.columns else 0
    )
    n_anomalas = (
        int((criticas["categoria_alerta"] == "ANOMALA_INVESTIGAR").sum())
        if "categoria_alerta" in criticas.columns else 0
    )
    contexto_extra = ""
    if n_esperadas + n_anomalas > 0:
        contexto_extra = (
            f"\n\nDe los avisos críticos, **{n_esperadas}** son esperados por el "
            f"calendario (evento conocido) y **{n_anomalas}** son anómalos "
            f"(sin evento que los justifique — revisar primero)."
        )
    text = (
        f"Para **{categoria_humana(cat)}** los próximos 14 días traen:\n\n"
        f"- **{n_stock} días** con riesgo de quedarse sin stock\n"
        f"- **{n_sobre} días** con riesgo de exceso de inventario\n"
        f"- **{n_ok} días** dentro de lo normal"
        f"{contexto_extra}\n\n"
        "Detalle día por día:"
    )
    cols_tabla = ["fecha", "tipo"]
    if "evento_calendario" in al_cat.columns:
        cols_tabla.append("evento_calendario")
    cols_tabla.append("mensaje")
    tabla = al_cat[cols_tabla].copy()
    tabla["fecha"] = pd.to_datetime(tabla["fecha"]).dt.strftime("%Y-%m-%d")
    if "evento_calendario" in tabla.columns:
        tabla["evento_calendario"] = tabla["evento_calendario"].fillna("—")
    tabla = tabla.rename(columns={
        "fecha":             "Fecha",
        "tipo":              "Tipo de aviso",
        "evento_calendario": "Evento del calendario",
        "mensaje":           "Acción recomendada",
    })
    chips = []
    if n_stock: chips.append((f"Stockout: {n_stock} días", "alert"))
    if n_sobre: chips.append((f"Sobre-stock: {n_sobre} días", "warn"))
    if n_ok:    chips.append((f"Normal: {n_ok} días", "ok"))
    if n_esperadas: chips.append((f"Esperados: {n_esperadas}", "ok"))
    if n_anomalas:  chips.append((f"Anómalos: {n_anomalas}", "alert"))
    return AgentResponse(
        intent="alertas_categoria",
        text=text,
        chips=chips,
        table=tabla,
        followups=[
            "¿Y qué pasa con las otras categorías?",
            "¿Qué tan certero es el pronóstico para esta categoría?",
        ],
    )


def handle_pronostico_categoria(text: str) -> AgentResponse:
    cat = detect_categoria(normalize(text))
    if cat is None:
        return AgentResponse(
            intent="pronostico_pregunta_categoria",
            text=(
                "¿De qué categoría quieres ver el pronóstico? Las que tienen "
                "predicción son: **salud y belleza**, **deportes**, "
                "**muebles**, **computadoras** y **ropa de cama**."
            ),
            followups=[
                "Pronóstico de salud y belleza",
                "Pronóstico de deportes",
                "Pronóstico de muebles",
            ],
        )
    series = D.load_series_diaria()
    al = D.load_alertas()
    al_cat = al[al["categoria"] == cat]
    metricas = D.load_metricas_series()
    m = metricas[(metricas["categoria"] == cat) &
                  metricas["modelo"].str.contains("SARIMA")]
    error = m["sMAPE_%"].iloc[0] if not m.empty else None
    hist = pd.DataFrame({"fecha": series["fecha"], "demanda": series[cat]})
    ultimos = hist.tail(14)["demanda"].mean()
    text = (
        f"Para **{categoria_humana(cat)}**, en los últimos 14 días se vendieron "
        f"alrededor de **{ultimos:.0f} productos al día**. "
    )
    if error is not None:
        text += (
            f"El error promedio del pronóstico es de **{error:.1f}%** "
            "(probado contra los últimos 30 días reales). "
        )
    if not al_cat.empty:
        text += (
            f"En los próximos 14 días hay **{len(al_cat)} avisos** "
            f"que conviene revisar."
        )
    return AgentResponse(
        intent="pronostico_categoria",
        text=text,
        chart=C.forecast_chart(hist, al_cat, categoria_humana(cat)),
        followups=[
            "¿Qué avisos hay para esta categoría?",
            "¿Cómo va con las otras categorías?",
        ],
    )


def handle_precision_retraso(_text: str) -> AgentResponse:
    _, schema = D.load_model()
    m = schema["metrics_test"]
    base = schema["base_rate"]
    text = (
        "El sistema que predice **si un pedido va a llegar tarde** se evalúa "
        "con datos que no vio durante el entrenamiento. Estos son los "
        "indicadores principales:\n\n"
        f"- **Calidad general** (F1): **{m['f1_1']:.3f}** — antes era 0.230 "
        f"(mejora de **+{(m['f1_1']-0.23)/0.23*100:.0f}%**).\n"
        f"- **Aciertos cuando avisa retraso** (precisión): **{m['precision_1']:.3f}** — "
        "de cada 100 pedidos que el sistema marca como riesgosos, "
        f"acierta en {m['precision_1']*100:.0f}.\n"
        f"- **Retrasos que sí detecta** (cobertura): **{m['recall_1']:.3f}** — "
        f"detecta {m['recall_1']*100:.0f} de cada 100 retrasos reales.\n"
        f"- **Capacidad de distinguir riesgo** (ROC-AUC): **{m['roc_auc']:.3f}**.\n\n"
        f"**Contexto:** el promedio de retraso del negocio es {base*100:.2f}%. "
        "Cuando el sistema avisa, acierta más del doble de las veces que "
        "lo haría una alerta hecha al azar."
    )
    return AgentResponse(
        intent="precision_retraso",
        text=text,
        chips=[
            (f"Calidad: {m['f1_1']:.3f}", "info"),
            (f"Aciertos: {m['precision_1']*100:.0f}%", "ok"),
            (f"Cobertura: {m['recall_1']*100:.0f}%", "warn"),
        ],
        followups=[
            "¿Qué factores pesan más para predecir retraso?",
            "¿Cuánto mejoró respecto a la versión 1?",
            "¿Qué tan certero es el pronóstico de ventas?",
        ],
    )


def handle_comparacion_modelos(_text: str) -> AgentResponse:
    cmp = D.load_comparacion_v1v2()
    text = (
        "Comparando la **versión 1** (árbol de decisión simple) con la "
        "**versión 2** (modelo afinado automáticamente):\n\n"
    )
    keep = ["Árbol decisión (v1)", "Árbol GridSearch (v1)",
            "Random Forest (baseline)", "Random Forest (GridSearchCV)"]
    sub = cmp[cmp["modelo"].isin(keep)].copy()
    bullets = []
    for _, r in sub.iterrows():
        bullets.append(
            f"- **{r['modelo']}** — calidad {r['f1_1']:.3f}, "
            f"aciertos {r['precision_1']:.2f}, cobertura {r['recall_1']:.2f}"
        )
    text += "\n".join(bullets) + (
        "\n\n**Conclusión:** la versión 2 acierta casi 3 veces más cuando avisa "
        "de retraso, sin perder demasiada cobertura, lo que la hace mucho más "
        "útil para la operación."
    )
    return AgentResponse(
        intent="comparacion_modelos",
        text=text,
        chart=C.bar_compare_v1v2(sub),
        followups=[
            "¿Qué factores pesan más para predecir retraso?",
            "¿Qué tan certero es en cada categoría?",
        ],
    )


def handle_factores_retraso(_text: str) -> AgentResponse:
    imp = D.load_importancias().head(8)
    nombres = {
        "seller_tasa_retraso_hist": "Historial de retraso del vendedor",
        "delivery_days_estimated":  "Días prometidos de entrega",
        "distancia_km":             "Distancia entre vendedor y cliente",
        "payment_value":            "Monto del pago",
        "seller_review_promedio":   "Calificación promedio del vendedor",
        "dia":                      "Día del mes",
        "mes":                      "Mes del año",
        "payment_installments":     "Cantidad de mensualidades",
        "dia_semana_num":           "Día de la semana",
        "num_items":                "Cantidad de productos en el pedido",
        "anio":                     "Año",
        "seller_cluster":           "Tipo de vendedor",
        "trimestre":                "Trimestre",
        "es_dia_no_laboral":        "Día no laboral (feriado/fin de semana)",
        "es_fin_semana":            "Si fue fin de semana",
        "es_evento_retail":         "Si fue día de oferta (Black Friday...)",
        "es_feriado_nacional":      "Si fue feriado nacional",
        "dias_a_proximo_evento":    "Días hasta el próximo evento",
    }
    bullets = []
    for _, r in imp.iterrows():
        humano = nombres.get(r["feature"], r["feature"])
        bullets.append(f"- **{humano}** — peso {r['importancia']*100:.1f}%")
    text = (
        "Los **factores que más influyen** en que un pedido llegue tarde son:\n\n"
        + "\n".join(bullets) + "\n\n"
        "El factor más relevante es el **historial del vendedor**: si "
        "tradicionalmente entrega tarde, lo más probable es que vuelva a "
        "hacerlo. Por eso conviene priorizar la mejora de los vendedores con "
        "peor cumplimiento."
    )
    return AgentResponse(
        intent="factores_retraso",
        text=text,
        chart=C.bar_importancias(D.load_importancias(), top=10),
        followups=[
            "¿Quiénes son los peores vendedores?",
            "¿Qué tan certero es el sistema?",
        ],
    )


def handle_precision_pronostico(_text: str) -> AgentResponse:
    metricas = D.load_metricas_series()
    sar = metricas[metricas["modelo"].str.contains("SARIMA")].sort_values("sMAPE_%")
    naive = metricas[metricas["modelo"].str.contains("Naïve|Naive", regex=True)]
    bullets = []
    for _, r in sar.iterrows():
        cat = categoria_humana(r["categoria"])
        n_row = naive[naive["categoria"] == r["categoria"]]
        n_err = n_row["sMAPE_%"].iloc[0] if not n_row.empty else None
        delta = (n_err - r["sMAPE_%"]) if n_err is not None else None
        extra = (f" (mejora de {delta:+.1f} puntos vs un método simple)"
                  if delta is not None else "")
        bullets.append(
            f"- **{cat}** — error {r['sMAPE_%']:.1f}%{extra}"
        )
    text = (
        "El **pronóstico de ventas** se evaluó comparando lo predicho contra "
        "los últimos 30 días reales. El error promedio (sMAPE) por categoría:\n\n"
        + "\n".join(bullets) + "\n\n"
        "Un error de 30%–50% es lo esperable en demanda diaria por categoría: "
        "no se trata de adivinar el número exacto, sino la tendencia y los "
        "días extremos para anticipar quiebres."
    )
    return AgentResponse(
        intent="precision_pronostico",
        text=text,
        followups=[
            "¿Qué categorías están en riesgo?",
            "¿Qué tan certero es para predecir retrasos?",
        ],
    )


def handle_acciones_urgentes(_text: str) -> AgentResponse:
    al = D.load_alertas()
    stock = al[al["tipo"] == "STOCKOUT"].sort_values("fecha")
    if stock.empty:
        return AgentResponse(
            intent="acciones_urgentes",
            text="Buenas noticias: no hay avisos de quiebre de stock para los próximos 14 días.",
            chips=[("Sin alertas críticas", "ok")],
        )
    primeros = stock.head(5)
    bullets = []
    for _, r in primeros.iterrows():
        cat = categoria_humana(r["categoria"])
        fecha = pd.to_datetime(r["fecha"]).strftime("%Y-%m-%d")
        bullets.append(f"- **{fecha} · {cat}** — {r['mensaje']}")
    n_total = len(stock)
    text = (
        f"Hay **{n_total} días con riesgo de quedarse sin stock** en los "
        f"próximos 14 días. Las **5 acciones más urgentes** (por fecha):\n\n"
        + "\n".join(bullets)
    )
    return AgentResponse(
        intent="acciones_urgentes",
        text=text,
        chips=[(f"{n_total} alertas críticas", "alert")],
        followups=[
            "¿Qué se espera vender en bed bath table?",
            "¿Qué tan certero es el pronóstico?",
        ],
    )


def handle_cuantos_vendedores(_text: str) -> AgentResponse:
    k = D.kpi_globales()
    perf = D.load_perfil_clusters().sort_values("n_sellers", ascending=False)
    bullets = "\n".join(
        f"- **{r['etiqueta']}**: {fmt_int(r['n_sellers'])}"
        for _, r in perf.iterrows()
    )
    return AgentResponse(
        intent="cuantos_vendedores",
        text=(
            f"En la plataforma hay **{fmt_int(k['n_sellers'])} vendedores activos**. "
            f"Distribuidos por tipo:\n\n{bullets}"
        ),
        followups=[
            "¿Qué hacer con cada grupo?",
            "¿Cuáles son los mejores vendedores?",
        ],
    )


def handle_fallback(text: str) -> AgentResponse:
    q = (text or "").strip()
    if q:
        text_out = (
            f"No encontré una pregunta predefinida que matchee con "
            f"\"{q}\". Las áreas que puedo cubrir son: **ventas y "
            "estacionalidad**, **retrasos en entregas**, **tipos de "
            "vendedores**, **avisos de inventario** y **certeza del "
            "sistema**. Prueba reformular o elige una sugerencia:"
        )
    else:
        text_out = (
            "No estoy seguro de haber entendido la pregunta. Puedes intentar "
            "reformularla o usar una de estas sugerencias:"
        )
    return AgentResponse(
        intent="fallback",
        text=text_out,
        followups=SUGERENCIAS_BASE,
    )


# ---------------------------------------------------------------------------
#  Router por puntaje de palabras clave
# ---------------------------------------------------------------------------

@dataclass
class _IntentRule:
    handler: Callable[[str], AgentResponse]
    keywords: list[str]       # frases o palabras; coincidencia exacta de substring
    must_have: list[str] = field(default_factory=list)  # al menos una de estas
    boost: int = 0            # prioridad base


_RULES: list[_IntentRule] = [
    # Saludos / ayuda ------------------------------------------------------
    _IntentRule(handle_saludo,
                keywords=["hola", "buenos dias", "buenas tardes", "buenas noches",
                          "que tal", "qué tal", "hey", "saludos"],
                boost=2),
    _IntentRule(handle_ayuda,
                keywords=["ayuda", "que puedes", "qué puedes", "que sabes",
                          "qué sabes", "capacidades", "funciones",
                          "que preguntas", "qué preguntas",
                          "para que sirves", "para qué sirves"],
                boost=2),

    # Precisión / modelo ---------------------------------------------------
    _IntentRule(handle_precision_retraso,
                keywords=["certero", "preciso", "precision", "exacto",
                          "que tan bueno", "qué tan bueno",
                          "que tan certero", "qué tan certero",
                          "calidad del modelo", "metricas del modelo",
                          "métricas del modelo", "funciona el modelo",
                          "confiable", "f1", "roc", "auc"],
                must_have=["model", "sistema", "predic", "retras", "tarde"]),
    _IntentRule(handle_precision_pronostico,
                keywords=["precision del pronostico", "precisión del pronóstico",
                          "que tan bueno es el pronostico", "qué tan bueno es el pronóstico",
                          "smape", "error del pronostico", "error del pronóstico",
                          "certero el pronostico", "certero el pronóstico"]),
    _IntentRule(handle_comparacion_modelos,
                keywords=["comparacion", "comparación", "version 1", "versión 1",
                          "version 2", "versión 2", "v1 vs v2", "v1 y v2",
                          "antes y ahora", "mejora del modelo", "cuanto mejoro",
                          "cuánto mejoró"]),
    _IntentRule(handle_factores_retraso,
                keywords=["factores", "que influye", "qué influye", "que pesa",
                          "qué pesa", "importancia", "variables importantes",
                          "que causa", "qué causa", "por que se retrasan",
                          "por qué se retrasan", "razon", "razón"]),

    # Resumen / retraso global ---------------------------------------------
    _IntentRule(handle_resumen_negocio,
                keywords=["resumen", "panorama", "como va el negocio",
                          "cómo va el negocio", "como esta el negocio",
                          "cómo está el negocio", "totales del negocio",
                          "vision general", "visión general", "estado general",
                          "numero generales", "números generales", "kpi",
                          "cuanto se ha vendido", "cuánto se ha vendido",
                          "ventas totales", "cuanto vendimos", "cuánto vendimos",
                          "ingresos totales", "ingreso total"]),
    _IntentRule(handle_retraso_global,
                keywords=["tasa de retraso", "porcentaje de retraso",
                          "cuantas entregas tardias", "cuántas entregas tardías",
                          "retraso global", "retraso general",
                          "como van los retrasos", "cómo van los retrasos",
                          "evolucion de retrasos", "evolución de retrasos"]),

    # Retraso por estado ---------------------------------------------------
    _IntentRule(handle_mejores_estados,
                keywords=["mejores estados", "estados con menos retraso",
                          "donde se entrega mejor", "dónde se entrega mejor",
                          "menos tardio", "menos tardío", "menos retraso",
                          "mejor cumplimiento por estado"]),
    _IntentRule(handle_retraso_por_estado,
                keywords=["estado", "estados", "region", "región", "uf",
                          "sao paulo", "rio de janeiro", "bahia"],
                must_have=["retras", "tard", "demor", "entreg", "cumplim", "como esta", "cómo está"]),

    # Categorías -----------------------------------------------------------
    _IntentRule(handle_top_categorias,
                keywords=["categoria", "categorias", "categoría", "categorías",
                          "que se vende", "qué se vende", "mas vendidos",
                          "más vendidos", "top productos", "ingresos por categoria",
                          "ingresos por categoría", "productos que mas",
                          "productos que más", "que productos", "qué productos"]),

    # Estacionalidad / temporal -------------------------------------------
    _IntentRule(handle_estacionalidad_ventas,
                keywords=["que mes", "qué mes", "cual mes", "cuál mes",
                          "mejor mes", "peor mes", "mes con mas",
                          "mes con más", "mes con mayor", "mes con menor",
                          "temporada", "estacionalidad", "epoca del año",
                          "época del año", "cuando se vende", "cuándo se vende",
                          "que año", "qué año", "cual año", "cuál año",
                          "mejor año", "peor año", "por año", "anual",
                          "mes a mes", "evolucion de ventas",
                          "evolución de ventas", "tendencia de ventas",
                          "tendencia mensual", "evolucion mensual",
                          "evolución mensual", "ventas mensuales",
                          "ventas por mes", "ingresos mensuales",
                          "ingresos por mes",
                          "pico de ventas", "valle de ventas"],
                must_have=["mes", "año", "anio", "temporad", "estacional",
                            "epoca", "época", "tendencia", "evolucion",
                            "evolución", "anual", "pico", "valle",
                            "mensual"],
                boost=2),

    # Ventas por estado del cliente ---------------------------------------
    _IntentRule(handle_ventas_por_estado,
                keywords=["donde se vende", "dónde se vende",
                          "donde vendemos", "dónde vendemos",
                          "que estado compra", "qué estado compra",
                          "estados que mas compran", "estados que más compran",
                          "estados compran", "estados que compran",
                          "estados con mas ventas", "estados con más ventas",
                          "ingresos por estado", "ventas por estado",
                          "estado con mas ventas", "estado con más ventas",
                          "que estado vende", "qué estado vende",
                          "que estados venden", "qué estados venden"],
                boost=1),

    # Efecto fechas --------------------------------------------------------
    _IntentRule(handle_efecto_fechas,
                keywords=["feriado", "carnaval", "black friday", "cyber monday",
                          "oferta", "fin de semana", "dia especial",
                          "día especial", "afectan los retrasos"]),

    # Vendedores -----------------------------------------------------------
    _IntentRule(handle_cuantos_vendedores,
                keywords=["cuantos vendedores", "cuántos vendedores",
                          "numero de vendedores", "número de vendedores",
                          "total de vendedores"]),
    _IntentRule(handle_peores_vendedores,
                keywords=["peores vendedores", "vendedores con mas retraso",
                          "vendedores con más retraso", "vendedores que entregan tarde",
                          "vendedores mas tardios", "vendedores más tardíos",
                          "peor cumplimiento", "vendedores problematicos",
                          "vendedores problemáticos"]),
    _IntentRule(handle_top_vendedores,
                keywords=["mejores vendedores", "top vendedores", "top sellers",
                          "vendedores mas grandes", "vendedores más grandes",
                          "vendedores que mas venden", "vendedores que más venden",
                          "ranking de vendedores", "vendedores con mas ingresos",
                          "vendedores con más ingresos"]),
    _IntentRule(handle_recomendacion_cluster,
                keywords=["power seller", "power-seller", "estrella", "premium",
                          "mediano regional", "cola larga", "long tail",
                          "que hacer con", "qué hacer con", "estrategia",
                          "recomendacion para", "recomendación para"]),
    _IntentRule(handle_tipos_vendedores,
                keywords=["tipos de vendedores", "grupos de vendedores",
                          "segmentos", "segmentacion", "segmentación",
                          "clusters", "que tipos hay", "qué tipos hay",
                          "como se agrupan los vendedores",
                          "cómo se agrupan los vendedores"]),

    # Inventario / pronóstico ---------------------------------------------
    _IntentRule(handle_acciones_urgentes,
                keywords=["acciones urgentes", "que hago hoy", "qué hago hoy",
                          "lo mas urgente", "lo más urgente", "prioridad",
                          "que tengo que hacer", "qué tengo que hacer",
                          "que reabastecer", "qué reabastecer"]),
    _IntentRule(handle_alertas_categoria,
                keywords=["alerta", "alertas", "aviso", "avisos",
                          "stockout", "quiebre de stock", "sobre stock",
                          "sobre-stock", "sobre estoque", "sin producto",
                          "sin inventario", "se acaba", "exceso de inventario",
                          "sin stock", "quedarnos sin", "riesgo de quedarnos",
                          "stock", "inventario"],
                boost=1),
    _IntentRule(handle_pronostico_categoria,
                keywords=["pronostico", "pronóstico", "prediccion de venta",
                          "predicción de venta", "que se va a vender",
                          "qué se va a vender", "cuanto se va a vender",
                          "cuánto se va a vender", "forecast", "demanda futura",
                          "proximas semanas", "próximas semanas"]),
]


def _score(text_norm: str, rule: _IntentRule) -> int:
    """Cuenta cuántas keywords aparecen en el texto."""
    hits = sum(1 for k in rule.keywords if normalize(k) in text_norm)
    if rule.must_have:
        if not any(normalize(m) in text_norm for m in rule.must_have):
            return 0
    return hits + (rule.boost if hits else 0)


# Vocabularios temáticos para el pre-router con entidades
_VOC_TEMPORAL = ["mes", "meses", "anio", "año", "anios", "años",
                  "temporada", "estacional", "estacionalidad",
                  "epoca", "época", "trimestre", "semestre",
                  "cuando", "cuándo"]
_VOC_VENTAS = ["vent", "vend", "ingreso", "factur", "compra",
                "gmv", "demanda"]
_VOC_DONDE = ["donde", "dónde", "en que estado", "en qué estado",
              "que estado", "qué estado", "que region", "qué región",
              "que estados", "qué estados", "estados que",
              "estados compran", "estados venden", "estado vende",
              "estado compra"]
_VOC_CALIDAD = ["certero", "preciso", "precision", "exact", "que tan bueno",
                "qué tan bueno", "calidad", "confiable", "f1", "roc", "auc",
                "smape", "error", "metric"]
_VOC_FORECAST = ["pronostico", "pronóstico", "prediccion de venta",
                 "predicción de venta", "forecast", "demanda futura",
                 "que se va a vender", "qué se va a vender",
                 "cuanto se va a vender", "cuánto se va a vender",
                 "proximas semanas", "próximas semanas",
                 "que se espera vender", "qué se espera vender",
                 "se espera vender"]
_VOC_INVENTARIO = ["alerta", "aviso", "stock", "inventario", "urgente",
                   "stockout", "reabast", "quiebre"]
_VOC_RETRASO = ["retras", "tard", "demor", "entreg", "cumplim"]
_VOC_MODELO = ["modelo", "sistema", "predic"]
_VOC_PREGUNTA_ESTADO = ["como esta", "como está", "como va", "como van",
                        "que tal", "qué tal", "dime de", "info de",
                        "informacion de", "información de"]
_VOC_ESTRATEGIA = ["que hago", "qué hago", "estrategia", "recomenda",
                   "que hacer", "qué hacer", "como tratar", "cómo tratar",
                   "como manejo", "cómo manejo"]


def _has_any(text_norm: str, vocab: list[str]) -> bool:
    return any(normalize(v) in text_norm for v in vocab)


def route(text: str) -> AgentResponse:
    """Punto de entrada principal del agente.

    Estrategia en dos pasos:
    1. Pre-router por entidades + tema (resuelve ambigüedades como
       "qué tan certero es el pronóstico" vs "qué se espera vender en X").
    2. Si nada disparó, scoring de keywords sobre las reglas de respaldo.
    """
    if not text or not text.strip():
        return handle_fallback(text)
    text_norm = normalize(text)

    estado = detect_state_raw(text)
    categoria = detect_categoria(text_norm)
    cluster = detect_cluster(text_norm)

    es_calidad     = _has_any(text_norm, _VOC_CALIDAD)
    es_forecast    = _has_any(text_norm, _VOC_FORECAST)
    es_inventario  = _has_any(text_norm, _VOC_INVENTARIO)
    es_retraso     = _has_any(text_norm, _VOC_RETRASO)
    es_modelo      = _has_any(text_norm, _VOC_MODELO)
    es_pregunta_estado = _has_any(text_norm, _VOC_PREGUNTA_ESTADO)
    es_estrategia  = _has_any(text_norm, _VOC_ESTRATEGIA)
    es_temporal    = _has_any(text_norm, _VOC_TEMPORAL)
    es_ventas      = _has_any(text_norm, _VOC_VENTAS)
    es_donde       = _has_any(text_norm, _VOC_DONDE)

    # --- Calidad/certeza del sistema --------------------------------------
    if es_calidad and es_forecast:
        return handle_precision_pronostico(text)
    if es_calidad and (es_retraso or es_modelo):
        return handle_precision_retraso(text)

    # --- Estacionalidad / preguntas temporales sobre ventas --------------
    # "qué mes se vende más", "mejor año", "cuándo facturamos más"
    if es_temporal and es_ventas and not es_retraso and not es_forecast \
            and not es_inventario and categoria is None:
        return handle_estacionalidad_ventas(text)
    # "qué mes fue el peor en retrasos" → retraso global ya cubre
    if es_temporal and es_retraso and not es_forecast and not categoria \
            and not estado:
        return handle_retraso_global(text)

    # --- Dónde se vende más / ingresos por estado -------------------------
    if es_donde and es_ventas and not es_retraso:
        return handle_ventas_por_estado(text)
    # "qué se vende en SP" — estado específico + verbo de venta sin retraso
    if estado and es_ventas and not es_retraso and not es_forecast:
        return handle_ventas_por_estado(text)

    # --- Inventario sin categoría específica → alertas globales -----------
    # cubre "en qué categorías hay stockout/sin stock", "qué alertas hay"
    if es_inventario and categoria is None and not es_forecast \
            and not es_retraso:
        return handle_alertas_categoria(text)

    # --- Categoría mencionada explícitamente ------------------------------
    if categoria:
        if es_inventario:
            return handle_alertas_categoria(text)
        if es_forecast or "vend" in text_norm or "demand" in text_norm:
            return handle_pronostico_categoria(text)
        # mención de categoría sola → asumimos que quiere pronóstico/avisos
        return handle_alertas_categoria(text)

    # --- Estado mencionado ------------------------------------------------
    if estado and (es_retraso or es_pregunta_estado):
        return handle_retraso_por_estado(text)

    # --- Cluster + estrategia --------------------------------------------
    if cluster and es_estrategia:
        return handle_recomendacion_cluster(text)

    # --- Fallback: scoring por keywords ----------------------------------
    best_rule: Optional[_IntentRule] = None
    best_score = 0
    for rule in _RULES:
        s = _score(text_norm, rule)
        if s > best_score:
            best_score = s
            best_rule = rule
    if best_rule is None or best_score == 0:
        return handle_fallback(text)
    return best_rule.handler(text)
