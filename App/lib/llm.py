"""Integración con Gemini para el agente conversacional.

Política
--------
- Para las preguntas escritas por el usuario, la app primero intenta
  responder con Gemini, inyectando un snapshot textual de los datos del
  negocio y el historial de la conversación.
- Gemini puede pedir adjuntar UNA "visualización" predefinida (una de las
  19 respuestas del motor rule-based en :mod:`lib.agent`). En ese caso
  reusamos el chart/tabla/chips del handler correspondiente, pero el
  texto de la respuesta lo escribe el LLM.
- Si no hay clave configurada o la llamada falla, caemos al motor
  rule-based como respaldo determinista.

Configuración
-------------
La clave se lee de ``st.secrets["gemini"]["api_key"]`` (Streamlit
Community Cloud o ``.streamlit/secrets.toml`` local) o, como respaldo,
de la variable de entorno ``GEMINI_API_KEY``. El modelo se puede
sobreescribir con ``st.secrets["gemini"]["model"]`` o ``GEMINI_MODEL``.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import streamlit as st

from . import agent as A
from . import data as D


DEFAULT_MODEL = "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
#  Secrets / env
# ---------------------------------------------------------------------------

def _secret(*path: str) -> Optional[str]:
    try:
        node = st.secrets
        for key in path:
            node = node[key]
        return str(node).strip() or None
    except (KeyError, FileNotFoundError, AttributeError):
        return None


def get_api_key() -> Optional[str]:
    return _secret("gemini", "api_key") or os.environ.get("GEMINI_API_KEY") or None


def get_model_name() -> str:
    return (_secret("gemini", "model")
            or os.environ.get("GEMINI_MODEL")
            or DEFAULT_MODEL)


def is_available() -> bool:
    return get_api_key() is not None


# ---------------------------------------------------------------------------
#  Snapshot textual de los datos (contexto del LLM)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_data_context() -> str:
    """Resumen denso y compacto de KPIs/agregados clave para el system prompt."""
    k = D.kpi_globales()
    estados = D.retraso_por_estado()
    cats = D.top_categorias(10)
    perf = D.load_perfil_clusters()
    al = D.load_alertas()
    metricas = D.load_metricas_series()
    _, schema = D.load_model()
    m = schema["metrics_test"]
    cmp = D.load_comparacion_v1v2()
    imp = D.load_importancias().head(10)
    ped = D.load_pedidos()

    desde, hasta = k["rango_fechas"]
    L: list[str] = []
    L.append(f"# Smart Supply Chain — snapshot del negocio (Nexus Supply / Olist, Brasil)")
    L.append(f"Ventana de datos: {desde:%Y-%m} a {hasta:%Y-%m}")
    L.append("")
    L.append("## KPIs globales")
    L.append(f"- pedidos: {k['n_pedidos']:,}")
    L.append(f"- vendedores activos: {k['n_sellers']:,}")
    L.append(f"- productos distintos: {k['n_productos']:,}")
    L.append(f"- GMV total (producto + envío): R$ {k['gmv']:,.0f}")
    L.append(f"- tasa global de retraso: {k['tasa_retraso']*100:.2f}%")
    L.append(f"- avisos de inventario activos (14 días): {k['n_alertas']}")

    L.append("")
    L.append("## Retraso por estado — 8 peores")
    for _, r in estados.head(8).iterrows():
        L.append(f"- {r['customer_state']}: {r['tasa_retraso']*100:.2f}% sobre "
                 f"{int(r['pedidos']):,} pedidos")
    L.append("## Retraso por estado — 5 mejores")
    for _, r in estados.tail(5).iloc[::-1].iterrows():
        L.append(f"- {r['customer_state']}: {r['tasa_retraso']*100:.2f}% sobre "
                 f"{int(r['pedidos']):,} pedidos")

    L.append("")
    L.append("## Ventas por mes (aniomes)")
    vm = D.ventas_por_mes()
    if not vm.empty:
        top_m = vm.sort_values("ingresos", ascending=False).head(5)
        bot_m = vm.sort_values("ingresos", ascending=True).head(3)
        for _, r in top_m.iterrows():
            L.append(f"- {r['aniomes']:%b %Y}: R$ {r['ingresos']:,.0f} "
                     f"sobre {int(r['pedidos']):,} pedidos")
        L.append(f"Mes pico: {top_m.iloc[0]['aniomes']:%B %Y}")
        L.append(f"Meses más bajos: " + ", ".join(
            f"{r['aniomes']:%b %Y}" for _, r in bot_m.iterrows()))
        vmc = D.ventas_por_mes_calendario()
        if not vmc.empty:
            L.append("## Estacionalidad — promedio por número de mes (1-12)")
            nombres_es = {1:"enero",2:"febrero",3:"marzo",4:"abril",5:"mayo",
                            6:"junio",7:"julio",8:"agosto",9:"septiembre",
                            10:"octubre",11:"noviembre",12:"diciembre"}
            for _, r in vmc.head(6).iterrows():
                L.append(f"- {nombres_es.get(int(r['mes']), int(r['mes']))}: "
                         f"R$ {r['ingresos_promedio']:,.0f} promedio")
        anual = (vm.groupby("anio", as_index=False)
                   .agg(ingresos=("ingresos", "sum"),
                         pedidos=("pedidos", "sum")))
        L.append("## Ventas por año")
        for _, r in anual.sort_values("anio").iterrows():
            L.append(f"- {int(r['anio'])}: R$ {r['ingresos']:,.0f} "
                     f"sobre {int(r['pedidos']):,} pedidos")
        L.append("Nota: 2016 y 2018 son fragmentos; 2017 es el año con "
                 "cobertura completa.")

    L.append("")
    L.append("## Ventas por estado del cliente (top 5)")
    ve = D.ventas_por_estado_cliente().head(5)
    total = ve["ingresos"].sum() if not ve.empty else 0
    for _, r in ve.iterrows():
        share = (r["ingresos"] / total * 100) if total else 0
        L.append(f"- {r['customer_state']}: R$ {r['ingresos']:,.0f} "
                 f"({share:.1f}% del top 5) sobre {int(r['pedidos']):,} pedidos")

    L.append("")
    L.append("## Mes pico de cada categoría top (categoría × mes calendario)")
    nombres_mes = {1:"enero",2:"febrero",3:"marzo",4:"abril",5:"mayo",
                    6:"junio",7:"julio",8:"agosto",9:"septiembre",
                    10:"octubre",11:"noviembre",12:"diciembre"}
    pico_cat = D.mes_pico_por_categoria(5)
    for _, r in pico_cat.iterrows():
        L.append(f"- {r['product_category_name_english']}: mejor en "
                 f"**{nombres_mes.get(int(r['mes']), int(r['mes']))}** "
                 f"(R$ {r['ingresos_promedio']:,.0f} promedio)")

    L.append("")
    L.append("## Top categoría dentro de los meses pico (mes × categoría)")
    vcm = D.ventas_cat_por_mes()
    if not vcm.empty:
        top5_meses = (D.ventas_por_mes()
                          .sort_values("ingresos", ascending=False)
                          .head(5)["aniomes"].tolist())
        for aniomes in top5_meses:
            sub = vcm[vcm["aniomes"] == aniomes]
            if sub.empty:
                continue
            top = sub.sort_values("ingresos", ascending=False).head(3)
            partes = [
                f"{r['product_category_name_english']} (R$ {r['ingresos']:,.0f})"
                for _, r in top.iterrows()
            ]
            L.append(f"- {aniomes:%b %Y}: " + ", ".join(partes))

    L.append("")
    L.append("## Top categorías por estado del cliente (estado × categoría)")
    tcs = D.top_cats_por_estado(5, 3)
    if not tcs.empty:
        for estado, grupo in tcs.groupby("customer_state", sort=False):
            partes = [
                f"{r['product_category_name_english']} (R$ {r['ingresos']:,.0f})"
                for _, r in grupo.iterrows()
            ]
            L.append(f"- {estado}: " + " > ".join(partes))

    L.append("")
    L.append("## Distribución de tipos de vendedor por estado del vendedor")
    cps = D.clusters_por_seller_state(6)
    if not cps.empty:
        for estado, grupo in cps.groupby("seller_state", sort=False):
            total = int(grupo["n_sellers"].sum())
            partes = [f"{r['etiqueta']}={int(r['n_sellers'])}"
                      for _, r in grupo.iterrows()]
            L.append(f"- {estado} ({total} sellers): " + ", ".join(partes))

    L.append("")
    L.append("## Top categorías por ingresos (con su retraso)")
    for _, r in cats.head(10).iterrows():
        L.append(f"- {r['product_category_name_english']}: "
                 f"R$ {r['ingresos']:,.0f} | retraso {r['tasa_retraso']*100:.2f}%")

    L.append("")
    L.append("## Tipos de vendedores (K-Means k=3)")
    for _, r in perf.iterrows():
        L.append(f"- {r['etiqueta']}: {int(r['n_sellers']):,} vendedores | "
                 f"retraso {r['tasa_retraso']*100:.2f}% | "
                 f"review {r['review_promedio']:.2f}/5 | "
                 f"alcance ~{int(r['n_estados_clientes'])} estados | "
                 f"estrategia: {r['estrategia_reabastecimiento']}")

    L.append("")
    L.append("## Picos de ventas asociados a eventos del calendario")
    uplift = D.eventos_uplift_ventas()
    base = D.eventos_uplift_base()
    if not uplift.empty:
        L.append(f"Línea base diaria sin evento: {base['pedidos']:.0f} pedidos · "
                 f"R$ {base['revenue']:,.0f} de revenue.")
        L.append("Eventos que SUBEN las ventas (uplift positivo vs día normal):")
        for _, r in uplift[uplift["uplift_revenue_pct"] > 0].iterrows():
            L.append(f"- {r['nombre_feriado']}: "
                     f"{r['pedidos_prom']:.0f} pedidos ({r['uplift_pedidos_pct']:+.0f}%) · "
                     f"R$ {r['revenue_prom']:,.0f} revenue ({r['uplift_revenue_pct']:+.0f}%) "
                     f"en {int(r['dias_observados'])} días observados")
        bajas = uplift[uplift["uplift_revenue_pct"] <= 0]
        if not bajas.empty:
            L.append("Eventos que BAJAN las ventas (uplift negativo — días no laborales o feriados sin componente comercial):")
            for _, r in bajas.iterrows():
                L.append(f"- {r['nombre_feriado']}: "
                         f"{r['pedidos_prom']:.0f} pedidos ({r['uplift_pedidos_pct']:+.0f}%) · "
                         f"R$ {r['revenue_prom']:,.0f} revenue ({r['uplift_revenue_pct']:+.0f}%)")

    L.append("")
    L.append("## Efecto del calendario sobre la tasa de retraso")
    media = ped["is_late_delivery"].mean()
    cohortes = [
        ("promedio general", media),
        ("feriado nacional", ped.loc[ped["es_feriado_nacional"] == 1,
                                      "is_late_delivery"].mean()),
        ("Carnaval", ped.loc[ped["es_carnaval"] == 1, "is_late_delivery"].mean()),
        ("día de oferta (Black Friday / Cyber)",
            ped.loc[ped["es_evento_retail"] == 1, "is_late_delivery"].mean()),
        ("fin de semana",
            ped.loc[ped["es_fin_semana"] == 1, "is_late_delivery"].mean()),
    ]
    for name, rate in cohortes:
        ratio = rate / media if media else 1.0
        L.append(f"- {name}: {rate*100:.2f}% (×{ratio:.2f} vs promedio)")

    L.append("")
    L.append("## Avisos de inventario (próximos 14 días)")
    counts = al["tipo"].value_counts()
    for tipo in ("STOCKOUT", "SOBRE-STOCK", "OK"):
        L.append(f"- {tipo}: {int(counts.get(tipo, 0))}")
    grp = al.groupby(["categoria", "tipo"]).size().unstack(fill_value=0)
    L.append("Por categoría:")
    for cat in grp.index:
        row = grp.loc[cat]
        parts = [f"{t}={int(row.get(t, 0))}"
                 for t in ("STOCKOUT", "SOBRE-STOCK", "OK")
                 if int(row.get(t, 0)) > 0]
        L.append(f"- {cat}: {', '.join(parts)}")

    L.append("")
    L.append("## Avisos cruzados con el calendario")
    resumen_cal = D.alertas_resumen_calendario()
    L.append(f"- avisos críticos esperados por calendario (caen en un evento o ventana pre-evento): "
             f"{resumen_cal['esperadas']}")
    L.append(f"- avisos críticos anómalos (sin evento que los justifique — prioritarios): "
             f"{resumen_cal['anomalas']}")
    if "categoria_alerta" in al.columns:
        criticas = al[al["tipo"] != "OK"]
        anomalas = criticas[criticas["categoria_alerta"] == "ANOMALA_INVESTIGAR"]
        if not anomalas.empty:
            L.append("Detalle de avisos anómalos (sin evento conocido — investigar primero):")
            for _, r in anomalas.head(8).iterrows():
                fecha_str = r["fecha"].strftime("%Y-%m-%d") if hasattr(r["fecha"], "strftime") else str(r["fecha"])[:10]
                L.append(f"- {r['categoria']} · {fecha_str} · {r['tipo']}")
        con_evento = criticas[criticas["evento_calendario"].notna()] if "evento_calendario" in criticas.columns else pd.DataFrame()
        if not con_evento.empty:
            L.append("Eventos del calendario que disparan avisos:")
            for evento, sub in con_evento.groupby("evento_calendario"):
                cats = sub["categoria"].unique().tolist()
                L.append(f"- {evento}: en {', '.join(cats)} ({len(sub)} avisos)")
    L.append("Cómo usar este corte: si el usuario pregunta por un aviso, "
             "menciona si es 'esperado por calendario' (citar el evento) o "
             "'anómalo' (recomendar investigar campañas no planeadas, problemas "
             "de seller o demand shock).")

    L.append("")
    L.append("## Calidad del modelo de retraso (Random Forest tuneado, v2)")
    L.append(f"- F1 clase 'tarde': {m['f1_1']:.3f} (versión 1: 0.230 → "
             f"mejora +{(m['f1_1']-0.23)/0.23*100:.0f}%)")
    L.append(f"- precisión clase 'tarde': {m['precision_1']:.3f}")
    L.append(f"- recall clase 'tarde': {m['recall_1']:.3f}")
    L.append(f"- ROC-AUC: {m['roc_auc']:.3f}")
    L.append(f"- tasa base del problema: {schema['base_rate']*100:.2f}%")

    L.append("")
    L.append("## Comparación v1 vs v2")
    for _, r in cmp.iterrows():
        L.append(f"- {r['modelo']}: F1={r['f1_1']:.3f} | "
                 f"P={r['precision_1']:.2f} | R={r['recall_1']:.2f}")

    L.append("")
    L.append("## Factores con mayor importancia para predecir retraso (top 10)")
    humanos = {
        "seller_tasa_retraso_hist": "historial de retraso del vendedor",
        "delivery_days_estimated":  "días prometidos de entrega",
        "distancia_km":             "distancia vendedor-cliente",
        "payment_value":            "monto del pago",
        "seller_review_promedio":   "calificación del vendedor",
        "dia":                      "día del mes",
        "mes":                      "mes del año",
        "payment_installments":     "cantidad de mensualidades",
        "dia_semana_num":           "día de la semana",
        "num_items":                "cantidad de productos en el pedido",
        "seller_cluster":           "tipo de vendedor",
    }
    for _, r in imp.iterrows():
        L.append(f"- {humanos.get(r['feature'], r['feature'])}: "
                 f"{r['importancia']*100:.1f}%")

    L.append("")
    L.append("## Pronóstico SARIMA — error sMAPE por categoría")
    sar = metricas[metricas["modelo"].str.contains("SARIMA")].sort_values("sMAPE_%")
    naive = metricas[metricas["modelo"].str.contains("Naïve|Naive", regex=True)]
    for _, r in sar.iterrows():
        n_row = naive[naive["categoria"] == r["categoria"]]
        n_err = n_row["sMAPE_%"].iloc[0] if not n_row.empty else None
        extra = (f" (modelo Naïve: {n_err:.1f}%)" if n_err is not None else "")
        L.append(f"- {r['categoria']}: {r['sMAPE_%']:.1f}%{extra}")

    return "\n".join(L)


# ---------------------------------------------------------------------------
#  Catálogo de visualizaciones que el LLM puede adjuntar
# ---------------------------------------------------------------------------

VIZ_CATALOG: dict[str, str] = {
    "resumen_negocio":        "chips con KPIs del negocio en general",
    "retraso_global":         "tendencia mensual del retraso (gráfica de líneas)",
    "retraso_por_estado":     "barras del retraso por estado (úsalo cuando hablen de un estado o ranking de estados)",
    "mejores_estados":        "lista de los estados con mejor cumplimiento",
    "top_categorias":         "barras de ingresos por categoría",
    "estacionalidad_ventas":  "ventas mensuales con pico y valle (usa cuando pregunten qué mes/año se vendió más, estacionalidad o tendencia de ventas)",
    "ventas_por_estado":      "ranking de estados por ingresos del cliente (úsalo cuando pregunten dónde se vende más, qué estado compra más)",
    "efecto_fechas":          "tasa de retraso por tipo de día (feriado, oferta, etc.)",
    "tipos_vendedores":       "resumen de los 3 grupos de vendedores",
    "recomendacion_cluster":  "detalle y estrategia para UN grupo de vendedores (mencionar el cluster en viz_hint)",
    "peores_vendedores":      "tabla de los 10 vendedores con peor cumplimiento",
    "top_vendedores":         "tabla de los 10 vendedores que más facturan",
    "alertas_globales":       "tabla de avisos de inventario agregados",
    "alertas_categoria":      "detalle día por día de avisos para UNA categoría (mencionar la categoría en viz_hint)",
    "pronostico_categoria":   "histórico + pronóstico para UNA categoría (mencionar la categoría en viz_hint)",
    "precision_retraso":      "métricas del modelo de retraso (F1, precisión, recall, AUC)",
    "comparacion_modelos":    "comparación v1 vs v2 del modelo de retraso",
    "factores_retraso":       "importancia de variables del modelo (gráfica de barras)",
    "precision_pronostico":   "errores sMAPE del pronóstico por categoría",
    "acciones_urgentes":      "acciones inmediatas más urgentes para los próximos días",
    "cuantos_vendedores":     "conteo total y por tipo de vendedores",
}

_INTENT_HANDLERS = {
    "resumen_negocio":        A.handle_resumen_negocio,
    "retraso_global":         A.handle_retraso_global,
    "retraso_por_estado":     A.handle_retraso_por_estado,
    "mejores_estados":        A.handle_mejores_estados,
    "top_categorias":         A.handle_top_categorias,
    "estacionalidad_ventas":  A.handle_estacionalidad_ventas,
    "ventas_por_estado":      A.handle_ventas_por_estado,
    "efecto_fechas":          A.handle_efecto_fechas,
    "tipos_vendedores":       A.handle_tipos_vendedores,
    "recomendacion_cluster":  A.handle_recomendacion_cluster,
    "peores_vendedores":      A.handle_peores_vendedores,
    "top_vendedores":         A.handle_top_vendedores,
    "alertas_globales":       A.handle_alertas_globales,
    "alertas_categoria":      A.handle_alertas_categoria,
    "pronostico_categoria":   A.handle_pronostico_categoria,
    "precision_retraso":      A.handle_precision_retraso,
    "comparacion_modelos":    A.handle_comparacion_modelos,
    "factores_retraso":       A.handle_factores_retraso,
    "precision_pronostico":   A.handle_precision_pronostico,
    "acciones_urgentes":      A.handle_acciones_urgentes,
    "cuantos_vendedores":     A.handle_cuantos_vendedores,
}


# ---------------------------------------------------------------------------
#  System prompt
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    catalog = "\n".join(f"- `{k}`: {v}" for k, v in VIZ_CATALOG.items())
    snapshot = build_data_context()
    return (
        "Eres el asistente conversacional del tablero **Smart Supply Chain** de "
        "Nexus Supply, un marketplace ficticio que opera sobre datos reales de "
        "Olist (Brasil, 2016–2018). Hablas en español neutro, con tono claro y "
        "profesional, dirigido a personas de negocio (no técnicas). Evita jerga "
        "técnica innecesaria. No usas emojis.\n"
        "\n"
        "# Fuente de verdad\n"
        "Tienes abajo un snapshot agregado con todas las métricas y tablas "
        "disponibles. Úsalo como ÚNICA fuente de verdad. NUNCA inventes números: "
        "si una cifra no está en el snapshot, dilo. Usa los porcentajes y "
        "montos exactamente como aparecen.\n"
        "\n"
        "# Visualizaciones\n"
        "Puedes adjuntar UNA visualización predefinida que complemente tu "
        "respuesta, si la pregunta lo amerita. Usa el campo `viz` con uno de los "
        "identificadores de la lista. Déjalo en null cuando la pregunta sea "
        "conversacional (saludo, ayuda, aclaración) o cuando el texto solo "
        "alcance.\n"
        "\n"
        f"Visualizaciones disponibles:\n{catalog}\n"
        "\n"
        "# Cuándo poner algo en viz_hint\n"
        "- Si la pregunta menciona un estado de Brasil (SP, RJ, MG, BA, CE, PR, "
        "SC, RS, PE, PA, DF, ES, AM, AC, AL, AP, GO, MA, MT, MS, PB, PI, RN, "
        "RO, RR, SE, TO) y eliges `retraso_por_estado`, pon el código de dos "
        "letras del estado en `viz_hint`.\n"
        "- Si eliges `alertas_categoria` o `pronostico_categoria`, pon en "
        "`viz_hint` el nombre de la categoría: bed_bath_table, health_beauty, "
        "sports_leisure, furniture_decor o computers_accessories (son las "
        "únicas con pronóstico).\n"
        "- Si eliges `recomendacion_cluster`, pon en `viz_hint` uno de: "
        "'Vendedores grandes y confiables', 'Vendedores medianos regionales' "
        "o 'Vendedores pequeños en riesgo'.\n"
        "- En cualquier otro caso `viz_hint` puede ser null.\n"
        "\n"
        "# Formato de salida\n"
        "Responde SIEMPRE con un JSON válido, sin texto antes ni después, con "
        "esta estructura exacta:\n"
        "{\n"
        '  "text": "respuesta en markdown — puedes usar **negritas**, listas '
        'con guiones y saltos de línea. Sé concreto: 2 a 5 párrafos cortos o '
        'una lista. Cita las cifras del snapshot textualmente.",\n'
        '  "viz": "<id_de_la_lista> o null",\n'
        '  "viz_hint": "<entidad ej. SP, bed_bath_table, Vendedores pequeños en riesgo> '
        'o null"\n'
        "}\n"
        "\n"
        "# Reglas\n"
        "- Para saludos o preguntas tipo '¿qué puedes hacer?', responde "
        "brevemente describiendo las áreas (ventas, retrasos, vendedores, "
        "inventario, certeza de las predicciones) y deja `viz` en null.\n"
        "- Si la pregunta es ambigua, pide una aclaración corta y deja `viz` "
        "en null.\n"
        "- Si la pregunta está fuera del alcance del proyecto, decláralo con "
        "cortesía y propone una pregunta válida.\n"
        "- Nunca expliques que estás siguiendo un formato JSON ni que eres un "
        "modelo — habla siempre como el asistente del tablero.\n"
        "\n"
        "# Snapshot de datos\n"
        f"{snapshot}\n"
    )


# ---------------------------------------------------------------------------
#  Llamada a Gemini
# ---------------------------------------------------------------------------

def _format_history(history: list) -> list[dict]:
    """Convierte el historial del agente al formato `contents` de Gemini.

    `history` es la lista guardada en ``st.session_state.chat_history`` con
    entradas ``(kind, text, resp)``. Sólo tomamos las últimas 4 vueltas para
    mantener el prompt acotado.
    """
    contents: list[dict] = []
    for kind, text, resp in history[-8:]:
        if kind == "user":
            contents.append({"role": "user",
                              "parts": [{"text": text or ""}]})
        else:
            ans = (resp.text if resp is not None else "") or ""
            contents.append({"role": "model",
                              "parts": [{"text": ans}]})
    return contents


def _attach_visualization(viz: Optional[str],
                          viz_hint: Optional[str],
                          text: str,
                          original_query: str) -> A.AgentResponse:
    """Junta el texto del LLM con la viz del handler rule-based."""
    if not viz:
        return A.AgentResponse(text=text, intent="llm:plain")
    handler = _INTENT_HANDLERS.get(viz)
    if handler is None:
        return A.AgentResponse(text=text, intent=f"llm:unknown:{viz}")
    # Enriquecemos el query con el hint para que `detect_state`,
    # `detect_categoria` y `detect_cluster` del handler funcionen.
    hint = (viz_hint or "").strip()
    enriched = f"{original_query} {hint}".strip() if hint else original_query
    try:
        viz_resp = handler(enriched)
    except Exception:
        return A.AgentResponse(text=text, intent=f"llm:{viz}:viz_error")
    return A.AgentResponse(
        text=text,
        intent=f"llm:{viz}",
        chips=viz_resp.chips,
        table=viz_resp.table,
        table_config=viz_resp.table_config,
        chart=viz_resp.chart,
        followups=viz_resp.followups,
    )


def _record_error(reason: str) -> None:
    """Guarda el último motivo de fallback en session_state para mostrarlo en UI."""
    try:
        st.session_state["_llm_last_error"] = reason
    except Exception:
        pass


def answer(query: str, history: list) -> Optional[A.AgentResponse]:
    """Llama a Gemini y devuelve una :class:`AgentResponse`, o ``None`` si falla.

    El llamador debe usar el motor rule-based como respaldo cuando esta
    función devuelva ``None`` (sin API key, sin SDK, error de red, etc.).
    """
    api_key = get_api_key()
    if not api_key:
        _record_error("sin api key configurada (st.secrets['gemini']['api_key'] o GEMINI_API_KEY)")
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _record_error("paquete `google-genai` no está instalado en el entorno")
        return None

    try:
        client = genai.Client(api_key=api_key)
        contents = _format_history(history)
        contents.append({"role": "user", "parts": [{"text": query}]})
        response = client.models.generate_content(
            model=get_model_name(),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_system_prompt(),
                response_mime_type="application/json",
                temperature=0.35,
            ),
        )
        raw = (getattr(response, "text", "") or "").strip()
        if not raw:
            _record_error("respuesta vacía del modelo")
            return None
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _record_error(f"JSON inválido en la respuesta del modelo: {exc}")
        return None
    except Exception as exc:
        _record_error(f"fallo en la llamada a Gemini ({type(exc).__name__}): {exc}")
        return None

    text = (payload.get("text") or "").strip()
    if not text:
        _record_error("la respuesta del modelo no incluyó campo `text` no vacío")
        return None
    # Limpia el registro de error cuando esta llamada SÍ funcionó.
    try:
        st.session_state.pop("_llm_last_error", None)
    except Exception:
        pass
    viz = payload.get("viz")
    if isinstance(viz, str) and viz.lower() in {"null", "none", ""}:
        viz = None
    viz_hint = payload.get("viz_hint")
    if isinstance(viz_hint, str) and viz_hint.lower() in {"null", "none", ""}:
        viz_hint = None
    return _attach_visualization(viz, viz_hint, text, query)


def answer_or_fallback(query: str, history: list) -> A.AgentResponse:
    """Punto de entrada para el chat: LLM primero, rule-based de respaldo."""
    resp = answer(query, history)
    if resp is not None:
        return resp
    return A.route(query)
