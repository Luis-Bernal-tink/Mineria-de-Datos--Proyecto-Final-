"""Vista 1 — Visión general / KPIs ejecutivos."""
import streamlit as st

from .. import charts as C
from .. import data as D
from .. import theme as T


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def fmt_money(v: float) -> str:
    if v >= 1e9:
        return f"R$ {v/1e9:.2f}"
    if v >= 1e6:
        return f"R$ {v/1e6:.2f}"
    if v >= 1e3:
        return f"R$ {v/1e3:.1f}"
    return f"R$ {v:.0f}"


def render():
    k = D.kpi_globales()

    # ---- KPI grid -----------------------------------------------------
    money_unit = "B" if k["gmv"] >= 1e9 else ("M" if k["gmv"] >= 1e6 else "K")
    st.markdown(T.kpi_grid(
        T.kpi_card("Pedidos totales",   fmt_int(k["n_pedidos"]),
                    delta=f'{k["rango_fechas"][0].year}-{k["rango_fechas"][1].year}',
                    delta_dir="neutral"),
        T.kpi_card("Productos vendidos",    fmt_int(k["n_items"])),
        T.kpi_card("Ventas totales (producto + envío)", fmt_money(k["gmv"]), unit=money_unit),
        T.kpi_card("Vendedores activos",   fmt_int(k["n_sellers"])),
        T.kpi_card("Productos distintos",  fmt_int(k["n_productos"])),
        T.kpi_card("Entregas con retraso",
                    f'{k["tasa_retraso"]*100:.2f}',
                    unit="%",
                    delta=f'Por producto: {k["tasa_retraso_items"]*100:.2f}%',
                    delta_dir="neutral"),
    ), unsafe_allow_html=True)

    st.markdown(T.callout(
        "El número más importante para el negocio es el "
        "<strong>porcentaje de entregas que llegan tarde</strong>: hoy es "
        f"{k['tasa_retraso']*100:.2f}% de todos los pedidos. En las siguientes "
        "secciones verás dónde ocurren los retrasos, qué vendedores son los "
        "más críticos y qué se espera vender en las próximas dos semanas."
    ), unsafe_allow_html=True)

    st.markdown(T.method_note(
        "Esta sección es un <strong>resumen directo</strong> de la operación: "
        "cuenta pedidos, suma ventas y calcula promedios y porcentajes sobre "
        "los datos reales del negocio. No usa modelos ni estimaciones; "
        "describe lo que ya ocurrió y sirve como punto de partida para las "
        "demás secciones.",
        label="Cómo se obtiene esta sección",
    ), unsafe_allow_html=True)

    # ---- Tendencia mensual -------------------------------------------
    st.markdown(T.section("Pedidos mes a mes y cuántos llegaron tarde",
                           badge="Gráfica 1",
                           meta="evolución en el tiempo"),
                 unsafe_allow_html=True)
    st.plotly_chart(C.line_tendencia_mensual(D.retraso_por_mes()),
                     use_container_width=True, theme=None)

    # ---- Top categorías + Estados -------------------------------------
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(T.section("Categorías que más dinero generan",
                               badge="Gráfica 2",
                               meta="top productos por ingresos"),
                     unsafe_allow_html=True)
        st.plotly_chart(C.bar_categorias(D.top_categorias(10), "ingresos"),
                         use_container_width=True, theme=None)
    with c2:
        st.markdown(T.section("Estados con más entregas tardías",
                               badge="Gráfica 3",
                               meta="por estado del cliente"),
                     unsafe_allow_html=True)
        st.plotly_chart(C.bar_estado_retraso(D.retraso_por_estado(), top_n=12),
                         use_container_width=True, theme=None)

    # ---- Calendario y ventas ------------------------------------------
    uplift = D.eventos_uplift_ventas()
    if not uplift.empty:
        base = D.eventos_uplift_base()
        st.markdown(T.section("Cómo mueven las ventas los días festivos y de oferta",
                               badge="Gráfica 4",
                               meta="diferencia vs día normal"),
                     unsafe_allow_html=True)
        st.markdown(T.method_note(
            "Cada barra compara el promedio diario del evento contra la línea "
            f"base de un día sin evento (<strong>{base['pedidos']:.0f} pedidos</strong> y "
            f"<strong>R$ {base['revenue']:,.0f}</strong>). Los positivos suben las "
            "ventas (Black Friday, Día de la Madre); los negativos las deprimen "
            "(Carnaval, feriados largos). Este es el mismo calendario que el "
            "sistema usa para distinguir picos esperados de anomalías reales en "
            "los avisos de inventario.",
            label="Cómo leerlo",
        ), unsafe_allow_html=True)
        st.plotly_chart(C.bar_uplift_eventos(uplift),
                         use_container_width=True, theme=None)
