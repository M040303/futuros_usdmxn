"""Necesidades de margen hipotéticas por fecha, sobre todas las rutas."""
import numpy as np

def resumir_aportaciones(diario, fechas, modelo, reserva):
    acumulado=np.zeros(diario.shape[1],dtype=float)
    filas=[]
    for j,fecha in enumerate(fechas):
        llamadas=np.asarray(diario[j])
        acumulado+=llamadas
        mediana,p95=np.quantile(acumulado,[.5,.95])
        mediana_dia,p95_dia=np.quantile(llamadas,[.5,.95])
        filas.append({'modelo':modelo,'fecha':fecha.isoformat(),'dia':j,
            'simulaciones':diario.shape[1],
            'aporte_acumulado_mediana':float(mediana),
            'aporte_acumulado_p95':float(p95),
            'aporte_diario_mediana':float(mediana_dia),
            'aporte_diario_p95':float(p95_dia),
            'prob_aporte_diario':float(np.mean(llamadas>0)),
            'reserva_disponible':float(reserva),
            'prob_reserva_suficiente':float(np.mean(acumulado<=reserva+1e-9))})
    return filas

def graficas_aportaciones(tabla):
    import plotly.graph_objects as go
    colores={'logistica':'#ff5733','gbm':'#1570A6','garch':'#8B5CF6'}
    nombres={'logistica':'Logística','gbm':'MBG','garch':'ARMA-GARCH-t'}
    lineas=go.Figure(); barras=go.Figure()
    for modelo,grupo in tabla.groupby('modelo',sort=False):
        color=colores.get(modelo,'#333333');nombre=nombres.get(modelo,modelo)
        for columna,etiqueta,trazo in [('aporte_acumulado_p95','P95','solid'),('aporte_acumulado_mediana','Mediana','dot')]:
            lineas.add_trace(go.Scatter(x=grupo.fecha,y=grupo[columna],name=f'{nombre} · {etiqueta}',
                legendgroup=modelo,mode='lines',line={'color':color,'dash':trazo,'shape':'hv'},
                hovertemplate='%{x}<br>$%{y:,.2f} MXN<extra>%{fullData.name}</extra>'))
        barras.add_trace(go.Bar(x=grupo.fecha,y=grupo.aporte_diario_p95,name=nombre,
            marker_color=color,customdata=grupo.prob_aporte_diario,
            hovertemplate='%{x}<br>P95: $%{y:,.2f} MXN<br>Probabilidad de llamada: %{customdata:.2%}<extra>%{fullData.name}</extra>'))
    reserva=float(tabla.reserva_disponible.iloc[0])
    lineas.add_hline(y=reserva,line_dash='dash',line_color='#263238',
        annotation_text=f'Reserva disponible: ${reserva:,.2f}')
    for fig,titulo in [(lineas,'Aportaciones acumuladas para mantener la posición'),(barras,'Aportación diaria: percentil 95 por fecha')]:
        fig.update_layout(title=titulo,xaxis_title='Fecha',yaxis_title='Aportaciones ($ MXN)',
            yaxis_tickprefix='$',yaxis_tickformat=',.2f',template='plotly_white',hovermode='x unified')
    barras.update_layout(barmode='group')
    return lineas,barras

def mostrar_aportaciones(tabla, comparacion):
    import streamlit as st
    from formato import tabla_formateada
    st.subheader('Aportaciones para mantener la posición')
    st.caption('Cuenta hipotética: paga todas las llamadas hasta el cierre programado, incluso si la reserva se agotaría. Las aportaciones transfieren caja al margen; no son pérdidas adicionales.')
    elegidos=st.multiselect('Modelos para comparar aportaciones',list(tabla.modelo.unique()),
        default=list(tabla.modelo.unique()),key='modelos_aportaciones')
    if not elegidos:
        st.info('Seleccione al menos un modelo para ver las aportaciones.')
        return
    vista=tabla[tabla.modelo.isin(elegidos)]
    lineas,barras=graficas_aportaciones(vista)
    st.plotly_chart(lineas,use_container_width=True)
    st.caption('Línea continua: P95; punteada: mediana. Cada fecha usa todas las simulaciones del modelo. Son percentiles por fecha, no una trayectoria individual ni una banda con cobertura conjunta del 95 %.')
    st.plotly_chart(barras,use_container_width=True)
    st.caption('El P95 diario puede ser cero cuando las llamadas son poco frecuentes. No sume las barras: la suma de percentiles diarios no es el percentil de las aportaciones acumuladas. En días sin sesión y al cierre programado no se repone margen.')
    final=vista.sort_values('dia').groupby('modelo',sort=False).tail(1)
    resumen=final[['modelo','simulaciones','aporte_acumulado_p95','reserva_disponible','prob_reserva_suficiente']].copy()
    resumen['aporte_adicional_p95']=(resumen.aporte_acumulado_p95-resumen.reserva_disponible).clip(lower=0)
    resumen=resumen.merge(comparacion[['modelo','p95_llamada_maxima']],on='modelo',validate='one_to_one')
    etiquetas=['Modelo','Simulaciones','Reserva requerida P95','Reserva disponible',
               'Probabilidad de suficiencia','Faltante de reserva P95','Mayor llamada individual P95']
    st.dataframe(tabla_formateada(resumen).relabel_index(etiquetas,axis=1),hide_index=True,use_container_width=True)
    st.caption('aporte_acumulado_p95: reserva para llamadas de todo el periodo al P95, sin margen inicial ni costos. aporte_adicional_p95: faltante sobre la reserva que ya tiene. p95_llamada_maxima: P95 de la mayor llamada individual de cada ruta. La suficiencia excluye la liquidación final y el costo de salida; coincide con el criterio de capital requerido del simulador.')
    with st.expander('Tabla de aportaciones por fecha'):
        st.dataframe(tabla_formateada(vista),hide_index=True,use_container_width=True)
