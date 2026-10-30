"""Aplicacion interactiva: streamlit run app.py"""
from copy import deepcopy
from datetime import date,timedelta
from io import BytesIO
import hashlib,json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from modelos import cargar,calendario
from motor import simular,calibrar_precios,MODELOS,validar,libro_ruta
from excel import crear_excel,DEFINICIONES
from formato import tabla_formateada, estadisticas_formateadas, valor_formateado

st.set_page_config(page_title='Futuros USD/MXN',page_icon='📈',layout='wide')
st.title('Futuros USD/MXN | Simulador de liquidez y riesgo')
st.caption('Escenario académico prospectivo. Precio del futuro aproximado por el spot; revise precios, márgenes y costos vigentes antes de tomar decisiones.')
base=cargar(); c=deepcopy(base); p=c['operacion']
with st.sidebar:
    st.header('Escenario')
    modelos=st.multiselect('Modelos de precio',list(MODELOS),default=list(MODELOS),format_func=lambda x:{'logistica':'Logística inversa','gbm':'Movimiento browniano geométrico','garch':'ARMA-GARCH-t'}[x])
    capital=st.number_input('Capital disponible ($ MXN)',min_value=1.0,value=float(p['capital']),step=10000.0)
    contratos=st.number_input('Contratos',min_value=1,value=int(p['contratos_base']),step=1)
    direccion=st.selectbox('Posición',['Larga: compra USD','Corta: venta USD'])
    modo=st.radio('Margen',['fijo','porcentaje'],horizontal=True)
    if modo=='fijo':
        mi=st.number_input('Margen inicial por contrato ($ MXN)',min_value=1.0,value=float(p['margen_inicial']))
        mt=st.number_input('Umbral de mantenimiento por contrato ($ MXN)',min_value=0.0,value=float(p['margen_mantenimiento']))
        obj=st.number_input('Objetivo de reposición por contrato ($ MXN)',min_value=1.0,value=float(p['objetivo_reposicion']))
        p.update(margen_inicial=mi,margen_mantenimiento=mt,objetivo_reposicion=obj)
    else:
        frac=st.number_input('Margen inicial / nocional (%)',min_value=.1,max_value=100.0,value=100*float(p['margen_fraccion']),format='%.2f')/100
        maint=st.number_input('Mantenimiento / margen inicial (%)',min_value=.1,max_value=99.9,value=100*float(p['mantenimiento_fraccion']),format='%.4f')/100
        target=st.number_input('Objetivo / margen inicial (%)',min_value=100.0,value=100*float(p['objetivo_fraccion']),format='%.2f')/100
        p.update(margen_fraccion=frac,mantenimiento_fraccion=maint,objetivo_fraccion=target)
    com_e=st.number_input('Comisión entrada ($ MXN/contrato)',min_value=0.0,value=float(p['comision_entrada']))
    com_s=st.number_input('Comisión salida ($ MXN/contrato)',min_value=0.0,value=float(p['comision_salida']))
    slip=st.number_input('Deslizamiento por lado ($ MXN/USD)',min_value=0.0,value=float(p['deslizamiento_por_lado']),format='%.4f')
    st.divider()
    inicio=st.date_input('Fecha de apertura',value=date.fromisoformat(c['inicio']))
    dias=st.number_input('Días naturales',min_value=1,max_value=180,value=int(c['dias']))
    s0=st.number_input('Spot inicial ($ MXN/USD)',min_value=.0001,value=float(c['s0']),format='%.4f')
    rutas=st.number_input('Simulaciones por modelo',min_value=100,max_value=1000000,value=100000,step=10000)
    lote=st.number_input('Rutas por lote',min_value=100,max_value=50000,value=5000,step=1000)
    semilla=st.number_input('Semilla',min_value=0,value=int(c['semilla']))
    st.divider()
    archivo=st.file_uploader('Precios históricos para recalibrar logística y MBG',type=['csv','xlsx'])
    with st.expander('Parámetros avanzados'):
        loc=st.number_input('Logística: ubicación ($ MXN/USD)',value=float(c['logistica']['location']),format='%.7f')
        scale=st.number_input('Logística: escala ($ MXN/USD)',min_value=1e-8,value=float(c['logistica']['scale']),format='%.7f')
        mu=st.number_input('MBG: media log diaria (%)',value=100*float(c['gbm']['media_log_diaria']),format='%.6f')/100
        sigma=st.number_input('MBG: sigma diaria (%)',min_value=1e-8,value=100*float(c['gbm']['sigma_diaria']),format='%.6f')/100
        st.caption('ARMA-GARCH-t conserva juntos coeficientes y estado del ajuste histórico en configuracion.json. Para otros datos, reestime ambos antes de seleccionarlo.')
    ejecutar=st.button('Ejecutar Monte Carlo',type='primary',use_container_width=True)

p.update(capital=capital,contratos_base=contratos,direccion=1 if direccion.startswith('Larga') else -1,comision_entrada=com_e,comision_salida=com_s,deslizamiento_por_lado=slip)
c.update(inicio=inicio.isoformat(),dias=int(dias),s0=s0,semilla=int(semilla))
if archivo is not None:
    try:
        tabla=pd.read_csv(archivo) if archivo.name.lower().endswith('.csv') else pd.read_excel(archivo)
        numcols=[col for col in tabla if pd.api.types.is_numeric_dtype(tabla[col])]
        elegido=st.selectbox('Columna de precios históricos',numcols)
        if elegido:
            ajustada=calibrar_precios(tabla[elegido].dropna().values,c)
            c['logistica']=ajustada['logistica'];c['gbm']=ajustada['gbm']
            st.info('Logística y MBG recalibrados. El spot inicial y GARCH quedan bajo control explícito.')
            if 'garch' in modelos:
                st.error('GARCH usa otro estado histórico. Retírelo de la selección o reestime el modelo y su estado fuera de la app.')
                ejecutar=False
    except Exception as ex: st.error(f'No fue posible leer/calibrar el archivo: {ex}');ejecutar=False
else:
    c['logistica'].update(location=loc,scale=scale)
    c['gbm'].update(media_log_diaria=mu,sigma_diaria=sigma)
if (s0!=base['s0'] or inicio.isoformat()!=base['inicio']) and 'garch' in modelos:
    st.warning('Cambiar S0 o fecha mantiene el estado ARMA-GARCH histórico: escenario hipotético no recalibrado.')
if not modelos: st.warning('Seleccione al menos un modelo.');ejecutar=False
fingerprint=hashlib.sha256(json.dumps({'c':c,'modelos':modelos,'rutas':rutas,'lote':lote,'modo':modo},sort_keys=True).encode()).hexdigest()
if ejecutar:
    try:
        validar(c,contratos,modo)
        progress=st.progress(0,text='Simulando...')
        progreso={m:0 for m in modelos}
        def avance(m,hecho,total):
            progreso[m]=hecho/total
            progress.progress(min(1.,sum(progreso.values())/len(modelos)),text=f'{m}: {hecho:,}/{total:,}')
        with st.spinner('Calculando llamadas, capital y riesgo por trayectoria...'):
            comp,detalle,ejemplos,bandas,estadisticas,cierres,distribuciones=simular(c,modelos,int(rutas),int(lote),int(semilla),modo,contratos,progreso=avance,con_estadisticas=True,con_distribuciones=True)
            xlsx=crear_excel(c,comp,detalle,ejemplos,bandas,estadisticas,cierres,modo=modo)
        progress.empty()
        st.session_state['resultado']=(comp,detalle,ejemplos,bandas,estadisticas,cierres,distribuciones,xlsx,deepcopy(c),fingerprint,modo)
    except Exception as ex: st.error(f'Simulación detenida: {ex}')
if 'resultado' in st.session_state and len(st.session_state['resultado']) != 11:
    del st.session_state['resultado']
    st.info('Se agregaron gráficas con todas las rutas. Ejecute nuevamente la simulación para generarlas.')
if 'resultado' in st.session_state:
    comp,detalle,ejemplos,bandas,estadisticas,cierres,distribuciones,xlsx,cr,old,modo_resultado=st.session_state['resultado']
    if old!=fingerprint: st.warning('Los parámetros cambiaron. Los resultados corresponden a la última ejecución; vuelva a ejecutar para actualizarlos.')
    st.download_button('Descargar Excel',xlsx,file_name='futuros_usdmxn_montecarlo.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.subheader('Comparativa de modelos')
    cols=['modelo','simulaciones','prob_capital_suficiente','prob_cierre_forzoso','mediana_neta','p95_capital_requerido','VaR95_neto','VaR99_neto']
    st.dataframe(tabla_formateada(comp[cols]),use_container_width=True)
    st.caption('Capital suficiente = margen inicial + costo de entrada + llamadas hipotéticas hasta el cierre ≤ capital. VaR positivo indica pérdida; puede ser negativo si el cuantil aún representa ganancia.')
    tabs=st.tabs(['Trayectorias','Liquidez y riesgo','Estadísticos por variable','Diccionario','Supuestos'])
    with tabs[0]:
        fechas,_=calendario(cr)
        visibles=st.slider('Trayectorias visibles por modelo',min_value=1,max_value=5,value=1)
        fig=go.Figure()
        for modelo,paths in ejemplos.items():
            for i,path in enumerate(paths[:visibles]):
                fig.add_trace(go.Scatter(x=fechas,y=path,name=f'{modelo} {i+1}',mode='lines',line={'width':1.8},opacity=.85))
        fig.update_layout(title='Muestra de trayectorias de spot',xaxis_title='Fecha',yaxis_title='MXN/USD',yaxis_tickprefix='$',yaxis_tickformat=',.4f',template='plotly_white')
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Cada línea es un posible camino del dólar. El control solo cambia cuántos ejemplos ves; los estadísticos siguen usando todas las simulaciones.')
    with tabs[1]:
        fig2=go.Figure()
        for _,row in comp.iterrows(): fig2.add_trace(go.Bar(name=row.modelo,x=['Capital suficiente','Cierre forzoso','Pérdida neta'],y=[row.prob_capital_suficiente,row.prob_cierre_forzoso,row.prob_perdida_neta]))
        fig2.update_layout(barmode='group',yaxis_tickformat='.0%',template='plotly_white',title='Probabilidades empíricas')
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown('**Flujo diario de una trayectoria ilustrativa**')
        modelo_flujo=st.selectbox('Modelo del flujo',list(ejemplos),key='modflujo')
        ruta_flujo=st.selectbox('Ruta de muestra',list(range(1,len(ejemplos[modelo_flujo])+1)),key='rutflujo')
        flujo=libro_ruta(ejemplos[modelo_flujo][ruta_flujo-1],cr,modo=modo_resultado)
        fc=go.Figure()
        for col,label in [('liquidez','Caja externa'),('margen','Margen'),('patrimonio','Patrimonio')]:
            fc.add_trace(go.Scatter(x=flujo.fecha,y=flujo[col],name=label,mode='lines+markers'))
        fc.update_layout(title='Caja, margen y patrimonio por día',xaxis_title='Fecha',yaxis_title='MXN',yaxis_tickprefix='$',yaxis_tickformat=',.2f',template='plotly_white')
        st.plotly_chart(fc,use_container_width=True)
        st.dataframe(tabla_formateada(flujo),use_container_width=True)
        st.caption('La distribución del capital requerido se puede comparar entre modelos en la pestaña Estadísticos por variable.')
    with tabs[2]:
        filtro=st.selectbox('Modelo',list(ejemplos),key='filtroestad')
        st.markdown('**Distribución de resultados por ruta**')
        variables=list(estadisticas.variable.drop_duplicates())+['motivo_cierre']
        variable=st.selectbox('Variable para la gráfica',variables,
                              format_func=lambda v: DEFINICIONES.get(v,(v,'',''))[0],
                              key='variable_grafica')
        nombre,unidad,descripcion=DEFINICIONES.get(variable,(variable,'',''))
        st.caption(descripcion)
        grafica=distribuciones[distribuciones.variable==variable].sort_values('orden')
        fig_dist=go.Figure()
        for modelo in ejemplos:
            datos=grafica[grafica.modelo==modelo]
            fig_dist.add_trace(go.Bar(name=modelo,x=datos.grupo,y=datos.rutas,
                                      customdata=datos.proporcion,
                                      hovertemplate='%{x}<br>Rutas: %{y:,}<br>Porcentaje: %{customdata:.2%}<extra>%{fullData.name}</extra>'))
        fig_dist.update_layout(title=f'{nombre} por modelo',barmode='group',
                               xaxis_title=unidad,yaxis_title='Número de rutas',
                               xaxis={'categoryorder':'array','categoryarray':grafica.grupo.drop_duplicates().tolist(),
                                      'tickangle':-50},
                               template='plotly_white',height=570,legend_title='Modelo',
                               margin={'b':140})
        st.plotly_chart(fig_dist,use_container_width=True)
        st.caption('Cada barra cuenta rutas del modelo. Los intervalos son idénticos entre modelos; las colas agrupan extremos y “Sin llamada” conserva los casos sin evento. Se incluyen todas las simulaciones, aunque la inspección individual se limite a 20,000 rutas.')
        tabla=estadisticas[estadisticas.modelo==filtro].copy()
        tabla['nombre']=tabla.variable.map(lambda v: DEFINICIONES.get(v,(v,'',''))[0])
        tabla['unidad']=tabla.variable.map(lambda v: DEFINICIONES.get(v,('','',''))[1])
        visibles_stats=['nombre','unidad','observaciones','sin_dato','media','varianza','desviacion',
                        'minimo','p05','p25','mediana','p75','p95','p99','maximo']
        st.dataframe(estadisticas_formateadas(tabla, visibles_stats),
                     hide_index=True,use_container_width=True)
        st.caption(f'Cada estadístico se calculó con las {int(comp.loc[comp.modelo==filtro,"simulaciones"].iloc[0]):,} rutas del modelo; las observaciones de una llamada solo incluyen rutas donde ocurrió.')
        st.caption('La varianza es muestral (divide entre n−1) y queda en unidades al cuadrado. Para indicadores 0/1, la media es la proporción de rutas donde ocurrió el evento.')
        st.markdown('**Motivo del cierre**')
        st.dataframe(tabla_formateada(cierres[cierres.modelo==filtro][['motivo_cierre','rutas','proporcion']]),
                     hide_index=True,use_container_width=True)
        with st.expander('Inspeccionar una ruta individual'):
            muestra=detalle[detalle.modelo==filtro]
            numero=st.number_input('Número de ruta guardada',min_value=1,max_value=len(muestra),value=1,step=1)
            ruta=muestra[muestra.ruta==numero].iloc[0]
            st.dataframe(pd.DataFrame({'Variable':list(ruta.index),
                                       'Valor':[valor_formateado(k,v) for k,v in ruta.items()]}),hide_index=True,use_container_width=True)
            st.caption('Se conserva un máximo de 20,000 rutas individuales por modelo; los estadísticos usan todas.')
    with tabs[3]:
        st.markdown('Las variables **realizadas** siguen la operación hasta su cierre efectivo. Las terminadas en **_horizonte** siguen una cuenta hipotética hasta el día final, aun cuando faltaría dinero para una llamada.')
        dic=pd.DataFrame([{'variable':v,'nombre':n,'unidad':u,'definición':d}
                          for v,(n,u,d) in DEFINICIONES.items()])
        st.dataframe(dic,hide_index=True,use_container_width=True)
        st.caption('Los días se cuentan desde la apertura: día 0 es la fecha inicial. −1 en el día de llamada significa que no hubo llamada.')
    with tabs[4]:
        st.markdown('**Tres modelos:** logística inversa sobre niveles históricos iid; MBG sobre log rendimientos; ARMA(2,2)-GARCH(1,1) con innovaciones t estandarizadas. Fisher quedó fuera del diseño acordado.')
        st.markdown('**Futuros:** contrato DA de USD 10,000; F=S redondeado al tick como aproximación. Posición cerrada antes del vencimiento. Sin intereses, impuestos, cambios de margen, profundidad ni base de futuros observada.')
        st.markdown('**Cuentas:** P&G diario = dirección × contratos × USD por contrato × cambio del futuro entre sesiones. Se repone el margen si cae bajo mantenimiento; sin caja suficiente, se cierra la posición.')
        st.markdown('**Colas:** ARMA-GARCH-t exponenciado puede producir colas extremas y una media teórica del precio no finita. Por eso se priorizan medianas y cuantiles. El ES es muestral y puede ser inestable.')
else: st.info('Configure el escenario y pulse **Ejecutar Monte Carlo**.')


