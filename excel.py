"""Excel ejecutivo generado desde los resultados exactos de la simulacion."""
from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.chart import LineChart,Reference
from openpyxl.utils import get_column_letter
from modelos import calendario
from motor import libro_ruta

NAVY='13253B'; BLUE='1570A6'; LIGHT='EAF2F8'; WHITE='FFFFFF'

def crear_excel(c,comparacion,detalle,ejemplos,bandas,max_filas=100000,modo='fijo'):
    wb=Workbook(); ws=wb.active; ws.title='Resumen'
    ws.append(['FUTUROS USD/MXN | COMPARACION MONTE CARLO'])
    ws.append(['Escenario academico prospectivo; cotizacion del futuro aproximada por F = spot'])
    ws.append(['Capital (MXN)',c['operacion']['capital'],'Contratos',c['operacion']['contratos_base']])
    ws.append(['Apertura',c['inicio'],'Dias naturales',c['dias']])
    ws.append(['VaR y ES muestrales; cierre forzoso incluido en el P&G neto. Capital requerido: libro sin cierre forzoso.'])
    ws.append([])
    principales=['modelo','simulaciones','capital','prob_capital_suficiente','prob_cierre_forzoso','prob_perdida_neta','mediana_neta','p05_neta','p95_neta','p95_capital_requerido','p99_capital_requerido','VaR95_neto','VaR99_neto','ES95_neto_muestral','ES99_neto_muestral']
    ws.append(principales)
    for row in comparacion[principales].itertuples(index=False,name=None): ws.append([None if str(v)=='nan' else v for v in row])
    st=wb.create_sheet('Estadisticos')
    st.append(['Modelo','Variable','Estadistico','Valor'])
    for _,row in comparacion.iterrows():
        for col in comparacion.columns:
            if col not in principales:
                suffix=next((x for x in ('_promedio_muestral','_desviacion_muestral','_mediana','_min','_max') if col.endswith(x)),None)
                variable=col[:-len(suffix)] if suffix else col
                estadistico=suffix[1:] if suffix else 'valor'
                st.append([row['modelo'],variable,estadistico,None if str(row[col])=='nan' else row[col]])
    md=wb.create_sheet('Parametros')
    md.append(['PARAMETRO','VALOR'])
    def flat(x,prefix=''):
        for k,v in x.items():
            key=f'{prefix}.{k}' if prefix else k
            if isinstance(v,dict): yield from flat(v,key)
            else: yield key,str(v) if isinstance(v,(list,tuple)) else v
    for row in flat(c): md.append(list(row))
    notes=wb.create_sheet('Notas')
    for line in [
        'Contrato DA: USD 10,000 por contrato. Verificar ficha y tarifa vigente antes de operar.',
        'Futuro proxy: F_t = redondeo(S_t/tick)*tick; no incorpora base, carry ni diferencial de tasas.',
        'MXN 50 por lado y MXN 15,000/11,000 de margen son supuestos del ejercicio, no tarifas o garantias oficiales vigentes.',
        'Modelo logistico: niveles iid; se ajusta a precios historicos, no a rendimientos.',
        'MBG: S_t=S_(t-1)*exp(media_log + sigma*Z_t).',
        'GARCH: ARMA(2,2) para media; h_t=omega+alpha*e_(t-1)^2+beta*h_(t-1); innovacion t estandarizada.',
        'En GARCH-t, la esperanza teorica del precio exponenciado puede ser infinita: comparar medianas y cuantiles.',
        'Margen = margen previo + P&G diario. Si baja del mantenimiento, reponer hasta objetivo.',
        'Si no hay caja para la llamada: cierre forzoso al precio de la sesion y pago de costos de salida.',
        'Capital requerido se calcula siguiendo el libro hipotetico hasta cierre sin limitar la caja.',
        'VaR 95/99 = cuantiles 95/99 de la perdida neta (-P&G neto). ES = promedio muestral sobre el cuantil.',
        'Estas probabilidades dependen de supuestos, ventana de calibracion, datos y semilla. No son garantias.',
        'No se incluyen intereses, impuestos, cambios de margen, liquidez del futuro ni liquidacion fisica.'
    ]: notes.append([line])
    m=wb.create_sheet('Resultados_ruta'); m.append(list(detalle.columns))
    for row in detalle.head(max_filas).itertuples(index=False,name=None):
        m.append([None if str(v)=='nan' else v for v in row])
    if len(detalle)>max_filas: notes.append([f'Resultados_ruta limitado a {max_filas:,} filas de {len(detalle):,}; resumen usa todas las rutas.'])
    fechas,_=calendario(c)
    tr=wb.create_sheet('Trayectorias_muestra')
    tr.append(['Modelo','Ruta_muestra','Fecha','Spot_MXN_USD'])
    for modelo,paths in ejemplos.items():
        for i,path in enumerate(paths,1):
            for d,spot in zip(fechas,path): tr.append([modelo,i,d.isoformat(),float(spot)])
    for modelo,paths in ejemplos.items():
        if len(paths):
            ledger=libro_ruta(paths[0],c,modo=modo)
            sheet=wb.create_sheet('Flujo_'+modelo[:20]);sheet.append(list(ledger.columns))
            for row in ledger.itertuples(index=False,name=None): sheet.append(list(row))
    b=wb.create_sheet('Bandas_muestra');b.append(['Modelo','Dia','p05','Mediana','p95'])
    for row in bandas.itertuples(index=False): b.append(list(row))
    for sheet in wb:
        sheet.freeze_panes='B2'; sheet.sheet_view.showGridLines=False
        for cell in sheet[1]: cell.fill=PatternFill('solid',fgColor=NAVY);cell.font=Font(bold=True,color=WHITE,size=11)
        sheet.row_dimensions[1].height=24
        for col in sheet.columns:
            letter=get_column_letter(col[0].column)
            maxlen=max((len(str(v.value)) for v in list(col)[:1000] if v.value is not None),default=10)
            sheet.column_dimensions[letter].width=min(max(13,maxlen+2),70)
    for cell in ws[7]: cell.fill=PatternFill('solid',fgColor=BLUE);cell.font=Font(bold=True,color=WHITE)
    wb.active=0
    stream=BytesIO();wb.save(stream);return stream.getvalue()
