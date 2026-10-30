"""Formatos de presentación; conserva los valores numéricos originales."""
import re

DINERO = {
    'capital', 'pnl_bruto', 'ganancia_neta', 'efectivo_retirar',
    'capital_inyectado', 'depositos_llamadas', 'llamada_maxima',
    'caja_minima', 'patrimonio_minimo', 'pnl_bruto_horizonte',
    'capital_requerido_horizonte', 'depositos_llamadas_horizonte',
    'llamada_maxima_horizonte', 'pnl_dia', 'pnl_acumulado',
    'llamada', 'liquidez', 'margen', 'patrimonio',
    'mediana_neta', 'p05_neta', 'p95_neta', 'mediana_efectivo_retirar',
    'p95_capital_requerido', 'p99_capital_requerido', 'p95_llamada_maxima',
}
PRECIOS = {'spot', 'futuro_proxy', 'precio_dolar_final',
           'dolar_llamada_maxima', 'mediana_dolar_final', 'Spot_MXN_USD',
           'p05', 'Mediana', 'p95', 'p05_muestra', 'mediana_muestra', 'p95_muestra'}

def tipo(columna):
    nombre = str(columna)
    if re.fullmatch(r'(p05|mediana|p95)_(spot|caja|margen|patrimonio)',nombre):
        return 'precio' if nombre.endswith('_spot') else 'dinero'
    if nombre.startswith('prob_') or nombre in ('proporcion', 'Proporción'):
        return 'porcentaje'
    base = re.sub(r'_(promedio_muestral|desviacion_muestral|varianza_muestral|mediana|min|max)$', '', nombre)
    if nombre in PRECIOS or base in PRECIOS:
        return 'precio'
    if nombre in DINERO or base in DINERO or nombre.startswith(('VaR', 'ES')):
        return 'dinero'
    return None

def tabla_formateada(tabla):
    formatos = {'dinero': '${:,.2f}', 'precio': '${:,.4f}', 'porcentaje': '{:.2%}'}
    return tabla.style.format({col: formatos[tipo(col)] for col in tabla if tipo(col)}, na_rep='—')

def formato_excel(columna):
    return {'dinero': '"$"#,##0.00;[Red]-"$"#,##0.00',
            'precio': '"$"#,##0.0000;[Red]-"$"#,##0.0000',
            'porcentaje': '0.00%'}.get(tipo(columna))

def valor_formateado(variable, valor):
    import pandas as pd
    if pd.isna(valor):
        return '—'
    formato = {'dinero': '${:,.2f}', 'precio': '${:,.4f}', 'porcentaje': '{:.2%}'}.get(tipo(variable))
    return formato.format(valor) if formato else str(valor)

def estadisticas_formateadas(tabla, columnas):
    stats = ['media','varianza','desviacion','minimo','p05','p25','mediana','p75','p95','p99','maximo']
    estilo = tabla[columnas].style.format({col:'{:,.2f}' for col in stats}, na_rep='—')
    for indice, fila in tabla.iterrows():
        clase = tipo(fila.variable)
        normales = [col for col in stats if col != 'varianza']
        if clase in ('dinero', 'precio'):
            estilo.format('${:,.4f}' if clase == 'precio' else '${:,.2f}', subset=([indice], normales), na_rep='—')
            estilo.format('{:,.4f} (MXN/USD)²' if clase == 'precio' else '{:,.2f} MXN²', subset=([indice], ['varianza']), na_rep='—')
        elif fila.variable in ('capital_suficiente_horizonte', 'cierre_forzoso'):
            estilo.format('{:.2%}', subset=([indice], normales), na_rep='—')
    return estilo

