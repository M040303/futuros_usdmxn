"""Motor de Monte Carlo para futuros USD/MXN.

Tres generadores de spot: niveles logisticos iid, movimiento browniano geometrico
(MBG) y ARMA(2,2)-GARCH(1,1)-t. La cotizacion del futuro se aproxima por
F=S redondeado al tick; sustituirla por una base observada al disponer de ella.
Las llamadas solo ocurren en sesiones de mercado. El capital requerido se
calcula en un libro contrafactual sin cierre forzoso para evitar censura.
"""
from copy import deepcopy
from datetime import date
import numpy as np
import pandas as pd
from modelos import calendario, generar

MODELOS = ('logistica', 'gbm', 'garch')


def validar(c, n, modo):
    p=c['operacion']
    if n < 1 or int(n)!=n: raise ValueError('Numero de contratos invalido')
    if p['capital']<=0 or c['s0']<=0 or c['dias']<1: raise ValueError('Capital, S0 y plazo deben ser positivos')
    if modo not in ('fijo','porcentaje'): raise ValueError('Modo de margen invalido')
    if p['direccion'] not in (-1,1): raise ValueError('Direccion debe ser +1 o -1')
    if p['comision_entrada']<0 or p['comision_salida']<0 or p['deslizamiento_por_lado']<0: raise ValueError('Costos negativos')
    fechas,sesiones=calendario(c)
    if not sesiones[0] or not sesiones[-1]: raise ValueError('La apertura y el cierre deben ser dias de sesion')
    if fechas[-1]>=date.fromisoformat(p['vencimiento_referencia']): raise ValueError('El cierre debe ser previo al vencimiento del contrato de referencia')
    mi,mt,objetivo,costo_e,costo_s=parametros(c,n,modo)
    if not 0<mt<mi or objetivo<mi: raise ValueError('Se requiere 0 < mantenimiento < inicial <= objetivo')
    if mi+costo_e>p['capital']: raise ValueError('Capital insuficiente para abrir la posicion')
    return fechas,sesiones


def parametros(c,n,modo):
    p=c['operacion']; usd=p['usd_por_contrato']
    mi_unit=p['margen_inicial'] if modo=='fijo' else p['margen_fraccion']*usd*c['s0']
    mt_unit=(p['margen_mantenimiento'] if modo=='fijo' else mi_unit*p['mantenimiento_fraccion'])
    objetivo_unit=(p['objetivo_reposicion'] if modo=='fijo' else mi_unit*p['objetivo_fraccion'])
    ce=n*(p['comision_entrada']+usd*p['deslizamiento_por_lado'])
    cs=n*(p['comision_salida']+usd*p['deslizamiento_por_lado'])
    return n*mi_unit,n*mt_unit,n*objetivo_unit,ce,cs


def procesar_lote(precios,c,n,modo):
    """Cuentas vectorizadas. Una ruta puede cerrar por falta de liquidez.

    El libro ilimitado sigue hasta el cierre para estimar el capital que habria
    evitado las llamadas impagas. Es hipotetico: no representa P&G realizado.
    """
    fechas,sesiones=validar(c,n,modo)
    x=np.asarray(precios,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(fechas): raise ValueError('Dimension incorrecta de trayectorias')
    malos=~np.isfinite(x)|(x<=0)
    if malos.any(): raise ArithmeticError(f'{int(malos.sum())} cotizaciones invalidas; no se descartan ni se recortan')
    p=c['operacion']; N=len(x); mi,mt,obj,ce,cs=parametros(c,n,modo)
    tick=p['tick']; f=np.round(x/tick)*tick
    caja=np.full(N,p['capital']-mi-ce); margen=np.full(N,mi)
    caja_hip=caja.copy(); margen_hip=margen.copy()
    activo=np.ones(N,dtype=bool); cerrada=np.full(N,c['dias'],dtype=int)
    motivo=np.full(N,'programado',dtype='U24')
    pnl=np.zeros(N); pnl_hip=np.zeros(N)
    llamadas=np.zeros(N); n_llamadas=np.zeros(N,dtype=int)
    max_llamada=np.zeros(N); dia_max=np.full(N,-1,dtype=int); fx_max=np.full(N,np.nan)
    llamadas_hip=np.zeros(N); n_llamadas_hip=np.zeros(N,dtype=int)
    max_llamada_hip=np.zeros(N); dia_max_hip=np.full(N,-1,dtype=int)
    caja_min=caja.copy(); patrimonio_min=caja+margen
    prev=f[:,0]
    for j in range(1,len(fechas)):
        if not sesiones[j]: continue
        delta=p['direccion']*n*p['usd_por_contrato']*(f[:,j]-prev)
        prev=f[:,j]
        margen_hip+=delta; pnl_hip+=delta
        if j<c['dias']:
            req_hip=np.where(margen_hip<mt,obj-margen_hip,0.)
            nueva=req_hip>max_llamada_hip
            dia_max_hip=np.where(nueva,j,dia_max_hip)
            max_llamada_hip=np.maximum(max_llamada_hip,req_hip)
            llamadas_hip+=req_hip; n_llamadas_hip+=(req_hip>0)
            caja_hip-=req_hip; margen_hip+=req_hip
        delta_real=np.where(activo,delta,0.)
        margen+=delta_real; pnl+=delta_real
        if j<c['dias']:
            req=np.where(activo&(margen<mt),obj-margen,0.)
            impaga=(req>caja)&(req>0)&activo
            pagada=(req>0)&~impaga
            nueva=pagada&(req>max_llamada)
            dia_max=np.where(nueva,j,dia_max)
            fx_max=np.where(nueva,f[:,j],fx_max)
            max_llamada=np.where(pagada,np.maximum(max_llamada,req),max_llamada)
            llamadas+=np.where(pagada,req,0.)
            n_llamadas+=pagada
            caja-=np.where(pagada,req,0.); margen+=np.where(pagada,req,0.)
            cerrar=impaga
            motivo=np.where(cerrar,'sin_liquidez',motivo)
        else: cerrar=activo.copy()
        caja+=np.where(cerrar,margen-cs,0.)
        margen=np.where(cerrar,0.,margen)
        cerrada=np.where(cerrar,j,cerrada)
        activo[cerrar]=False
        caja_min=np.minimum(caja_min,caja)
        patrimonio_min=np.minimum(patrimonio_min,caja+margen)
    if not np.allclose(caja+margen,p['capital']+pnl-ce-cs,atol=1e-6):
        raise AssertionError('No concilia patrimonio al cierre')
    if not np.allclose(caja_hip+margen_hip,p['capital']+pnl_hip-ce,atol=1e-6):
        raise AssertionError('No concilia libro hipotetico')
    cap_req=mi+ce+llamadas_hip
    return pd.DataFrame({
        'pnl_bruto':pnl,'ganancia_neta':caja-p['capital'],'efectivo_retirar':caja,
        'capital_inyectado':mi+ce+llamadas,'depositos_llamadas':llamadas,
        'numero_llamadas':n_llamadas,'llamada_maxima':max_llamada,
        'dia_llamada_maxima':dia_max,'dolar_llamada_maxima':fx_max,
        'caja_minima':caja_min,'patrimonio_minimo':patrimonio_min,
        'dia_cierre':cerrada,'motivo_cierre':motivo,'precio_dolar_final':x[np.arange(N),cerrada],
        'pnl_bruto_horizonte':pnl_hip,'capital_requerido_horizonte':cap_req,
        'depositos_llamadas_horizonte':llamadas_hip,'numero_llamadas_horizonte':n_llamadas_hip,
        'llamada_maxima_horizonte':max_llamada_hip,'dia_llamada_maxima_horizonte':dia_max_hip,
        'capital_suficiente_horizonte':cap_req<=p['capital']+1e-9,
        'cierre_forzoso':motivo=='sin_liquidez',
    })


def resumen(df,c,modelo):
    """Cuantiles empiricos; ES de la muestra, no esperanza teorica."""
    A=c['operacion']['capital']; loss=-df['ganancia_neta'].to_numpy()
    horizon_loss=-df['pnl_bruto_horizonte'].to_numpy()
    row={'modelo':modelo,'simulaciones':len(df),'capital':A,
         'prob_capital_suficiente':float(df['capital_suficiente_horizonte'].mean()),
         'prob_cierre_forzoso':float(df['cierre_forzoso'].mean()),
         'prob_perdida_neta':float((df['ganancia_neta']<0).mean()),
         'mediana_neta':float(df['ganancia_neta'].median()),
         'p05_neta':float(df['ganancia_neta'].quantile(.05)),
         'p95_neta':float(df['ganancia_neta'].quantile(.95)),
         'mediana_efectivo_retirar':float(df['efectivo_retirar'].median()),
         'p95_capital_requerido':float(df['capital_requerido_horizonte'].quantile(.95)),
         'p99_capital_requerido':float(df['capital_requerido_horizonte'].quantile(.99)),
         'p95_llamada_maxima':float(df['llamada_maxima_horizonte'].quantile(.95)),
         'mediana_dolar_final':float(df['precio_dolar_final'].median())}
    for level in (.95,.99):
        q=float(np.quantile(loss,level)); qh=float(np.quantile(horizon_loss,level))
        key=int(100*level)
        row[f'VaR{key}_neto']=q
        row[f'ES{key}_neto_muestral']=float(loss[loss>=q].mean())
        row[f'VaR{key}_mercado_horizonte']=qh
    # Estadisticos descriptivos de cada variable solicitada. En GARCH son
    # estrictamente muestrales; algunas esperanzas teoricas pueden no existir.
    columnas=['pnl_bruto','ganancia_neta','depositos_llamadas','capital_inyectado',
              'efectivo_retirar','llamada_maxima','caja_minima',
              'capital_requerido_horizonte','llamada_maxima_horizonte',
              'precio_dolar_final','dolar_llamada_maxima','dia_llamada_maxima']
    for col in columnas:
        v=df[col].to_numpy(dtype=float)
        if col=='dia_llamada_maxima': v=v[v>=0]
        else: v=v[np.isfinite(v)]
        if len(v):
            row[f'{col}_min']=float(np.min(v))
            row[f'{col}_max']=float(np.max(v))
            row[f'{col}_promedio_muestral']=float(np.mean(v))
            row[f'{col}_desviacion_muestral']=float(np.std(v,ddof=1)) if len(v)>1 else 0.
            row[f'{col}_mediana']=float(np.median(v))
        else:
            for suf in ('min','max','promedio_muestral','desviacion_muestral','mediana'):
                row[f'{col}_{suf}']=np.nan
    return row


def simular(c,modelos=MODELOS,rutas=10000,lote=5000,semilla=None,modo='fijo',contratos=None,muestras=5,progreso=None):
    """Simula en lotes, conserva resultados por ruta y pocas trayectorias.

    La misma semilla y los mismos parametros de precio reproducen identicas
    rutas si cambian solo los parametros operativos. Lote no cambia la secuencia
    aleatoria si se mantiene el mismo tamano de lote.
    """
    n=c['operacion']['contratos_base'] if contratos is None else contratos
    validar(c,n,modo)
    if rutas<1 or lote<1: raise ValueError('Rutas y tamano de lote deben ser positivos')
    modelo_list=list(modelos); todos=[]; resumidos=[]; ejemplos={}; bandas=[]
    base=c['semilla'] if semilla is None else int(semilla)
    for k,nombre in enumerate(modelo_list):
        if nombre not in MODELOS: raise ValueError(f'Modelo desconocido: {nombre}')
        partes=[]; sample=[]; done=0; num=0
        while done<rutas:
            cantidad=min(lote,rutas-done)
            # Semillas por lote: estabilidad de rutas ante cambios contables.
            paths=generar(c,nombre,cantidad,base+100000*k+num)
            result=procesar_lote(paths,c,n,modo)
            result.insert(0,'ruta',np.arange(done+1,done+cantidad+1))
            result.insert(0,'modelo',nombre)
            partes.append(result)
            if len(sample)<muestras:
                take=min(muestras-len(sample),cantidad)
                sample.extend(paths[:take].tolist())
            done+=cantidad; num+=1
            if progreso: progreso(nombre,done,rutas)
        df=pd.concat(partes,ignore_index=True)
        resumidos.append(resumen(df,c,nombre))
        todos.append(df.head(min(20000,len(df))).copy())
        del df
        ejemplos[nombre]=np.asarray(sample)
        # Banda de precios de muestra guardada; no se confunde con la banda de todas las rutas.
        q=np.quantile(ejemplos[nombre],[.05,.5,.95],axis=0)
        for j in range(c['dias']+1): bandas.append({'modelo':nombre,'dia':j,'p05_muestra':q[0,j],'mediana_muestra':q[1,j],'p95_muestra':q[2,j]})
    detalle=pd.concat(todos,ignore_index=True)
    comparacion=pd.DataFrame(resumidos)
    return comparacion,detalle,ejemplos,pd.DataFrame(bandas)


def calibrar_precios(precios,c):
    """Recalibra logistica de niveles y MBG de diferencias logaritmicas.

    No toca ARMA-GARCH: sus coeficientes y estado deben reestimarse juntos.
    """
    from scipy.stats import logistic
    y=np.asarray(precios,dtype=float)
    if len(y)<30 or not np.isfinite(y).all() or (y<=0).any(): raise ValueError('Se requieren al menos 30 precios positivos validos')
    nuevo=deepcopy(c)
    loc,scale=logistic.fit(y)
    rets=np.diff(np.log(y)); sigma=float(np.std(rets,ddof=1)); media=float(np.mean(rets))
    nuevo['logistica'].update(location=float(loc),scale=float(scale))
    nuevo['gbm'].update(media_log_diaria=media,sigma_diaria=sigma,mu_diaria=media+sigma*sigma/2)
    nuevo['s0']=float(y[-1])
    return nuevo


def libro_ruta(precios,c,n=None,modo='fijo'):
    """Flujo diario auditable de una trayectoria individual.

    Una llamada transfiere caja a margen; no es perdida. En dias sin sesion no
    se marca a mercado. La columna liquidez es caja externa al intermediario.
    """
    n=c['operacion']['contratos_base'] if n is None else n
    fechas,sesiones=validar(c,n,modo);p=c['operacion']
    x=np.asarray(precios,dtype=float)
    if len(x)!=len(fechas) or not np.isfinite(x).all() or (x<=0).any(): raise ValueError('Precios invalidos')
    f=np.round(x/p['tick'])*p['tick']
    mi,mt,obj,ce,cs=parametros(c,n,modo)
    cash=p['capital']-mi-ce; margin=mi; pnl_acum=0.; prev=f[0]; rows=[]
    rows.append({'fecha':fechas[0].isoformat(),'spot':x[0],'futuro_proxy':f[0],'sesion':True,'pnl_dia':0.,'pnl_acumulado':0.,'llamada':0.,'liquidez':cash,'margen':margin,'patrimonio':cash+margin,'estado':'apertura'})
    for j in range(1,len(fechas)):
        dp=0.;call=0.;estado='sin_sesion'
        if sesiones[j]:
            dp=p['direccion']*n*p['usd_por_contrato']*(f[j]-prev)
            prev=f[j];margin+=dp;pnl_acum+=dp;estado='abierto'
            if j==c['dias']: estado='cierre_programado'
            elif margin<mt:
                need=obj-margin
                if need>cash: estado='cierre_sin_liquidez'
                else: call=need;cash-=need;margin+=need;estado='llamada_pagada'
            if estado.startswith('cierre'):
                cash+=margin-cs;margin=0.
        if not np.isclose(cash+margin,p['capital']+pnl_acum-ce-(cs if estado.startswith('cierre') else 0),atol=1e-6): raise AssertionError('No concilia libro')
        rows.append({'fecha':fechas[j].isoformat(),'spot':x[j],'futuro_proxy':f[j],'sesion':bool(sesiones[j]),'pnl_dia':dp,'pnl_acumulado':pnl_acum,'llamada':call,'liquidez':cash,'margen':margin,'patrimonio':cash+margin,'estado':estado})
        if estado.startswith('cierre'): break
    return pd.DataFrame(rows)
