"""Motor de Monte Carlo para futuros USD/MXN.

Tres generadores de spot: niveles logisticos iid, movimiento browniano geometrico
(MBG) y ARMA(2,2)-GARCH(1,1)-t. La cotizacion del futuro se aproxima por
F=S redondeado al tick; sustituirla por una base observada al disponer de ella.
Las llamadas solo ocurren en sesiones de mercado. El capital requerido se
calcula en un libro contrafactual sin cierre forzoso para evitar censura.
"""
from copy import deepcopy
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
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


def procesar_lote(precios,c,n,modo,con_series=False):
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
    series=np.empty((2,len(fechas),N),dtype=np.float64) if con_series else None
    if con_series:
        series[0,0,:]=caja
        series[1,0,:]=margen
    prev=f[:,0]
    for j in range(1,len(fechas)):
        if not sesiones[j]:
            if con_series:
                series[0,j,:]=caja
                series[1,j,:]=margen
            continue
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
        if con_series:
            series[0,j,:]=caja
            series[1,j,:]=margen
    if not np.allclose(caja+margen,p['capital']+pnl-ce-cs,atol=1e-6):
        raise AssertionError('No concilia patrimonio al cierre')
    if not np.allclose(caja_hip+margen_hip,p['capital']+pnl_hip-ce,atol=1e-6):
        raise AssertionError('No concilia libro hipotetico')
    cap_req=mi+ce+llamadas_hip
    resultado=pd.DataFrame({
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
    return (resultado,series) if con_series else resultado


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


def estadisticas_por_variable(df, modelo):
    """Estadísticos sobre todas las rutas, incluidos los indicadores binarios.

    El día -1 significa que no hubo llamada y se excluye del resumen de días.
    Las cotizaciones de llamada faltantes se excluyen del resumen de cotizaciones.
    """
    filas = []
    for columna in df.columns:
        if columna in ('modelo', 'ruta', 'motivo_cierre'):
            continue
        if not (pd.api.types.is_numeric_dtype(df[columna]) or pd.api.types.is_bool_dtype(df[columna])):
            continue
        valores = df[columna].to_numpy(dtype=float)
        if columna.startswith('dia_llamada_maxima'):
            valores = np.where(valores < 0, np.nan, valores)
        validos = valores[np.isfinite(valores)]
        fila = {'modelo': modelo, 'variable': columna,
                'observaciones': int(len(validos)),
                'sin_dato': int(len(valores) - len(validos))}
        if len(validos):
            fila.update(media=float(np.mean(validos)),
                        varianza=float(np.var(validos, ddof=1)) if len(validos) > 1 else 0.,
                        desviacion=float(np.std(validos, ddof=1)) if len(validos) > 1 else 0.,
                        minimo=float(np.min(validos)),
                        p05=float(np.quantile(validos, .05)),
                        p25=float(np.quantile(validos, .25)),
                        mediana=float(np.median(validos)),
                        p75=float(np.quantile(validos, .75)),
                        p95=float(np.quantile(validos, .95)),
                        p99=float(np.quantile(validos, .99)),
                        maximo=float(np.max(validos)))
        else:
            fila.update({k: np.nan for k in ('media', 'varianza', 'desviacion',
                                               'minimo', 'p05', 'p25', 'mediana',
                                               'p75', 'p95', 'p99', 'maximo')})
        filas.append(fila)
    return filas


def distribuciones_por_variable(valores_por_modelo, max_barras=35):
    """Cuenta todas las rutas en barras comparables entre modelos.

    Las barras interiores usan límites comunes. Las dos barras de cola reúnen
    los valores extremos y evitan que unas pocas rutas oculten el centro.
    Ausencias y días -1 de llamadas se muestran como "Sin llamada".
    """
    if not valores_por_modelo:
        return pd.DataFrame(columns=['variable','modelo','grupo','orden','rutas','proporcion'])
    modelos=list(valores_por_modelo)
    variables=list(next(iter(valores_por_modelo.values())))
    filas=[]
    for variable in variables:
        series={modelo:valores_por_modelo[modelo][variable] for modelo in modelos}
        if variable=='motivo_cierre':
            categorias=sorted(set().union(*(set(v.dropna().astype(str)) for v in series.values())))
            for modelo,valores in series.items():
                conteos=valores.value_counts(dropna=False)
                for orden,categoria in enumerate(categorias):
                    n=int(conteos.get(categoria,0))
                    filas.append((variable,modelo,categoria,orden,n,n/len(valores)))
            continue

        numericas={modelo:pd.to_numeric(v,errors='coerce').to_numpy(dtype=float)
                   for modelo,v in series.items()}
        validos={modelo:a[np.isfinite(a) & ((a>=0) if variable.startswith('dia_llamada_maxima') else True)]
                 for modelo,a in numericas.items()}
        todos=np.concatenate(list(validos.values()))
        if len(todos)==0:
            etiquetas=[]; limites=None
        elif variable in ('capital_suficiente_horizonte','cierre_forzoso'):
            etiquetas=['No (0)','Sí (1)']; limites=None
        else:
            unicos=np.unique(todos)
            if len(unicos)<=max_barras and np.all(np.isclose(unicos,np.round(unicos))):
                etiquetas=[f'{int(v):,}' for v in unicos]; limites=None
            else:
                bajo,alto=np.quantile(todos,[.005,.995])
                if bajo==alto:
                    bajo,alto=float(np.min(todos)),float(np.max(todos))
                if bajo==alto:
                    etiquetas=[f'{bajo:,.4g}']; limites=None
                else:
                    limites=np.linspace(bajo,alto,max_barras+1)
                    etiquetas=[f'< {bajo:,.4g}']+[f'{l:,.4g} a {r:,.4g}' for l,r in zip(limites[:-1],limites[1:])]+[f'> {alto:,.4g}']
        for modelo,a in numericas.items():
            v=validos[modelo]
            if variable in ('capital_suficiente_horizonte','cierre_forzoso'):
                conteos=[int(np.sum(v==0)),int(np.sum(v==1))]
            elif limites is not None:
                interior=np.histogram(v[(v>=limites[0]) & (v<=limites[-1])],bins=limites)[0]
                conteos=[int(np.sum(v<limites[0]))]+interior.tolist()+[int(np.sum(v>limites[-1]))]
            elif etiquetas:
                conteos=[int(np.sum(v==float(e.replace(',','')))) for e in etiquetas]
            else:
                conteos=[]
            for orden,(etiqueta,n) in enumerate(zip(etiquetas,conteos)):
                filas.append((variable,modelo,etiqueta,orden,n,n/len(a)))
            sin_llamada=int(len(a)-len(v))
            if sin_llamada:
                etiqueta='Sin llamada' if variable.startswith('dia_llamada_maxima') or variable=='dolar_llamada_maxima' else 'Sin dato'
                filas.append((variable,modelo,etiqueta,len(etiquetas),sin_llamada,sin_llamada/len(a)))
    return pd.DataFrame(filas,columns=['variable','modelo','grupo','orden','rutas','proporcion'])


def bandas_de_todas_las_rutas(almacen,modelo,dias):
    """Cuantiles exactos por fecha de spot, caja, margen y patrimonio.

    La cuenta cerrada permanece con margen cero y caja final constante.
    Se procesa una fecha por vez para acotar el uso de memoria.
    """
    filas=[]
    for dia in range(dias+1):
        fila={'modelo':modelo,'dia':dia}
        for canal,nombre in enumerate(('spot','caja','margen')):
            q=np.quantile(almacen[canal,dia,:],[.05,.5,.95])
            fila.update({f'p05_{nombre}':float(q[0]),f'mediana_{nombre}':float(q[1]),
                         f'p95_{nombre}':float(q[2])})
        patrimonio=np.asarray(almacen[1,dia,:],dtype=np.float64)+np.asarray(almacen[2,dia,:],dtype=np.float64)
        q=np.quantile(patrimonio,[.05,.5,.95])
        fila.update(p05_patrimonio=float(q[0]),mediana_patrimonio=float(q[1]),
                    p95_patrimonio=float(q[2]))
        filas.append(fila)
    return filas


def simular(c,modelos=MODELOS,rutas=100000,lote=5000,semilla=None,modo='fijo',contratos=None,muestras=5,progreso=None,con_estadisticas=False,con_distribuciones=False,con_bandas=False):
    """Simula en lotes, conserva resultados por ruta y pocas trayectorias.

    La misma semilla y los mismos parametros de precio reproducen identicas
    rutas si cambian solo los parametros operativos. Lote no cambia la secuencia
    aleatoria si se mantiene el mismo tamano de lote.
    """
    n=c['operacion']['contratos_base'] if contratos is None else contratos
    validar(c,n,modo)
    if rutas<1 or lote<1: raise ValueError('Rutas y tamano de lote deben ser positivos')
    modelo_list=list(modelos); todos=[]; resumidos=[]; estadisticos=[]; cierres=[]; ejemplos={}; bandas=[]
    valores_graficas={} if con_distribuciones else None
    base=c['semilla'] if semilla is None else int(semilla)
    for k,nombre in enumerate(modelo_list):
        if nombre not in MODELOS: raise ValueError(f'Modelo desconocido: {nombre}')
        with (TemporaryDirectory(prefix='futuros_bandas_') if con_bandas else nullcontext()) as carpeta:
            almacen=(np.memmap(Path(carpeta)/'series.bin',dtype=np.float64,mode='w+',
                               shape=(3,c['dias']+1,rutas)) if con_bandas else None)
            partes=[]; sample=[]; done=0; num=0
            while done<rutas:
                cantidad=min(lote,rutas-done)
                # Semillas por lote: estabilidad de rutas ante cambios contables.
                paths=generar(c,nombre,cantidad,base+100000*k+num)
                if con_bandas:
                    result,series=procesar_lote(paths,c,n,modo,con_series=True)
                    almacen[0,:,done:done+cantidad]=paths.T
                    almacen[1:,:,done:done+cantidad]=series
                else:
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
            estadisticos.extend(estadisticas_por_variable(df,nombre))
            conteos=df['motivo_cierre'].value_counts()
            cierres.extend({'modelo':nombre,'motivo_cierre':motivo,'rutas':int(cantidad),
                            'proporcion':float(cantidad/len(df))}
                           for motivo,cantidad in conteos.items())
            if con_distribuciones:
                valores_graficas[nombre]={col:df[col].copy() for col in df.columns if col not in ('modelo','ruta')}
            todos.append(df.head(min(20000,len(df))).copy())
            del df
            ejemplos[nombre]=np.asarray(sample)
            if con_bandas:
                bandas.extend(bandas_de_todas_las_rutas(almacen,nombre,c['dias']))
                del almacen
            else:
                # Se conserva la salida previa para llamadas sin bandas completas.
                q=np.quantile(ejemplos[nombre],[.05,.5,.95],axis=0)
                for j in range(c['dias']+1): bandas.append({'modelo':nombre,'dia':j,'p05_muestra':q[0,j],'mediana_muestra':q[1,j],'p95_muestra':q[2,j]})
    detalle=pd.concat(todos,ignore_index=True)
    comparacion=pd.DataFrame(resumidos)
    resultado_base=(comparacion,detalle,ejemplos,pd.DataFrame(bandas))
    if not con_estadisticas:
        # Compatibilidad durante una actualización de Streamlit: una sesión
        # que todavía ejecuta el app.py anterior espera cuatro elementos.
        return resultado_base
    resultado=resultado_base+(pd.DataFrame(estadisticos),pd.DataFrame(cierres))
    if con_distribuciones:
        return resultado+(distribuciones_por_variable(valores_graficas),)
    return resultado


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


