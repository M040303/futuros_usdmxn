"""Modelos de USD/MXN y motor contable para un escenario académico.

Ejecutar: python modelos.py --rutas 32
No realiza optimización ni estima probabilidades de riesgo con la muestra de prueba.
"""
from pathlib import Path
import argparse
import csv
import json
from datetime import date, timedelta
import numpy as np
from scipy.stats import norm, t as student_t, logistic

ROOT = Path(__file__).resolve().parent


def cargar():
    return json.loads((ROOT / 'configuracion.json').read_text())


def calendario(c):
    start = date.fromisoformat(c['inicio'])
    fechas = [start + timedelta(days=i) for i in range(c['dias'] + 1)]
    cerrados = set(c['feriados'])
    sesiones = np.array([d.weekday() < 5 and d.isoformat() not in cerrados for d in fechas])
    return fechas, sesiones


def calibrar_binomial(precios):
    """Estima p y dos saltos logarítmicos a partir de precios ordenados.

    Un ensayo Bernoulli representa la dirección de un cambio no nulo.
    La magnitud de cada dirección es su media condicional histórica.
    """
    y=np.asarray(precios,dtype=float)
    if len(y)<30 or not np.isfinite(y).all() or (y<=0).any():
        raise ValueError('Binomial requiere al menos 30 precios históricos positivos y válidos')
    r=np.diff(np.log(y))
    arriba=r[r>0]; abajo=r[r<0]
    if len(arriba)==0 or len(abajo)==0:
        raise ValueError('Binomial requiere tanto subidas como bajadas históricas')
    return {'prob_subida':float(len(arriba)/(len(arriba)+len(abajo))),
            'log_subida':float(arriba.mean()),'log_bajada':float(abajo.mean()),
            'observaciones':int(len(r)),'subidas':int(len(arriba)),
            'bajadas':int(len(abajo)),'sin_cambio':int(np.sum(r==0)),
            'fuente':'Rendimientos logarítmicos de la serie histórica cargada'}


def estadisticos_binomial(c):
    """Momentos exactos del generador binomial bajo sesiones independientes."""
    p=c['binomial']['prob_subida'];u=c['binomial']['log_subida'];d=c['binomial']['log_bajada']
    _,sesiones=calendario(c);n=int(sesiones[1:].sum())
    media=p*u+(1-p)*d
    var=p*(1-p)*(u-d)**2
    factor=p*np.exp(u)+(1-p)*np.exp(d)
    factor_cuadrado=p*np.exp(2*u)+(1-p)*np.exp(2*d)
    media_spot=c['s0']*factor**n
    var_spot=max(0.,c['s0']**2*(factor_cuadrado**n-factor**(2*n)))
    return {'sesiones':n,'subidas_esperadas':n*p,'varianza_subidas':n*p*(1-p),
            'media_log_diaria':media,'varianza_log_diaria':var,
            'desviacion_log_diaria':np.sqrt(var),
            'spot_final_esperado':media_spot,'desviacion_spot_final':np.sqrt(var_spot)}


def generar(c, nombre, rutas=32, semilla=None):
    """Devuelve precios en MXN/USD, con día cero común. Sin recortes de precios."""
    if rutas < 1:
        raise ValueError('Se requiere al menos una trayectoria')
    rng = np.random.default_rng(c['semilla'] if semilla is None else semilla)
    H = c['dias']
    out = np.empty((rutas, H + 1))
    out[:, 0] = c['s0']
    if nombre == 'logistica':
        u = rng.random((rutas, H))
        # Exclusión numérica de U=0; no cambia ni recorta los precios generados.
        u = np.maximum(u, np.nextafter(0., 1.))
        p = c['logistica']
        out[:, 1:] = p['location'] + p['scale'] * (np.log(u) - np.log1p(-u))
    elif nombre == 'binomial':
        p=c['binomial']; prob=p['prob_subida']
        up=p['log_subida']; down=p['log_bajada']
        if not 0<prob<1 or not np.isfinite([up,down]).all() or up<=0 or down>=0:
            raise ValueError('Binomial exige 0 < p < 1, salto de subida positivo y bajada negativa')
        # Cada sesión es un ensayo independiente. En días sin sesión se
        # mantiene el último spot, en vez de inventar una observación nueva.
        _,sesiones=calendario(c)
        for j in range(1,H+1):
            if sesiones[j]:
                salto=np.where(rng.random(rutas)<prob,up,down)
                out[:,j]=out[:,j-1]*np.exp(salto)
            else:
                out[:,j]=out[:,j-1]
    elif nombre == 'gbm':
        p = c['gbm']
        r = p['media_log_diaria'] + p['sigma_diaria'] * rng.standard_normal((rutas, H))
        with np.errstate(over='ignore', under='ignore'):
            out[:, 1:] = c['s0'] * np.exp(np.cumsum(r, axis=1))
    elif nombre == 'garch':
        p = c['garch']['coeficientes']
        state = c['garch']['estado']
        r1, r2 = [np.full(rutas, x) for x in state['r_ultimo_penultimo']]
        e1, e2 = [np.full(rutas, x) for x in state['residuo_ultimo_penultimo']]
        h = np.full(rutas, state['varianza_ultima'])
        logp = np.full(rutas, np.log(c['s0']))
        for j in range(1, H + 1):
            m = p['ar1']*r1 + p['ar2']*r2 + p['ma1']*e1 + p['ma2']*e2
            h = p['omega'] + p['alpha1']*e1**2 + p['beta1']*h
            z = rng.standard_t(p['shape'], rutas) * np.sqrt((p['shape']-2)/p['shape'])
            e = np.sqrt(h)*z
            r = m + e
            logp += r
            with np.errstate(over='ignore', under='ignore'):
                out[:, j] = np.exp(logp)
            r2, r1, e2, e1 = r1, r, e1, e
    else:
        raise ValueError(nombre)
    return out


def pronostico_garch(c):
    """Media del log-rendimiento y varianza de innovación; no media del precio."""
    p = c['garch']['coeficientes']; s = c['garch']['estado']
    r1, r2 = s['r_ultimo_penultimo']; e1, e2 = s['residuo_ultimo_penultimo']
    h = s['varianza_ultima']; rows = []
    for k in range(1, c['dias']+1):
        m = p['ar1']*r1+p['ar2']*r2+p['ma1']*e1+p['ma2']*e2
        h = (p['omega']+p['alpha1']*e1**2+p['beta1']*h if k == 1
             else p['omega']+(p['alpha1']+p['beta1'])*h)
        rows.append((k, m, h, np.sqrt(h)))
        r2, r1, e2, e1 = r1, m, e1, 0.
    return rows


def parametros_margen(c, contratos, modo='fijo'):
    if contratos < 1 or int(contratos) != contratos:
        raise ValueError('Contratos debe ser un entero positivo')
    p = c['operacion']
    unit = p['margen_inicial'] if modo == 'fijo' else p['margen_fraccion']*p['usd_por_contrato']*c['s0']
    if modo not in ('fijo', 'porcentaje'):
        raise ValueError(modo)
    mantenimiento = unit * p['margen_mantenimiento']/p['margen_inicial']
    return contratos*unit, contratos*mantenimiento


def cuenta(futuros, c, contratos=None, modo='fijo'):
    """Libro mayor por una trayectoria, incluidas llamadas y cierre por falta de caja.

    La entrada y salida pagan comisión y deslizamiento explícitos, sin duplicarlos
    en el precio de ejecución. No retira excedentes ni devenga intereses.
    """
    n = c['operacion']['contratos_base'] if contratos is None else contratos
    p = c['operacion']; fechas, sesiones = calendario(c)
    f = np.asarray(futuros, dtype=float)
    if len(f) != len(fechas) or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError('Trayectoria de precios inválida; no se recorta ni descarta silenciosamente')
    initial, maintenance = parametros_margen(c, n, modo)
    coste = n*(p['comision_por_lado']+p['usd_por_contrato']*p['deslizamiento_por_lado'])
    if initial+coste > p['capital']:
        raise ValueError('Capital insuficiente para abrir')
    cash = p['capital']-initial-coste; margin = initial
    prev = f[0]; pnl_total = 0.; calls = 0.; ledger = []
    def row(j, pnl, call, cierre, motivo):
        return {'fecha': fechas[j].isoformat(), 'precio_futuro': float(f[j]),
                'pnl': float(pnl), 'llamada': float(call), 'caja': float(cash),
                'margen': float(margin), 'patrimonio': float(cash+margin),
                'cerrado': cierre, 'motivo': motivo}
    ledger.append(row(0, 0., 0., False, 'apertura'))
    for j in range(1, len(f)):
        if not sesiones[j]:
            ledger.append(row(j, 0., 0., False, 'sin_sesion'))
            continue
        pnl = p['direccion']*n*p['usd_por_contrato']*(f[j]-prev)
        prev = f[j]; margin += pnl; pnl_total += pnl
        call = 0.; close = j == c['dias']; reason = 'cierre_programado' if close else 'sesion'
        if not close and margin < maintenance:
            required = initial-margin
            if required > cash:
                close = True; reason = 'cierre_por_falta_de_caja'
            else:
                call = required; cash -= call; margin += call; calls += call
        if close:
            cash += margin-coste; margin = 0.
        expected = p['capital']+pnl_total-coste*(2 if close else 1)
        if not np.isclose(cash+margin, expected, atol=1e-7):
            raise AssertionError('No concilia el patrimonio')
        ledger.append(row(j, pnl, call, close, reason))
        if close:
            break
    return ledger


def csv_escribir(path, campos, filas):
    with open(path, 'w', newline='', encoding='utf-8') as out:
        w = csv.DictWriter(out, fieldnames=campos); w.writeheader(); w.writerows(filas)


def ejecutar(rutas=32):
    c = cargar(); dest = ROOT/'resultados'; dest.mkdir(exist_ok=True)
    fechas, sesiones = calendario(c)
    # Pequeña muestra de verificación, no Monte Carlo para inferencia.
    resumen = {}; rows = []
    for i, nombre in enumerate(('logistica', 'gbm', 'garch', 'binomial')):
        paths = generar(c, nombre, rutas, c['semilla']+i)
        invalid = ~np.isfinite(paths) | (paths <= 0)
        if invalid.any():
            raise ArithmeticError(f'{nombre}: {invalid.sum()} precios inválidos; revisar muestra sin filtrarla')
        resumen[nombre] = {'rutas_de_prueba': rutas, 'minimo': float(paths.min()), 'maximo': float(paths.max()),
                           'precios_invalidos': int(invalid.sum())}
        # Cotización proxy F=S, base cero, redondeada a la puja del contrato.
        futuros = np.round(paths[0]/c['operacion']['tick'])*c['operacion']['tick']
        ledger = cuenta(futuros, c)
        csv_escribir(dest/f'libro_ejemplo_{nombre}.csv', list(ledger[0]), ledger)
        csv_escribir(dest/f'trayectorias_prueba_{nombre}.csv', ['fecha']+[f'ruta_{j+1}' for j in range(rutas)],
                     ({'fecha': d.isoformat(), **{f'ruta_{j+1}': float(paths[j,k]) for j in range(rutas)}}
                      for k,d in enumerate(fechas)))
        for k,d in enumerate(fechas):
            q = np.quantile(paths[:,k], [.05,.5,.95])
            rows.append({'modelo': nombre, 'fecha': d.isoformat(), 'p05_muestra': q[0],
                         'mediana_muestra': q[1], 'p95_muestra': q[2]})
    csv_escribir(dest/'resumen_prueba.csv', list(rows[0]), rows)
    forecast = pronostico_garch(c)
    csv_escribir(dest/'pronostico_garch_rendimientos.csv', ['dia','media_log','varianza_innovacion','sigma'],
                 (dict(zip(['dia','media_log','varianza_innovacion','sigma'],r)) for r in forecast))
    (dest/'diagnostico_prueba.json').write_text(json.dumps(resumen, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,1,figsize=(10,11),sharex=True)
    for ax,nombre in zip(axes,('logistica','gbm','garch','binomial')):
        ar = [r for r in rows if r['modelo']==nombre]
        ax.fill_between(fechas, [r['p05_muestra'] for r in ar], [r['p95_muestra'] for r in ar],alpha=.2)
        ax.plot(fechas, [r['mediana_muestra'] for r in ar],label='Mediana de muestra de prueba')
        ax.axhline(c['s0'],color='gray',ls='--',lw=.8)
        ax.set_title(nombre.upper()); ax.set_ylabel('MXN/USD'); ax.grid(alpha=.2)
    fig.suptitle(f'Verificación técnica: {rutas} rutas por modelo\nNo usar para inferencia de riesgo')
    fig.autofmt_xdate();fig.tight_layout(rect=(0,0,1,.94));fig.savefig(dest/'trayectorias_prueba.png',dpi=160);plt.close(fig)
    print(json.dumps(resumen,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--rutas',type=int,default=32)
    ejecutar(parser.parse_args().rutas)

