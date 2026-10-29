"""Verificación por reproducción de resultados guardados en R y casos contables."""
import csv
import json
import numpy as np
from scipy.stats import logistic, norm, t
from modelos import ROOT, cargar, generar, pronostico_garch, cuenta, calendario


def verificar():
    c = cargar(); p = c['garch']['coeficientes']; checks = {}
    with open(ROOT/'data/estado_historico_R.csv') as f:
        records = list(csv.DictReader(f))
    get = lambda k: np.array([float(x[k]) for x in records])
    r=get('rendimiento'); e=get('residuo_R'); h=get('varianza_R'); m=get('media_ajustada_R')
    mr=p['ar1']*r[1:-1]+p['ar2']*r[:-2]+p['ma1']*e[1:-1]+p['ma2']*e[:-2]
    hr=p['omega']+p['alpha1']*e[1:-1]**2+p['beta1']*h[1:-1]
    checks['error_max_media_historica_R'] = float(np.max(abs(mr-m[2:])))
    checks['error_max_varianza_historica_R'] = float(np.max(abs(hr-h[2:])))
    assert checks['error_max_media_historica_R'] < 1e-12
    assert checks['error_max_varianza_historica_R'] < 1e-12
    # Se excluyen los primeros dos estados de inicialización del filtro de R.
    row=pronostico_garch(c)[0]
    checks['pronostico_1_dia_media_log'] = row[1]
    checks['pronostico_1_dia_varianza_innovacion'] = row[2]
    checks['pronostico_1_dia_sigma'] = row[3]
    checks['ejecucion_independiente_ugarchforecast_R'] = 'Pendiente: R/rugarch no disponible en este entorno'
    # Fórmula de inversa contrastada contra implementación de referencia SciPy.
    u=np.array([.001,.025,.5,.975,.999]); lp=c['logistica']
    inv=lp['location']+lp['scale']*np.log(u/(1-u))
    np.testing.assert_allclose(inv,logistic.ppf(u,loc=lp['location'],scale=lp['scale']),atol=1e-12)
    # La escala t usada tiene varianza unitaria exacta, no la varianza nu/(nu-2).
    nu=p['shape']; assert np.isclose(t.var(nu)*(nu-2)/nu,1.)
    for name in ['logistica','gbm','garch']:
        a=generar(c,name,32);b=generar(c,name,32)
        np.testing.assert_array_equal(a,b)
        assert a.shape==(32,31) and np.all(a[:,0]==c['s0']) and np.all(np.isfinite(a)) and np.all(a>0)
    # Confirmar corrección de deriva: la mediana GBM de un día usa media_log.
    assert np.isclose(c['gbm']['mu_diaria']-c['gbm']['sigma_diaria']**2/2,c['gbm']['media_log_diaria'])
    # Casos manuales que detectan doble conteo de llamadas, costos y pérdidas.
    C=dict(c);C['operacion']=dict(c['operacion']);C['operacion'].update(contratos_base=1,capital=100000.)
    f=np.full(31,c['s0']);dates,ses=calendario(C)
    constant=cuenta(f,C); assert np.isclose(constant[-1]['caja'],99890.)
    first=int(np.where(ses & (np.arange(31)>0))[0][0])
    f[first:]-=.41;book=cuenta(f,C)
    assert np.isclose(sum(x['llamada'] for x in book),4100.)
    assert np.isclose(book[-1]['caja'],95790.)
    # Recuperación posterior: llamada no se convierte en pérdida adicional.
    f[-1]=c['s0'];book=cuenta(f,C);assert np.isclose(book[-1]['caja'],99890.)
    C['operacion']['capital']=17000.;f[-1]=c['s0']-.41
    book=cuenta(f,C);assert book[-1]['motivo']=='cierre_por_falta_de_caja'
    assert np.isclose(book[-1]['caja'],12790.)
    # Movimiento en día sin sesión se reconoce al siguiente cierre de sesión.
    C['operacion']['capital']=100000.;f=np.full(31,c['s0']);
    idx=next(i for i in range(1,29) if not ses[i] and ses[i+1])
    f[idx:]+=.1;book=cuenta(f,C)
    assert book[idx]['pnl']==0 and np.isclose(book[idx+1]['pnl'],1000.)
    checks['casos_contables']=['precio constante','caída y llamada','recuperación tras llamada','cierre por falta de caja','acumulación entre sesiones']
    checks['estado']='Verificaciones técnicas aprobadas; no es validación predictiva'
    checks['AR_modulos_raices']=abs(np.roots([-p['ar2'],-p['ar1'],1])).tolist()
    checks['MA_modulos_raices']=abs(np.roots([p['ma2'],p['ma1'],1])).tolist()
    (ROOT/'resultados/verificacion.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False))
    # Si se entrega la salida de referencia de R, compara realmente el pronóstico.
    ref=ROOT/'resultados/pronostico_referencia_R.csv'
    if ref.exists():
        with open(ref) as f: v=list(csv.DictReader(f))
        np.testing.assert_allclose(float(v[0]['media_log']),row[1],atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose(float(v[0]['sigma']),row[3],atol=1e-10,rtol=1e-8)
        checks['ejecucion_independiente_ugarchforecast_R']='Comparación aprobada con CSV suministrado'
        (ROOT/'resultados/verificacion.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False))
    print(json.dumps(checks,indent=2,ensure_ascii=False))


if __name__=='__main__': verificar()
