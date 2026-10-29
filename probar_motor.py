"""Pruebas de conciliacion significativas entre motor por lotes y libro diario."""
from copy import deepcopy
import numpy as np
from modelos import cargar,generar
from motor import procesar_lote,libro_ruta,simular
from excel import DEFINICIONES

def probar():
    c=cargar()
    for modo in ('fijo','porcentaje'):
        for nombre in ('logistica','gbm','garch'):
            paths=generar(c,nombre,20,9831)
            d=procesar_lote(paths,c,3,modo)
            for i in range(20):
                b=libro_ruta(paths[i],c,3,modo)
                assert np.isclose(b.iloc[-1].patrimonio,d.iloc[i].efectivo_retirar)
                assert np.isclose(b.llamada.sum(),d.iloc[i].depositos_llamadas)
                assert np.isclose(b.pnl_dia.sum(),d.iloc[i].pnl_bruto)
                assert int(d.iloc[i].dia_cierre)==len(b)-1
    c2=deepcopy(c);c2['operacion']['capital']=47000.
    path=np.full((1,c2['dias']+1),c2['s0']);path[:,1:]-=0.5
    x=procesar_lote(path,c2,3,'fijo').iloc[0]
    assert x.cierre_forzoso and not x.capital_suficiente_horizonte
    assert x.capital_requerido_horizonte>47000
    c3=deepcopy(c); c3['operacion']['capital']=100000.
    x=procesar_lote(np.full((1,c3['dias']+1),c3['s0']),c3,3,'fijo').iloc[0]
    assert np.isclose(x.ganancia_neta,-330.) and x.numero_llamadas==0
    assert len(simular(c,['gbm'],100,100,42))==4
    a=simular(c,['gbm'],100,100,42,con_estadisticas=True)
    b=simular(c,['gbm'],100,100,42,con_estadisticas=True)
    assert a[0].equals(b[0])
    detalle=a[1]; estadisticas=a[4]; cierres=a[5]
    assert set(detalle.columns)==set(DEFINICIONES)
    assert estadisticas.observaciones.max()==100
    x=estadisticas[estadisticas.variable=='ganancia_neta'].iloc[0]
    assert np.isclose(x.media,detalle.ganancia_neta.mean())
    assert np.isclose(x.varianza,detalle.ganancia_neta.var(ddof=1))
    assert cierres.rutas.sum()==100
    print('Aprobadas: 120 conciliaciones, cierre forzoso, precio plano, reproducibilidad y estadísticas completas.')
if __name__=='__main__': probar()
