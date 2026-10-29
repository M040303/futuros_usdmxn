"""Pruebas de conciliacion significativas entre motor por lotes y libro diario."""
from copy import deepcopy
import numpy as np
from modelos import cargar,generar
from motor import procesar_lote,libro_ruta,simular

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
    a=simular(c,['gbm'],100,100,42)[0]
    b=simular(c,['gbm'],100,100,42)[0]
    assert a.equals(b)
    print('Aprobadas: 120 conciliaciones, cierre forzoso, precio plano, reproducibilidad.')
if __name__=='__main__': probar()
