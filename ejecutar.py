"""CLI: python ejecutar.py --rutas 100000 --salida resultados/analisis.xlsx"""
import argparse
from pathlib import Path
from modelos import cargar
from motor import simular,MODELOS
from excel import crear_excel

def main():
    a=argparse.ArgumentParser(description='Monte Carlo de futuros USD/MXN')
    a.add_argument('--rutas',type=int,default=100000);a.add_argument('--lote',type=int,default=5000)
    a.add_argument('--modelos',nargs='+',choices=MODELOS,default=list(MODELOS))
    a.add_argument('--contratos',type=int);a.add_argument('--capital',type=float)
    a.add_argument('--modo',choices=['fijo','porcentaje'],default='fijo')
    a.add_argument('--semilla',type=int);a.add_argument('--salida',default='resultados/analisis_montecarlo.xlsx')
    args=a.parse_args(); c=cargar()
    if args.capital is not None: c['operacion']['capital']=args.capital
    if args.contratos is not None: c['operacion']['contratos_base']=args.contratos
    def progress(model,done,total): print(f'\r{model}: {done:,}/{total:,}',end='',flush=True)
    comp,detalle,ejemplos,bandas,estadisticas,cierres=simular(c,args.modelos,args.rutas,args.lote,args.semilla,args.modo,progreso=progress)
    print();dest=Path(args.salida);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(crear_excel(c,comp,detalle,ejemplos,bandas,estadisticas,cierres,modo=args.modo))
    columnas=['modelo','simulaciones','prob_capital_suficiente','prob_cierre_forzoso',
              'mediana_neta','p95_capital_requerido','VaR95_neto','VaR99_neto']
    print(comp[columnas].to_string(index=False));print(f'Excel: {dest.resolve()}')
if __name__=='__main__': main()
