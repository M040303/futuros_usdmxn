# Futuros USD/MXN — simulador Monte Carlo

Proyecto en Python con interfaz Streamlit, ejecución por consola y descarga de Excel. Implementa el escenario académico acordado para una posición en futuros de dólar (DA) de MexDer. **No envía órdenes ni utiliza cotizaciones de futuros en vivo.**

## Inicio rápido

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Alternativa por consola:

```bash
python ejecutar.py --rutas 100000 --lote 5000 --salida resultados/analisis.xlsx
python verificar.py
python probar_motor.py
```

La app inicia con 100,000 rutas por modelo. Para un millón: `python ejecutar.py --rutas 1000000 --lote 5000 --salida resultados/millon.xlsx`. El cálculo se divide en lotes. Las estadísticas por variable usan **todas** las rutas; la tabla grande de resultados individuales se sustituyó por un resumen y una ruta de ejemplo por modelo. La app conserva hasta 20,000 rutas por modelo para el histograma y para inspeccionar un caso. Requiere memoria suficiente para reunir temporalmente los resultados de un modelo.

Para publicar en Streamlit Community Cloud, subir el contenido de este directorio a un repositorio de GitHub y seleccionar `app.py` como archivo principal. Los datos incluidos son históricos y los parámetros corresponden al escenario de estudio; publicar la app no actualiza las tasas, precios ni contratos automáticamente.

## Escenario base

| Parámetro | Valor inicial |
|---|---:|
| Apertura | 13 de octubre de 2026 |
| Cierre | 12 de noviembre de 2026 |
| Capital | MXN 100,000 |
| Posición | 3 contratos largos de USD 10,000 cada uno |
| Spot inicial | 18.3765 MXN/USD |
| Margen inicial | MXN 15,000 por contrato |
| Umbral de mantenimiento | MXN 11,000 por contrato |
| Reposición | Hasta MXN 15,000 por contrato |
| Comisión supuesta | MXN 50 por contrato **por lado** |
| Deslizamiento supuesto | 0.0005 MXN/USD por lado |
| Días sin sesión | Sábados, domingos y 2 de noviembre de 2026 |

El monto de MXN 50 es **un supuesto de costo integral del escenario**, no una tarifa oficial actual de MexDer. Los MXN 15,000 y 11,000 son supuestos de margen del ejercicio, no garantías verificadas para todas las series o intermediarios. MexDer publica la [ficha del contrato DA](https://www.mexder.com.mx/wb3/wb/MEX/contratos_futuro) y sus [tarifas](https://www.mexder.com.mx/wb3/wb/MEX/tarifas_mexder); se deben cotejar antes de operar.

El valor simulado del futuro es `F_t = round(S_t/tick) × tick`. No incorpora base observada, diferencial de tasas, carry ni precio real de la serie. El contrato se cierra antes del vencimiento del escenario para evitar modelar entrega física.

## Tres generadores del precio

1. **Logística inversa sobre niveles:** `S_t = loc + scale × log(U_t/(1-U_t))`, con `U_t ~ Uniforme(0,1)` independiente por día. La ubicación y la escala provienen del ajuste histórico incluido. El nivel simulado puede saltar de un día a otro; esto eleva llamadas y cierres. No se recentra al spot actual. Si produce un precio no positivo, la corrida se detiene y lo reporta.
2. **Movimiento browniano geométrico:** `S_t = S_(t−1) × exp(media_log_diaria + sigma_diaria × Z_t)`, con `Z_t ~ N(0,1)`. La media y desviación se estiman sobre log rendimientos históricos; la volatilidad es constante.
3. **ARMA(2,2)–GARCH(1,1) con t estandarizada:** `r_t = ar1 r_(t−1) + ar2 r_(t−2) + ma1 ε_(t−1) + ma2 ε_(t−2) + ε_t`; `h_t = omega + alpha ε_(t−1)^2 + beta h_(t−1)`; `ε_t = sqrt(h_t) z_t` y `z_t` tiene t estandarizada a varianza 1. Después `S_t=S_(t−1)exp(r_t)`. Los coeficientes y el estado de arranque vienen del ajuste en R incluido en `data/`. Para usar datos nuevos hay que reestimar **coeficientes y estado juntos**.

La distribución de Fisher se excluyó por la decisión posterior del proyecto. La serie logística es un requerimiento académico, pero simular niveles independientes no conserva la dinámica cambiaria. El ajuste GARCH tiene persistencia casi unitaria y problemas diagnósticos documentados; sus cuantiles no deben presentarse como probabilidades calibradas del mundo real. Con innovaciones t exponenciadas, algunas esperanzas teóricas del precio pueden no existir. Por ello el tablero prioriza medianas y cuantiles; el ES presentado es solo de muestra.

## Contabilidad y métricas

Por cada sesión: `P&G_t = dirección × contratos × 10,000 × (F_t − F_última sesión)`. Se suma al margen. Si el margen cae debajo del mantenimiento, se transfiere caja externa hasta el objetivo. Una transferencia de margen **no es pérdida**. Si la caja no alcanza, la posición se cierra, se paga el costo de salida y se retira el saldo del margen (que puede ser negativo por una pérdida grande). No se agregan intereses, retiros mientras está abierta ni impuestos.

El resumen estadístico del Excel incluye mínimo, máximo, promedio muestral, desviación estándar muestral y mediana de las principales variables; los promedios del modelo GARCH no se interpretan como esperanzas teóricas.

El capital total inyectado realizado es `margen inicial + costo de entrada + llamadas pagadas`. El efectivo retirable final es caja más margen devuelto menos costo de salida. La ganancia neta es `efectivo retirable − capital inicial`. La llamada máxima, su día y spot se reportan para las llamadas **pagadas**; el Excel también muestra la llamada máxima hipotética hasta el horizonte.

Para estimar la suficiencia de capital se lleva una **segunda cuenta hipotética**, que sigue pagando todas las llamadas hasta el cierre previsto sin forzar la salida. `Capital requerido = margen inicial + costo de entrada + suma de llamadas hipotéticas`. La fracción de rutas donde este importe no excede el capital es la probabilidad empírica de suficiencia bajo cada modelo. Esta cuenta no representa P&G realizado.

`VaR_α` es el cuantil α de la pérdida neta `−ganancia neta`. `ES_α` es el promedio **muestral** de pérdidas por encima del VaR. También se reporta VaR del P&G bruto al cierre previsto. Los cuantiles, extremos y probabilidades dependen de los modelos, la calibración, la semilla y el tamaño de muestra. Cambiar solo los parámetros operativos preserva las mismas rutas si se mantienen modelo, horizonte, semilla, cantidad de rutas **y tamaño de lote**.

El modo porcentual permite explorar la estrategia del profesor: margen inicial como fracción del nocional (5 % inicialmente), mantenimiento como 11/15 del inicial y objetivo igual al inicial. El número de contratos se elige explícitamente: la app no equipara automáticamente todo el capital a margen porque esa elección eleva el riesgo de llamada y de pérdidas superiores al capital inicial.

## Aportaciones por fecha

La pestaña **Aportaciones por fecha** muestra, para cada modelo, mediana y P95
de las llamadas acumuladas, P95 de la llamada diaria y reserva disponible tras
pagar el margen inicial y el costo de entrada. Se calculan cuantiles exactos
sobre todas las rutas, usando un archivo temporal por modelo para el detalle
diario. El Excel de la app incluye la hoja `Aportaciones_fecha`.

Las curvas siguen la cuenta hipotética sin cierre forzoso. Excluyen margen
inicial, liquidación final y costo de salida; al cierre programado no hay
reposición, igual que en el motor original. Los percentiles son por fecha,
no una trayectoria ni una banda de confianza simultánea. No deben sumarse
los P95 diarios. El P95 acumulado final dimensiona la reserva de llamadas de
todo el periodo; el faltante descuenta la reserva que ya está disponible.
Una aportación transfiere caja al margen y no es una pérdida adicional.

Pruebas: `python probar_aportaciones.py` concilia con libros individuales,
verifica días sin sesión, posiciones largas/cortas, márgenes fijo/porcentual,
colas sin censura por cierre, más de 20,000 rutas y formatos Excel.

## Archivos

- `app.py`: controles y visualizaciones Streamlit.
- `aportaciones.py`: cuantiles por fecha y gráficas de llamadas de margen.
- `ejecutar.py`: ejecución por consola y generación de Excel.
- `modelos.py`: generadores de spot y verificaciones originales de R.
- `motor.py`: cuentas vectorizadas, suficiencia, VaR y flujo por trayectoria.
- `excel.py`: libro con resumen, estadísticas por variable, diccionario, parámetros, trayectorias y flujo diario.
- `configuracion.json`: parámetros editables de escenario y modelos.
- `data/`: serie histórica y estado de ARMA–GARCH en R.
- `verificar.py`, `probar_motor.py`: reproducción histórica y conciliación de cuentas.
- `resultados/ejemplo_100000.xlsx`: corrida ilustrativa de 100,000 rutas por modelo con semilla fija.

## Pendientes para una versión de uso real

Obtener cotizaciones y profundidad del futuro DA específico; verificar garantías y costos del intermediario; validar calendario de sesiones; reestimar y someter los modelos a pruebas fuera de muestra y backtesting de VaR; añadir choques de estrés y costos de financiamiento. El prototipo calcula riesgo condicional a supuestos y **no** valida que el precio futuro o la liquidez simulados sean correctos.

