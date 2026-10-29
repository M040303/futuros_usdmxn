# Viabilidad, riesgos y plan de validación

## Qué contesta el simulador

Con las hipótesis de `configuracion.json`, estima el P&G, los depósitos por llamadas, la llamada máxima, el efectivo retirable, el capital requerido, la probabilidad de insuficiencia de caja y el VaR/ES muestrales. Compara tres mecanismos distintos de generar USD/MXN. Las probabilidades son condicionales a cada modelo; no equivalen a frecuencias futuras verificadas.

## Comparación de viabilidad

| Modelo | Qué reproduce | Riesgo de subestimación | Riesgo de sobreestimación | Estado |
|---|---|---|---|---|
| Logística inversa de niveles | Distribución histórica marginal de precios | No captura tendencia, cambios de régimen ni dependencia temporal | Al volver a sortear el nivel cada día, crea saltos y llamadas excesivas | Requerimiento académico; usar como escenario de estrés, no como pronóstico central |
| MBG de log rendimientos | Precio positivo y movimientos acumulativos | Volatilidad constante omite agrupamiento de crisis y colas pesadas | Puede sobreestimar riesgo en periodos muy tranquilos si la muestra calibra volatilidad elevada | Línea base interpretable; requiere backtesting |
| ARMA–GARCH–t | Dependencia lineal, volatilidad variable y colas pesadas de rendimientos | Mal ajuste de estado inicial o parámetros puede ocultar cambios de régimen | Persistencia cercana a 1 y cola t pueden producir escenarios extremos irreales | Exploratorio; requiere diagnóstico y prueba fuera de muestra |

La logística se ajustó a **niveles** porque así quedó acordado para el ejercicio, aunque la independencia diaria no sea una hipótesis adecuada para un tipo de cambio. El GARCH incluido se reprodujo numéricamente desde el estado exportado por R; esto verifica implementación, no capacidad predictiva. Su ajuste reportó `condH=NaN`, persistencia aproximadamente 0.999 y una parte MA no invertible. No se reporta promedio teórico del precio GARCH exponenciado: una t estandarizada tiene colas polinomiales y su exponencial carece de momento positivo finito.

## Lectura de la corrida ilustrativa

Con 10,000 rutas por modelo, MXN 100,000 y tres contratos largos, semilla `20261026`, se obtiene aproximadamente:

| Modelo | Capital suficiente | Cierre forzoso | Mediana ganancia neta | VaR 95 % neto |
|---|---:|---:|---:|---:|
| Logística | 67.43 % | 32.57 % | −MXN 33,168 | MXN 71,946 |
| MBG | 100.00 % | 0.00 % | −MXN 675 | MXN 23,011 |
| ARMA–GARCH–t | 99.14 % | 0.86 % | MXN 1,382 | MXN 27,543 |

Estas cifras son resultados del prototipo y cambian con capital, contratos, ventana, parámetros, semilla y número de rutas. La distancia entre modelos es una señal de **riesgo de modelo**. El 100 % de suficiencia en 10,000 rutas MBG indica cero fallas observadas en esa muestra, no probabilidad real de 100 %.

## Estrategia del margen de 5 %

Una garantía de 5 % permite abrir una exposición nominal mayor que el capital. Por ejemplo, con MXN 100,000 y spot 18.3765, cada contrato de USD 10,000 representa unos MXN 183,765 y requeriría MXN 9,188.25 de margen bajo el supuesto de 5 %. Diez contratos consumirían MXN 91,882.50 de margen más MXN 550 de costo de entrada, dejando MXN 7,567.50 de caja. Un movimiento adverso pequeño podría activar una llamada superior a esa caja. La estrategia puede mejorar el retorno porcentual cuando el mercado favorece la posición, pero también aumenta la probabilidad de cierre forzoso y pérdidas. No se debe definir el número de contratos solo por el máximo que cabe en el margen inicial.

Para elegir `n`, correr una cuadrícula de contratos con **las mismas** rutas de precio y elegir el mayor `n` que cumpla criterios establecidos antes del análisis, por ejemplo probabilidad de suficiencia ≥ 99 % y VaR 99 % dentro de la pérdida tolerable. Esos límites son decisiones del usuario; el prototipo no presupone un umbral apropiado. La comparación de garantías fijas y 5 % puede hacerse con los controles de Streamlit.

## Cómo validar antes de confiar

1. **Datos:** comprobar origen, frecuencia, periodos faltantes, fines de semana, cambios de metodología, precios no positivos y tratamiento de FIX contra spot negociable. Validar que la serie de calibración termine antes del inicio de la proyección.
2. **Ajuste:** revisar residuos, autocorrelación, cobertura de colas, sensibilidad a ventana, parámetros y estabilidad. En logística, hacer pruebas de ajuste y dependencia; no basta con un ajuste visual o un parámetro de bondad no informado.
3. **Prueba fuera de muestra:** usar ventanas móviles históricas. En cada fecha, calibrar solo con pasado, pronosticar el horizonte, guardar cuantiles y confrontar con el precio y llamadas que se observaron después. Evitar usar un dato futuro para ajustar su propia predicción.
4. **Backtesting de VaR:** contar excepciones de VaR 95/99 y revisar si ocurren agrupadas. Comparar cobertura observada con nominal, junto con intervalos de confianza binomiales; aplicar pruebas de cobertura incondicional y de independencia.
5. **Liquidez y estrés:** contrastar el requerimiento de capital previsto con la caja realmente disponible, variaciones bruscas del dólar, aumentos de garantía, deslizamiento y cierre a precios peores. Revisar la supervivencia hasta el vencimiento bajo escenarios extremos.
6. **Mercado real:** sustituir `F=S` por cotizaciones del contrato y su calendario, verificar serie exacta, margen, cargos del intermediario y reglas de liquidación. La [ficha DA de MexDer](https://www.mexder.com.mx/wb3/wb/MEX/contratos_futuro) confirma USD 10,000 y tick 0.0001; las [tarifas publicadas](https://www.mexder.com.mx/wb3/wb/MEX/tarifas_mexder) difieren del supuesto de MXN 50 usado en este ejercicio.

Solo después de estas verificaciones se podrían interpretar probabilidades como estimaciones aplicables a una decisión financiera. El Excel de ejemplo sirve para revisar la mecánica, no para ejecutar una operación.
