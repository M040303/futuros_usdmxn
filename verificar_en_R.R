# Ejecutar desde la carpeta futuros_usdmxn si se desea contrastar el pronóstico
# con una ejecución independiente de ugarchforecast. No vuelve a estimar.
library(rugarch)
entorno <- new.env()
load("data/ajuste_original.RData", envir = entorno)
fc <- ugarchforecast(entorno$fit, n.ahead = 30)
out <- data.frame(dia = 1:30, media_log = as.numeric(fitted(fc)),
                  sigma = as.numeric(sigma(fc)))
write.csv(out, "resultados/pronostico_referencia_R.csv", row.names = FALSE)
# Después ejecutar python verificar.py para contrastar el primer día.
