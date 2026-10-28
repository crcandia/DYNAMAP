# Protocolo de selección fijado antes de consultar confirmation

Caso fijo: Chile, autorreporte exactamente3, propuesta37 Asamblea Constituyente. Ajuste principal101, sensibilidad201/301/401/501, sin reentrenar. Sectores definidos por distancia a37 con los cortes originales del piloto, aprendidos en discovery. Sector cercano≤primer corte, lejano≥segundo corte, intermedio entre ambos. Nunca redefinir sectores respecto de cada nueva propuesta.

Para cada persona, ordenar las90propuestas por distancia euclídea, rango1=mayor prioridad relativa inferida. Resumir con media del ranking, pesoigual porpersona.

Selección usando exclusivamente discovery, ajuste101:
1. Incluir siempre37.
2. Cercano: máximo2propuestas adicionales con rankingpromedio cercano≤30 y diferencia ranking_lejano−ranking_cercano≥10. Ordenar por diferencia descendente, luego rankcercanoascendente, luegoIDascendente.
3. Lejano: máximo2propuestas adicionales con rankinglejano≤30 y diferencia≤−10. Ordenar por diferencia ascendente, luego ranklejanoascendente, luegoIDascendente.
4. Compartida: hasta1propuesta restante con ambasmedias≤30 y diferenciaabsoluta≤5. Ordenar por máximo deambosranksascendente, luego diferenciaabsolutaascendente, luegoIDascendente. Si nohayejemplo, no sustituirregla.

Guardar selección conhash yhora antesde abrir outcomesconfirmation. Leerlos textos fuente de todas lasseleccionadas antesdeinterpretar. Nombres neutrales. Rankingsnoequivalen aapoyoabsoluto, rechazo oconsenso.

Figura única: A posiciones de participantesconfirmation conautorreport3 y propuestas seleccionadas, sectoresfijos101 incluidointermedio. B rankingpromedio de laspropuestas seleccionadas entrepersonasconfirmation de sectorescercano/lejano, etiquetadocomoinferido. C efecto observado cercano−lejano eneleccionesconfirmation, ajustado porefectosfijosdeoponente, método ybootstrap2000porparticipanteseed20261014delpiloto. Intervalos95percentilespuntuales, sinajustemultiplicidad. Mostrar todaslaspropuestas, inclusoinciertas/contradictorias. Soportesuficiente si≥30personas porextremo y≥10oponentesobservadosenambos,informarrecuentos yfracciónconsoportecomún sinocultarinsuficiencia.

Estabilidad: primero mismosIDs/sectores101 evaluadosenlascoordenadasdecadaseed, con perfiles90propuestas enambasfases. Después aplicarencadaseed suspropioscortesdiscovery originales yreportarmatrizcambios/Jaccard depertenencia. Noelegirpropuestas conconfirmation niestabilidadposterior. C noesexperimento confirmatorionuevo, losdatos fueronconsultadoshistóricamente. No benchmarks,peaks,dimensionalidad,clusteringnifragmentación. Noeditarrelease/Overleaf.
