# Entrega final — Fase 3: traslado a NIH y auditoría

BRAX no quedó disponible, así que el traslado se hace a **NIH ChestX-ray14**
\(Wang et al., 2017\): radiografías de adultos del NIH Clinical Center. El
protocolo que se había fijado para BRAX se conserva tal cual: población adulta,
vistas PA/AP, etiqueta binaria inequívoca, partición por paciente
`adapt_train/adapt_val/external_test` y `external_test` aislado de toda decisión.

## Ejecución

Desde la raíz del repositorio, con Kermany descargado (`make download-kermany`)
y la Fase 2 entrenada (`make phase2-train`, que deja los checkpoints):

```bash
make phase3-data         # metadatos NIH (SHA-256), partición y ~5.300 imágenes (≈2,5 GB)
make phase3-run          # evaluación directa, traslado, control ImageNet, curva y RF
make phase3-audit        # modelo final elegido en validación; Grad-CAM, pruebas, fallas
make phase3-report       # tablas LaTeX y figuras del informe
make phase3-replication  # brecha frente a Rahman et al.: umbral y partición por imagen
```

Todas las etapas son reanudables: una corrida con `metrics.json` no se repite.

### Ejecución desatendida

```bash
make resume   # corre todo (`make final`) en segundo plano, bajo supervisión
make status   # etapas terminadas, últimas líneas del log y uso de GPU
```

`run_final.sh` ejecuta las etapas en orden y deja una marca por etapa en
`logs/stages/`. `supervise_final.sh` lo reinicia si el proceso muere o si pasan
30 minutos sin que se escriba nada en `logs/final.log` ni en los resultados.
Tras tres reinicios seguidos sin que termine ninguna etapa se detiene y deja
`logs/ALERTA.txt`, porque un error que se repite no se arregla reintentando. Si
el equipo se apaga, basta con volver a ejecutar `make resume`. Para repetir una
etapa ya marcada, borra su archivo en `logs/stages/`.

## Datos

`prepare_nih.py` decide todo desde los metadatos, antes de descargar imágenes:

- Adultos (≥ 18 años), vistas PA/AP.
- Positivo: `Pneumonia` en `Finding Labels` (con o sin co-hallazgos). Negativo:
  `No Finding` en un paciente que nunca tuvo neumonía en ningún estudio.
- Diseño caso-control: las 1.319 imágenes con neumonía de 938 pacientes y 3
  controles por caso, uno por paciente (prevalencia 0,25 en cada split).
- Los pacientes caso se reparten 70/15/15 con semilla 42; los 110 con caja de
  neumonía del radiólogo (`BBox_List_2017.csv`) se reservan en `external_test`
  para auditar Grad-CAM y nunca entran al ajuste.

Las etiquetas vienen de NIH Box; las imágenes, del espejo
`alkzar90/NIH-Chest-X-ray-dataset` en Hugging Face, porque los enlaces directos
de Box ya no responden. `fetch_nih_images.py` lee cada ZIP remoto por rangos HTTP
y extrae solo las imágenes de la partición, verificando el CRC32 de cada una.

## Experimentos

| Etapa | Qué mide |
|---|---|
| `evaluate_zero_shot.py` | Modelos Kermany sobre NIH sin reentrenar, con umbral Kermany y con umbral reelegido en `adapt_val` |
| `adapt_nih.py` | Traslado desde Kermany: extractor congelado vs. fine-tuning parcial, 3 semillas |
| `adapt_nih.py --source imagenet` | Control: el mismo ajuste sin pasar por Kermany |
| `adapt_nih.py --train-fraction` | Curva de datos escasos: 10, 25 y 50 % de los pacientes de `adapt_train` |
| `baseline_random_forest.py` | Línea base de Fase 1 en la misma partición NIH |
| `compare_paired.py` | Bootstrap pareado por paciente de cada diferencia |
| `run_audit.py` | Elige el modelo final por AUC medio en `adapt_val` y lo audita |
| `audit_gradcam.py` | Grad-CAM contra las cajas del radiólogo y masa en el marco periférico |
| `audit_tests.py` | Pruebas de respuesta conocida, entradas fuera de dominio y test pediátrico de Kermany |
| `audit_failures.py` | Subgrupos, perfil de errores, calibración (temperatura y Platt) y valor predictivo |
| `replication_gap.py` | Por qué la especificidad difiere de la de Rahman et al. |

## Cambios respecto de la Fase 2

- Las seis corridas Kermany se reejecutaron y ahora registran `val_loss`.
- Las BatchNorm de las capas congeladas quedan en modo evaluación: antes el
  "extractor congelado" seguía ajustando sus estadísticas a Kermany.
- La latencia se mide sincronizando el dispositivo, con mediana e IQR de 100
  repeticiones.
- Los NORMAL de Kermany se agrupan por prefijo `IM-XXXX`: nombres como
  `IM-0501-0001-0001` e `IM-0501-0001-0002` corresponden al mismo niño y antes
  podían caer en splits distintos.
- `write_json` escribe `null` en lugar de `NaN`, que no es JSON válido.

## Entorno de ejecución

Las corridas de esta entrega se hicieron en WSL2 con una NVIDIA GTX 1650 (4 GB).
Su driver solo admite CUDA 11.x, por lo que se usó `torch==2.7.1+cu118`
(ver `requirements.txt`).
