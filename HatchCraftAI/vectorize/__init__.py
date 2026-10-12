"""Pipeline de vectorización de precisión (S1).

Método `method="precision"` de ImageToPatConverter, inspirado en
Vector Magic / Potrace (ver plan_vectorizacion_precision.md).

S1 incluye: F1 preproceso (preprocess), F2 clasificador de trazo (classify),
F3a centro de trazo sub-píxel (centerline) y orquestación (pipeline).
"""
