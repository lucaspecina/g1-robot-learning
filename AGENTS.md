# Reglas de trabajo

## Fuentes antes de diseñar

Antes de diseñar o cambiar algo:

1. Consultar primero la documentación oficial de lo que se usa (LeRobot,
   Unitree, el paper y el repositorio oficial de cada modelo).
2. Revisar después implementaciones y prácticas de comunidades técnicas
   confiables, priorizando repositorios oficiales, papers y laboratorios con
   resultados reproducibles.
3. Repetir esa consulta cada vez que aparezca una duda, una anomalía o una
   decisión nueva. Una investigación anterior no reemplaza verificar la
   recomendación específica del problema actual.
4. Registrar qué parte coincide con el flujo oficial y qué parte es una
   adaptación propia. Nunca describir algo como "oficial" si se modificaron
   versiones, configuración, entradas o datos.
5. Si nos apartamos de la práctica recomendada, hacerlo solamente por una
   necesidad medida y dejar documentados la evidencia, el costo y el camino
   para revertirlo.

## Disciplina experimental

- Cambiar una sola variable por experimento.
- Declarar antes la métrica que debería mejorar y el criterio de aprobación.
- Repetir las mediciones; una sola ejecución no prueba una tendencia.
- Complementar las métricas con inspección visual. Un resultado numérico no
  valida por sí solo que el movimiento tenga el sentido esperado.
- Si un cambio no mejora la métrica declarada, revertirlo.
- Mantener una referencia sin modificaciones y comparar contra ella.

## Convenciones

- Identificadores, módulos y funciones: en inglés.
- Carpetas y nombres de archivo: en inglés, en minúsculas y separados por
  guiones (`notes/data-validation.md`). Los módulos de Python usan guion bajo
  porque Python no puede importar nombres con guiones.
- Comentarios, docstrings, notas y mensajes para el usuario: en español.
- Los comentarios explican por qué existe una decisión, no repiten qué hace
  el código.
- Explicar robótica en lenguaje humano; definir cualquier término técnico la
  primera vez que se use.
- Los notebooks se versionan sin salidas (el filtro `nbstripout`, que instala
  `setup.sh`), porque las salidas incluyen imágenes del lugar real.

## Qué NO va en este repositorio

**Este repositorio es público.** Todo lo que se commitea queda accesible, y el
historial no se borra sacando el texto de un archivo.

Nunca escribir en archivos versionados:

- Nombres de la empresa, del grupo, de clientes o de socios comerciales.
- Presupuestos, cotizaciones, precios negociados, condiciones de compra o
  nombres de proveedores y sus contactos.
- Ubicaciones o instalaciones reales de despliegue.
- Credenciales, direcciones IP internas o nombres de máquinas de la
  organización.

Todo eso va a **`private/`**, que está en `.gitignore`.

En los documentos versionados se habla en general: "el equipo socio", "el sitio
real", "una planta industrial". El contenido técnico —arquitectura, mediciones,
decisiones de ingeniería, precios de lista públicos— sí puede ir versionado.

Ante la duda, va a `private/`.
