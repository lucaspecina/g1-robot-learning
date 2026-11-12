# g1-robot-learning

Notas y notebooks míos mientras aprendo robot learning con un Unitree G1 EDU
(humanoide, con manos Brainco). Lo que entendí, lo que medí y lo que no anduvo.

## Notebooks

En el orden en que se leen.

| Notebook | Qué hay |
|---|---|
| `01-lerobot-overview` | Qué es LeRobot y por qué existe, sus piezas, el ciclo de trabajo, cómo piden datos las distintas políticas |
| `02-raw-recordings` | Qué trae una grabación de teleoperación: estado, acción, cámara, duraciones |
| `03-convert-to-lerobot` | La conversión a LeRobot, cómo pide los datos ACT y qué se pierde al comprimir |
| `04-lerobot-dataset` | El formato por dentro, `LeRobotDataset`, ventanas de tiempo, aumentación, edición y la trampa de `episodes=` |
| `05-which-data-trained-act` | Cómo se reconstruyó con qué episodios se entrenó la primera ACT |
| `06-validate-dataset` | Validación del dataset: integridad, señales, lag, éxito, deriva, visual, split y fuga |
| `07-normalization-and-processors` | Por qué normalizar, los cuatro modos, los procesadores y un bug de LeRobot 0.4.1 |
| `08-act-data` a `12-act-inference` | ACT desarmado en cinco partes: los datos, el encoder del VAE, el decoder, el entrenamiento y el uso |
| `13-training` | El ciclo de `lerobot-train` armado a mano, la prueba de memorizar, costos y perillas |
| `14-offline-eval-act` | Evaluación offline de la primera ACT contra referencias tontas |

## Notas

- [`act-first-model.md`](notes/act-first-model.md): la primera ACT, con qué datos se entrenó y qué dio.
- [`act-split.json`](notes/act-split.json): qué episodios fueron a entrenamiento y cuáles a evaluación.
- [`act-memory-ideas.md`](notes/act-memory-ideas.md): seis formas de darle memoria a ACT.
- [`data-validation.md`](notes/data-validation.md): cómo validan los datos los laboratorios antes de entrenar.
- [`data-generation.md`](notes/data-generation.md): de dónde salen las demostraciones para manipulación.
- [`unitree-lerobot-patches.md`](notes/unitree-lerobot-patches.md): lo que hubo que cambiarle a `unitree_lerobot` para usar una sola cámara.
- [`glossary.md`](notes/glossary.md): glosario.

El código que usan los notebooks está en [`toolkit/`](toolkit/), con las
trampas de LeRobot 0.4.1 que fueron apareciendo.

## Qué sigue

1. Reentrenar ACT acá, con el dataset partido y normalizando sólo con los
   episodios de entrenamiento.
2. Experimentos de a una variable: aumentación, *temporal ensembling*,
   `kl_weight`.
3. Otros modelos con los mismos datos: Diffusion Policy, algún VLA chico.

## Entorno

Hace falta [uv](https://docs.astral.sh/uv/) y FFmpeg (en Ubuntu,
`sudo apt install ffmpeg`), que LeRobot usa para leer los videos.

```bash
bash setup.sh
source .venv/bin/activate
pytest
jupyter lab notebooks/
```

Los datos no están en el repo: las grabaciones crudas van en
`~/Desktop/episodios/` (o donde diga la variable `G1_RAW_RECORDINGS`), y los
datasets convertidos y los modelos en `data/`.

La simulación del G1 que hice antes está en
[g1-simulation](https://github.com/lucaspecina/g1-simulation).
