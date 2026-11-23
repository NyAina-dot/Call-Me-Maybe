# Call-Me-Maybe

## Utilisation

Sans argument, le programme lit les fichiers dans `data/input/` et écrit
les résultats dans `data/output/` :

```bash
uv run python -m src
```

Les chemins peuvent être personnalisés avec `--functions_definition`,
`--input` et `--output` :

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Avec `make`, les arguments doivent être transmis via `ARGS` :

```bash
make run ARGS="--output data/path.json"
```

Pour modifier uniquement le fichier de sortie, cette forme est aussi
disponible :

```bash
make run OUTPUT=data/path.json
```