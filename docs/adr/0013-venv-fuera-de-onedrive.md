# 13. El `venv/` se crea siempre fuera de OneDrive, en `%LOCALAPPDATA%\venvs\<repo>`

**Fecha:** 2026-08-16
**Estado:** Aceptada

## Contexto

El proyecto se desarrolla desde dos máquinas que comparten la misma carpeta
de OneDrive. `.gitignore` impide que *git* rastree `venv/`, pero eso no
impide que **OneDrive sincronice la carpeta igual** — son dos mecanismos
independientes. Hasta ahora esto se venía tratando como una serie de síntomas
puntuales a resolver con "borrar y recrear" (documentado en `AGENTS.md`
2026-08-15): el `venv/` llegaba con el `pyvenv.cfg` apuntando al intérprete
de la otra máquina, o se borraba en ambas máquinas a la vez al recrearlo.

El 2026-08-16 apareció un tercer síntoma, más sutil que los dos anteriores:
tras instalar `streamlit` en una máquina (lo que arrastró `pyarrow` como
dependencia transitiva de `pandas`), esta máquina recibió
`venv/Lib/site-packages/pyarrow/` con la estructura de subcarpetas completa
pero **cero archivos** — OneDrive sincronizó los directorios antes de bajar
su contenido. Python no lo trata como error (un directorio sin
`__init__.py` es un *namespace package* válido), así que `import pyarrow`
no falla; el fallo aparece un paso después, en
`pandas.compat.pyarrow`, con `AttributeError: module 'pyarrow' has no
attribute '__version__'` — un mensaje que no sugiere en absoluto "paquete
vacío por sincronización incompleta". Los 33 tests del proyecto no llegaban
ni a recolectarse.

Los tres síntomas comparten una única causa: un directorio mutable,
compartido, y escrito por dos procesos de instalación distintos en dos
máquinas distintas, dentro de una carpeta que un tercer proceso (OneDrive)
sincroniza sin coordinación con `pip`. Seguir tratando cada síntoma nuevo
como un caso aislado garantiza que aparezca un cuarto.

## Decisión

El `venv/` **nunca** vive dentro de la carpeta de OneDrive. Se crea en
`%LOCALAPPDATA%\venvs\<nombre-del-repo>` — fuera del árbol sincronizado,
propio de cada máquina:

```powershell
py -3.13 -m venv "$env:LOCALAPPDATA\venvs\liver-disease-fairness"
& "$env:LOCALAPPDATA\venvs\liver-disease-fairness\Scripts\python.exe" -m pip install -r requirements.txt
```

`%LOCALAPPDATA%` se usa como variable de entorno, nunca expandido a una ruta
literal en ningún archivo versionado — así `AGENTS.md`/`README.md` describen
el mismo comando para cualquier usuario y máquina, sin introducir una ruta
absoluta fija que `test_sin_rutas_absolutas` (o una revisión manual) tendría
que señalar.

Como consecuencia directa, `requirements.txt` se limpia de `streamlit`,
`pyarrow` y las 20 dependencias transitivas que arrastraron (añadidas por
error antes de que la Fase P las necesitara): mantenerlas fijadas solo
garantizaba repetir el síntoma 3 en cada reinstalación futura. Se
reincorporarán cuando la Fase P (tablero) las necesite de verdad.

## Consecuencias

- Cada máquina tiene su propio `venv/`, aislado: instalar o romper el
  entorno en una máquina no afecta a la otra — elimina de raíz los síntomas
  1 y 2 (intérprete ajeno, borrado compartido), no solo el 3 (paquete
  vacío).
- La ruta del entorno deja de ser la misma en ambas máquinas (antes era
  `./venv` relativo al repo, ahora depende de `%LOCALAPPDATA%` por usuario).
  Aceptable porque el `venv/` nunca se versiona ni se referencia por ruta
  absoluta en ningún documento — solo por la variable de entorno.
- `git status`/`git clean` dejan de ver `venv/` como parte del árbol de
  trabajo del repo en ningún sentido (antes tampoco lo rastreaba git, pero
  vivía físicamente dentro de la carpeta sincronizada; ahora ni eso).
- `streamlit`/`pyarrow` quedan fuera de `requirements.txt` hasta la Fase P;
  cualquier sesión que los necesite antes de esa fase debe instalarlos
  manualmente y no fijarlos en el archivo compartido.
