# 2. Entorno de notebooks sin Jupyter Lab completo (límite de rutas largas de Windows)

**Fecha:** 2026-07-26
**Estado:** Aceptada

## Contexto

`pip install jupyter` (metapaquete completo, incluye Jupyter Lab y
extensiones de widgets) falló al crear el `venv` del proyecto:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file
or directory: '...venv\share\jupyter\labextensions\@jupyter-widgets\
jupyterlab-manager\static\vendors-node_modules_...c79fc2bcc3f69676beda.js.map'
HINT: This error might have occurred since this system does not have
Windows Long Path support enabled.
```

La causa es la combinación de (a) el límite de 260 caracteres de ruta de
Windows por defecto y (b) la ruta ya profunda del proyecto dentro de
OneDrive (`...\OneDrive\Cursos\MIU data science\liver-disease-fairness\venv\...`),
que deja muy poco margen para los nombres de archivo largos que trae
Jupyter Lab. Habilitar soporte de rutas largas en Windows requiere permisos
de administrador (cambio de registro), fuera del alcance de lo que se quiere
resolver solo para poder trabajar con notebooks.

El proyecto se trabaja desde VS Code, cuya extensión de Jupyter solo
necesita un kernel registrado en el entorno (`ipykernel`) para abrir,
editar y ejecutar celdas de un `.ipynb` — no necesita el servidor web de
Jupyter Lab/Notebook.

## Decisión

Instalar `ipykernel` + `nbconvert` en vez del metapaquete `jupyter`.
`nbconvert --to notebook --execute` se usa como equivalente de
"restart & run all" para verificar cada notebook antes de cada checkpoint.

## Consecuencias

- Los notebooks se pueden crear, editar y ejecutar con normalidad desde
  VS Code, y se pueden verificar de punta a punta desde la terminal.
- **Limitación aceptada:** no se puede levantar un Jupyter Lab/Notebook
  standalone en el navegador (`jupyter lab` / `jupyter notebook`) con este
  entorno. Si en el futuro hace falta, la vía es habilitar "Windows Long
  Path support" (registro, requiere administrador) y reinstalar `jupyter`
  completo — este ADR quedaría superado por uno nuevo si eso ocurre.
