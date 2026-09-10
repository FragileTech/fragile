# Landing page assets

The home page uses the shared FragileTech logo and favicon from `docs/`.
Arcade and Control previews are copied from `docs/_static/arcade_lab/sonic-fog.png`
and `docs/_static/control_lab/tutorials/racing-overview.png`; CSS crops the source
screenshots around the game and racing circuit.

To refresh Optimization and LLM screenshots, build those labs, start
`python3 fractal-gas-web/tools/serve-control.py --port 8099`, and run
`npm --prefix fractal-gas-web run capture:landing`.
The capture uses the actual Optimization engine and LLM Analysis interface.
LLM requests are intercepted by the deterministic offline test provider; no API
key or remote generation is used. The home page labels this preview as demo data.

Run `npm --prefix fractal-gas-web run test:landing-browser` for responsive layout,
static assets, project-prefix routes, and navigation. Set `LANDING_ARCADE_SMOKE=1`
with the Arcade engine and bundled ROM assets available to exercise startup and
return navigation under the existing isolation service worker.
