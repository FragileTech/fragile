// Presets embed their types so exported scenes are portable. New global types
// become available in the editor on build, while scene-local definitions win.
import { readFile, writeFile } from "node:fs/promises";

export async function syncAgentCatalog(
  lab = new URL("../web/lab/", import.meta.url),
) {
  const readJSON = async (name) =>
    JSON.parse(await readFile(new URL(name, lab), "utf8"));
  const catalog = await readJSON("agent-catalog.json");
  for (const { id } of await readJSON("scenario-catalog.json")) {
    const path = new URL(`scenarios/${id}.json`, lab);
    const scene = JSON.parse(await readFile(path, "utf8"));
    const local = scene.agent_types || {};
    const missing = Object.keys(catalog).filter(
      (name) => !Object.hasOwn(local, name),
    );
    if (!missing.length) continue;
    scene.agent_types = { ...local };
    for (const name of missing) scene.agent_types[name] = catalog[name];
    await writeFile(path, JSON.stringify(scene, null, 2) + "\n");
  }
}
