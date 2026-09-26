import create from "./engine/optimization.mjs";
import {
  NativeOptimization,
  NativePopulationCoordinator,
  populationJson,
} from "./native.js";
const ready = create();
let engine,
  coordinator,
  queue = Promise.resolve();
self.onmessage = ({ data: message }) => {
  queue = queue.then(async () => {
    try {
      const module = await ready;
      engine ??= new NativeOptimization(module);
      coordinator ??= new NativePopulationCoordinator(module);
      let result;
      const exchange = (request) =>
        populationJson(module, "_fgo_exchange", engine.handle, request);
      switch (message.type) {
        case "catalog":
          result = engine.catalog();
          break;
        case "coordinatorCreate":
          result = coordinator.create(message.config);
          break;
        case "coordinator":
          result = coordinator.request(message.request);
          break;
        case "create":
          engine.create(message.member.settings);
          exchange({
            op: "configure",
            id: message.member.id,
            count: message.member.exchange_count,
          });
          result = {
            report: exchange({ op: "capture" }),
            frame: engine.snapshot(),
          };
          break;
        case "sync":
          exchange({ op: "sync", archive: message.archive });
          result = exchange({ op: "capture" });
          break;
        case "step":
          engine.step();
          result = {
            report: exchange({ op: "capture" }),
            frame: engine.snapshot(),
          };
          break;
        case "stage":
          result = exchange({ op: "stage", imports: message.imports });
          break;
        case "commit":
          exchange({ op: "commit" });
          result = { frame: engine.snapshot() };
          break;
        case "discard":
          result = exchange({ op: "discard" });
          break;
        case "settings":
          engine.updateSettings(message.patch);
          result = {
            report: exchange({ op: "capture" }),
            frame: engine.snapshot(),
          };
          break;
        case "sample":
          result = {
            values: engine.sample(message.positions, message.dimension),
          };
          break;
        default:
          throw new Error("Unknown population worker request");
      }
      self.postMessage(
        { id: message.id, result },
        [result?.frame?.buffer, result?.values?.buffer].filter(Boolean),
      );
    } catch (error) {
      self.postMessage({ id: message.id, error: error.message });
    }
  });
};
