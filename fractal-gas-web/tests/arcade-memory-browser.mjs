import assert from "node:assert/strict";
import { chromium, firefox } from "playwright";
const engine = process.env.ARCADE_BROWSER === "firefox" ? firefox : chromium;
const browser = await engine.launch({
  headless: true,
  args: engine === chromium ? ["--no-sandbox"] : [],
});
try {
  const page = await browser.newPage();
  await page.goto(process.env.ARCADE_TEST_URL || "http://127.0.0.1:8091/web/");
  const boundary = await page.evaluate(async () => {
    const script = `import create from '${new URL("fractal_gas.js", location.href).href}';
   const results=[];
   const rom=new Uint8Array(await(await fetch('${new URL("test-rom.nes", location.href).href}')).arrayBuffer());
   const params={n:2,distCoef:1,rewardCoef:1,useCumulativeReward:true,dtMin:1,dtMax:2,nElite:2,seed:7,nThreads:2,memoryLimitBytes:4294967296,obsMode:0,world:1,stage:1,console:0,game:0,algorithm:0,horizon:32,consensusPrefix:true,maxHorizon:0,maxWalkers:0,freezePrefixAfter:0,eraseCoef:0.05,aggBlock:5,visitReward:false,visitCoef:1,farmPtr:0,farmWorkers:0,farmBlobLen:0};
   for(const maximum of [8192,65536]) {
    const memory=new WebAssembly.Memory({initial:8192,maximum,shared:true});
    const fg=await create({wasmMemory:memory,arcadeThreadPoolSize:1});
    if(maximum===8192){const ptr=fg._malloc(600*1024**2)>>>0;results.push({capped:ptr===0,bytes:memory.buffer.byteLength});if(ptr)fg._free(ptr);
     // Bypass the estimator deliberately to force a real allocator failure
     // inside NES batch stepping, with an imported 512 MiB hard ceiling.
     if(!fg.init(rom,new Uint8Array(0),{...params,n:256,nElite:0,obsMode:3}))throw new Error(fg.lastError());
     const held=[];for(;;){const block=fg._malloc(1024**2)>>>0;if(!block)break;held.push(block);}
     fg._free(held.pop());
     const failed=fg.step();results[0].threadAllocationError=failed.error;
     for(const block of held)fg._free(block);
    }
    else {const big=fg._malloc(2*1024**3+32*1024**2)>>>0; const high=fg._malloc(65536)>>>0;
     if(!big||!high)throw new Error('High address allocation failed');
     const view=new Uint8Array(memory.buffer,high,65536);view[0]=73;view[65535]=91;
     results.push({high,bytes:memory.buffer.byteLength,readback:view[0]+view[65535]});

     if(!fg.init(rom,new Uint8Array(0),params))throw new Error(fg.lastError());
     const stats=fg.step();if(stats.error)throw new Error(stats.error);
     results[results.length-1].threadedFrames=stats.totalFrames;
     fg._free(high);fg._free(big);}
    fg.PThread.terminateAllThreads();
   }
   const {default:createShim}=await import('${new URL("retro_shim.js", location.href).href}');
   const shimMemory=new WebAssembly.Memory({initial:1024,maximum:2048});
   const shim=await createShim({wasmMemory:shimMemory});
   const allocation=shim._malloc(80*1024**2)>>>0;
   if(!allocation)throw new Error('Emulator memory did not grow');
   const refused=shim._malloc(64*1024**2)>>>0;
   results.push({emulatorBytes:shimMemory.buffer.byteLength,capped:refused===0});
   postMessage(results);`;
    const url = URL.createObjectURL(
      new Blob([script], { type: "text/javascript" }),
    );
    const w = new Worker(url, { type: "module" });
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        w.terminate();
        reject(new Error("Memory boundary worker timed out"));
      }, 180000);
      w.onmessage = (e) => {
        clearTimeout(timer);
        w.terminate();
        URL.revokeObjectURL(url);
        resolve(e.data);
      };
      w.onerror = (e) => {
        clearTimeout(timer);
        w.terminate();
        reject(new Error(e.message));
      };
    });
  });
  assert.equal(boundary[0].capped, true);
  assert.match(boundary[0].threadAllocationError, /bad_alloc/);
  assert.equal(boundary[0].bytes, 512 * 1024 ** 2);
  assert.ok(boundary[1].high > 2 * 1024 ** 3);
  assert.ok(boundary[1].bytes > 2 * 1024 ** 3);
  assert.equal(boundary[1].readback, 164);
  assert.ok(boundary[1].threadedFrames > 0);
  assert.ok(boundary[2].emulatorBytes > 64 * 1024 ** 2);
  assert.ok(boundary[2].emulatorBytes <= 128 * 1024 ** 2);
  assert.equal(boundary[2].capped, true);
  console.log("memory boundaries", boundary);
  const lifecycle = await page.evaluate(async () => {
    const w = new Worker("worker.js", { type: "module" }),
      messages = [];
    w.onmessage = ({ data }) => messages.push(data);
    let failure;
    w.onerror = (e) => (failure = e.message);
    async function wait(type, after = 0) {
      const deadline = performance.now() + 180000;
      while (true) {
        const incoming = messages.slice(after);
        const error = incoming.find((m) => m.type === "error");
        if (error) throw new Error(error.message);
        if (failure) throw new Error(failure);
        const m = incoming.find((m) => m.type === type);
        if (m) return m;
        if (performance.now() > deadline)
          throw new Error("Timeout waiting for " + type);
        await new Promise((r) => setTimeout(r, 10));
      }
    }
    const rom = await (await fetch("sonic.rom")).arrayBuffer();
    const params = {
      n: 1000,
      nThreads: 20,
      distCoef: 1,
      rewardCoef: 1,
      useCumulativeReward: true,
      dtMin: 1,
      dtMax: 1,
      nElite: 2,
      seed: 7,
      obsMode: 3,
      console: 2,
      game: 1,
      world: 0,
      stage: 0,
      algorithm: 0,
    };
    const output = [];
    for (const n of [1000, 2, 48, 2]) {
      const offset = messages.length;
      w.postMessage({
        type: "init",
        rom,
        params: { ...params, n },
        resources: { workers: 20, memoryGiB: 8 },
      });
      const ready = await wait("ready", offset);
      const stepOffset = messages.length;
      w.postMessage({ type: "start" });
      const step = await wait("step", stepOffset);
      w.postMessage({ type: "pause" });
      output.push({ n, ready: ready.resources, step: step.resources });
    }
    let offset = messages.length;
    w.postMessage({ type: "reset" });
    await wait("resetDone", offset);
    output.push({ reset: (await wait("ready", offset)).resources });
    offset = messages.length;
    w.postMessage({ type: "dispose" });
    await wait("disposed", offset);
    w.terminate();
    return output;
  });
  for (const r of lifecycle) {
    if (r.n) {
      assert.equal(r.ready.workers, Math.min(20, r.n));
      assert.equal(r.step.workers, r.ready.workers);
      if (r.n === 2) assert.equal(r.ready.mainBytes, 512 * 1024 ** 2);
    }
  }
  assert.equal(lifecycle.at(-1).reset.mainBytes, 512 * 1024 ** 2);
  console.log("lifecycle", JSON.stringify(lifecycle));
} finally {
  await browser.close();
}
