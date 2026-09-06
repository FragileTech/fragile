import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { MotionRecording, WorldCapture, bytesOf } from "../web/lab/motion.js";
import { exportRecording, importRecording } from "../web/lab/archive.js";
import { resolveBodies, resolveAgentTypes } from "../web/lab/agent-types.js";
import {
    createAgentModel,
    animatedParts,
    animateAgent,
} from "../web/lab/visuals/registry.js";
import { BodyLayer } from "../web/lab/visuals/body-layer.js";
import { WorldPlayback } from "../web/lab/playback.js";
import * as T from "../web/lab/vendor/three.module.js";

const module = await loadNative(false);
const catalog = JSON.parse(
    await readFile(
        new URL("../web/lab/agent-catalog.json", import.meta.url),
        "utf8",
    ),
);

test("Harvester uses portable ground controls, collection and exact batch restoration", () => {
    const config = {
        size: [100, 100],
        task: "forage",
        agent_types: catalog,
        bodies: [{ agent_type: "harvester", position: [50, 50] }],
        pickups: [{ position: [50, 50], radius: 0.4 }],
        respawn_seconds: 0.1,
    };
    const flat = {
        ...config,
        agent_types: undefined,
        bodies: resolveBodies(config).map(({ agent_type, ...body }) => body),
    };
    const a = new NativeEngine(module, config, 8, 1),
        b = new NativeEngine(module, flat, 8, 1);
    try {
        assert.equal(a.dim, 3);
        assert.equal(a.words, b.words);
        const actions = Float32Array.from({ length: 24 }, (_, i) =>
            i % 3 === 0 ? 0.8 : i % 3 === 1 ? (i - 12) / 24 : 0,
        );
        a.step(actions, 12);
        b.step(actions, 12);
        assert.deepEqual(bytesOf(a.states()), bytesOf(b.states()));
        const rows = a.states();
        assert(new Uint32Array(rows.buffer, rows.byteOffset)[5] > 0);
        const saved = a.snapshot();
        a.step(actions, 40);
        const future = a.snapshot();
        a.restore(saved);
        a.step(actions, 40);
        assert.deepEqual(a.snapshot(), future);
        a.gather(Int32Array.from([1, 0, 2, 3, 4, 5, 6, 7]));
        a.gather(Int32Array.from([1, 0, 2, 3, 4, 5, 6, 7]));
        assert.deepEqual(a.snapshot(), future);
    } finally {
        a.dispose();
        b.dispose();
    }
});

test("Harvester fits the common footprint and restores its animated pose when seeking", () => {
    const model = createAgentModel(catalog.harvester.visual);
    const parts = animatedParts(model);
    assert.equal(
        parts.filter(({ part }) => part.userData.motion === "steer").length,
        2,
    );
    assert.equal(
        parts.filter(({ part }) => part.userData.motion === "wheel").length,
        7,
    );
    const bounds = new T.Box3().setFromObject(model);
    assert(bounds.min.x >= -0.8 && bounds.max.x <= 0.8);
    assert(bounds.min.y >= -0.8 && bounds.max.y <= 0.8);
    const state = { time: 2, speed: 1.5, thrust: 0.4, steer: 0.2 };
    animateAgent(parts, state);
    model.updateMatrixWorld(true);
    const pose = parts.map(({ part }) => [...part.matrixWorld.elements]);
    animateAgent(parts, { ...state, time: 10, steer: -0.8 });
    animateAgent(parts, state);
    model.updateMatrixWorld(true);
    assert.deepEqual(
        parts.map(({ part }) => [...part.matrixWorld.elements]),
        pose,
    );
});

const scene = {
    version: 1,
    size: [80, 80],
    physics: { dt: 1 / 60 },
    agent_types: {
        base: {
            physics: {
                controlled: true,
                radius: 0.5,
                mass: 2,
                drag: 0.2,
                thrust: 5,
            },
            visual: { model: "drone" },
        },
        scout: {
            extends: "base",
            physics: { mass: 1 },
            visual: { color: "#ffce70" },
        },
    },
    bodies: [
        { agent_type: "scout", position: [20, 20] },
        { cargo: true, position: [22, 20], velocity: [0.1, 0.3] },
    ],
    tethers: [{ a: 0, b: 1, rest_length: 2, stiffness: 4 }],
    pickups: [{ position: [20, 20], radius: 0.3 }],
    respawn_seconds: 0.1,
};

test("archetypes resolve consistently in JS and native; no state overhead", () => {
    const bodies = resolveBodies(scene),
        flat = {
            ...scene,
            agent_types: undefined,
            bodies: bodies.map(({ agent_type, ...body }) => body),
        };
    assert.equal(bodies[0].mass, 1);
    assert.equal(bodies[0].visual.model, "drone");
    assert.equal(scene.bodies[0].controlled, undefined);
    const a = new NativeEngine(module, scene, 8, 1),
        b = new NativeEngine(module, flat, 8, 1);
    try {
        assert.equal(a.dim, 2);
        assert.equal(a.words, b.words);
        const action = new Float32Array(16).fill(0.5);
        a.step(action, 8);
        b.step(action, 8);
        assert.deepEqual(bytesOf(a.states()), bytesOf(b.states()));
    } finally {
        a.dispose();
        b.dispose();
    }
    assert.throws(
        () => resolveAgentTypes({ a: { extends: "b" }, b: { extends: "a" } }),
        /Cyclic/,
    );
    assert.throws(
        () =>
            new NativeEngine(module, {
                ...scene,
                bodies: [{ agent_type: "missing" }],
            }),
        /Unknown/,
    );
});

test("complete motion replay restores moving cargo, tethers, food respawns and RNG exactly", () => {
    const e = new NativeEngine(module, scene),
        verify = new NativeEngine(module, scene);
    try {
        const record = new MotionRecording(e.info, e.snapshot()),
            capture = new WorldCapture(e, ({ packet, label }) =>
                record.append(packet, label),
            );
        const action = new Float32Array([0.75, 0.1]);
        capture.capture(new Float32Array(2), 0, "Initial");
        capture.step(action, 300, 1);
        assert.equal(record.length, 301);
        assert.equal(record.chunks.length, 2);
        const saved = e.snapshot();
        for (const i of [0, 1, 7, 150, 255, 256, 300]) {
            const frame = record.frame(i);
            assert.equal(frame.tick, i);
            verify.restoreRows(record.rows(i));
            assert.deepEqual(
                bytesOf(verify.states()).subarray(0, e.words * 4),
                bytesOf(frame.state),
            );
            if (i < 300) {
                verify.step(action, 1);
                assert.deepEqual(
                    bytesOf(verify.states()).subarray(0, e.words * 4),
                    bytesOf(record.frame(i + 1).state),
                );
            }
        }
        assert.notEqual(
            record.frame(0).state[8 + 1],
            record.frame(300).state[8 + 1],
        );
        assert.notDeepEqual(
            bytesOf(record.frame(0).state).subarray(e.info[8] * 4),
            bytesOf(record.frame(300).state).subarray(e.info[8] * 4),
        );
        const archive = exportRecording(scene, {}, [], record),
            loaded = importRecording(archive);
        assert.equal(loaded.recording.entries.length, 0);
        assert.equal(loaded.motion.length, 301);
        assert.deepEqual(loaded.motion.pack(), record.pack());
        verify.restore(loaded.motion.root);
        verify.restoreRows(loaded.motion.rows(300));
        assert.deepEqual(verify.snapshot(), saved);
        const corrupt = JSON.parse(archive);
        corrupt.motion.frames = corrupt.motion.frames.slice(0, -4) + "AAAA";
        assert.throws(
            () => importRecording(JSON.stringify(corrupt)),
            /checksum/,
        );
        corrupt.motion = JSON.parse(archive).motion;
        corrupt.motion.segments = [{ frame: 10000, label: "Bad" }];
        assert.throws(
            () => importRecording(JSON.stringify(corrupt)),
            /segments/,
        );
        const legacy = {
            version: 1,
            engine: "fractal-control-1",
            scene,
            settings: {},
            entries: [],
        };
        assert.equal(importRecording(JSON.stringify(legacy)).motion, undefined);
        e.restoreRows(record.rows(100));
        capture.capture(action, 2, "Continued from replay");
        capture.step(action, 3, 2);
        assert.equal(record.frame(301).tick, 100);
        assert.equal(record.frame(304).tick, 103);
        assert.equal(record.segment(303), "Continued from replay");
    } finally {
        e.dispose();
        verify.dispose();
    }
});

test("model factories and nested mixed crowds support deterministic animations", () => {
    for (const model of ["kart", "rocket", "drone"]) {
        const group = createAgentModel({ model }, 0x6ffff1);
        assert(group.children.length > 10);
        const parts = animatedParts(group);
        assert(parts.length > 0);
        const params = { time: 3, speed: 4, thrust: 0.7, steer: -0.5 };
        animateAgent(parts, params);
        group.updateMatrixWorld(true);
        const matrix = parts[0].part.matrixWorld.clone();
        animateAgent(parts, { ...params, time: 10 });
        animateAgent(parts, params);
        group.updateMatrixWorld(true);
        assert.deepEqual(parts[0].part.matrixWorld.elements, matrix.elements);
    }
    const kit = createAgentModel(
        { model: "kit", parts: [{ shape: "box", size: [1, 0.3, 0.2] }] },
        0xffffff,
    );
    assert.equal(kit.children.length, 1);
    assert.throws(() => createAgentModel({ model: "missing" }), /Unknown/);
    const config = {
        ...scene,
        bodies: Array.from({ length: 48 }, (_, i) => ({
            agent_type: "scout",
            position: [5 + (i % 8) * 3, 5 + Math.floor(i / 8) * 3],
            visual: { model: ["kart", "rocket", "drone"][i % 3] },
        })),
        tethers: [],
        pickups: [],
    };
    const e = new NativeEngine(module, config);
    try {
        const parent = new T.Group(),
            layer = new BodyLayer(config, e.info, parent);
        layer.update(e.states(), new Float32Array(96).fill(0.5));
        assert.equal(layer.controlled.length, 48);
        assert(layer.instances.length > 0);
        for (const { mesh } of layer.instances)
            for (const value of mesh.instanceMatrix.array)
                assert(Number.isFinite(value));
    } finally {
        e.dispose();
    }
});

test("playback clock seeks, changes speed and stops without mutating the world", () => {
    let pending, shown;
    globalThis.requestAnimationFrame = (cb) => ((pending = cb), 1);
    globalThis.cancelAnimationFrame = () => {
        pending = undefined;
    };
    const e = new NativeEngine(module, scene);
    try {
        const record = new MotionRecording(e.info, e.snapshot());
        const capture = new WorldCapture(e, ({ packet, label }) =>
            record.append(packet, label),
        );
        capture.capture(new Float32Array(2));
        capture.step(new Float32Array([1, 0]), 10, 1);
        const end = e.snapshot();
        const p = new WorldPlayback({ show: (f, i) => (shown = i) });
        p.attach(record);
        p.seek(5);
        assert.equal(shown, 5);
        p.speed = 2;
        p.play();
        pending(0);
        pending(50);
        assert.equal(shown, 10);
        assert.equal(p.playing, false);
        assert.deepEqual(e.snapshot(), end);
        p.live();
        assert.equal(p.active, false);
    } finally {
        e.dispose();
        delete globalThis.requestAnimationFrame;
        delete globalThis.cancelAnimationFrame;
    }
});
