import { resolveAgentTypes, resolveBodies } from "./agent-types.js";
import { initHelp } from "../help.js";

export const ACTION_MULTIPLIER_MIN = 0;
export const ACTION_MULTIPLIER_MAX = 10;
export const ACTION_MULTIPLIER_DEFAULT = 1;

const pretty = (name) => name.replaceAll("_", " ");

function clampMultiplier(value) {
  return Math.max(
    ACTION_MULTIPLIER_MIN,
    Math.min(
      ACTION_MULTIPLIER_MAX,
      Number.isFinite(value) ? value : ACTION_MULTIPLIER_DEFAULT,
    ),
  );
}

function inferredChannels(actuator = {}) {
  switch (actuator.kind || "vector") {
    case "vector":
      return ["thrust", "torque"];
    case "kart":
      return ["throttle", "steering", "brake"];
    case "holonomic":
      return ["force_x", "force_y", "torque"];
    case "thrusters":
      return (actuator.thrusters || []).map((_, i) => `thruster_${i}`);
    default:
      return [];
  }
}

export function actionSettingGroups(scene, channels = []) {
  const types = resolveAgentTypes(scene.agent_types),
    bodies = resolveBodies(scene),
    groups = [];
  for (const [name, type] of types) {
    const bodyIndices = new Set(
      bodies.flatMap((body, index) =>
        body.agent_type === name && body.controlled ? [index] : [],
      ),
    );
    if (!bodyIndices.size) continue;
    const channelNames = [
      ...new Set(
        channels
          .filter((channel) => bodyIndices.has(channel.body))
          .map((channel) => channel.name),
      ),
    ];
    if (!channelNames.length)
      channelNames.push(...inferredChannels(type.physics?.actuator));
    groups.push({
      name,
      label: type.label || name,
      channels: channelNames,
    });
  }
  return groups;
}

export function actionMultiplierValues(
  scene,
  groups = actionSettingGroups(scene),
) {
  const types = resolveAgentTypes(scene.agent_types);
  return Object.fromEntries(
    groups.map(({ name, channels }) => {
      const actuator = types.get(name)?.physics?.actuator || {};
      const configured = actuator.action_multipliers || {};
      return [
        name,
        Object.fromEntries(
          channels.map((channel) => [
            channel,
            clampMultiplier(
              Number(configured[channel] ?? ACTION_MULTIPLIER_DEFAULT),
            ),
          ]),
        ),
      ];
    }),
  );
}

export function withActionMultipliers(scene, values) {
  const next = structuredClone(scene);
  if (!next.agent_types || typeof next.agent_types !== "object")
    throw new Error("Action multipliers require named agent types");
  for (const [name, channels] of Object.entries(values)) {
    const definition = next.agent_types[name];
    if (!definition) throw new Error(`Unknown agent type: ${name}`);
    if (!channels || typeof channels !== "object" || Array.isArray(channels))
      throw new Error(`Action multipliers for ${name} must be an object`);
    const resolved = resolveAgentTypes(scene.agent_types).get(name),
      inherited = resolved?.physics?.actuator || { kind: "vector" },
      ownPhysics = definition.physics || {},
      ownActuator = ownPhysics.actuator || {},
      actionMultipliers = {
        ...(inherited.action_multipliers || {}),
        ...(ownActuator.action_multipliers || {}),
      };
    for (const [channel, value] of Object.entries(channels)) {
      if (
        !Number.isFinite(value) ||
        value < ACTION_MULTIPLIER_MIN ||
        value > ACTION_MULTIPLIER_MAX
      )
        throw new RangeError(
          `${name} ${channel} multiplier must be between 0 and 10`,
        );
      actionMultipliers[channel] = value;
    }
    definition.physics = {
      ...ownPhysics,
      actuator: {
        ...inherited,
        ...ownActuator,
        action_multipliers: actionMultipliers,
      },
    };
  }
  return next;
}

const formatMultiplier = (value) => `${Number(value.toFixed(2))}×`;

export class ActionSettings {
  constructor(container, button, apply) {
    this.container = container;
    this.button = button;
    this.apply = apply;
    this.inputs = [];
    this.enabled = false;
    this.button.onclick = () => {
      const values = {};
      for (const input of this.inputs) {
        const value = Number(input.value);
        (values[input.dataset.agentType] ||= {})[input.dataset.channel] = value;
      }
      this.apply(values);
    };
    this.setEnabled(false);
  }
  render(scene, channels = []) {
    this.container.replaceChildren();
    this.inputs = [];
    const groups = actionSettingGroups(scene, channels),
      values = actionMultiplierValues(scene, groups),
      root = this.container.closest("details");
    if (root) root.hidden = !groups.length;
    for (const group of groups) {
      const fieldset = document.createElement("fieldset");
      fieldset.className = "action-type-settings";
      const legend = document.createElement("legend");
      legend.textContent = group.label;
      fieldset.append(legend);
      for (const channel of group.channels) {
        const label = document.createElement("label");
        label.className = "field action-multiplier-field";
        label.dataset.help = `Scale the native ${channel} action range for ${group.label}. 1× is the default; 0× disables this degree of freedom.`;
        label.textContent = pretty(channel);
        const row = document.createElement("span");
        row.className = "action-multiplier-inputs";
        const input = document.createElement("input");
        input.type = "range";
        input.min = ACTION_MULTIPLIER_MIN;
        input.max = ACTION_MULTIPLIER_MAX;
        input.step = 0.1;
        input.value = values[group.name][channel];
        input.dataset.agentType = group.name;
        input.dataset.channel = channel;
        input.setAttribute(
          "aria-label",
          `${group.label} ${pretty(channel)} multiplier`,
        );
        const output = document.createElement("output");
        output.textContent = formatMultiplier(+input.value);
        input.oninput = () => {
          output.textContent = formatMultiplier(+input.value);
        };
        row.append(input, output);
        label.append(row);
        fieldset.append(label);
        this.inputs.push(input);
      }
      this.container.append(fieldset);
    }
    initHelp(this.container);
    this.setEnabled(this.enabled);
  }
  setEnabled(enabled) {
    this.enabled = enabled;
    for (const input of this.inputs) input.disabled = !enabled;
    this.button.disabled = !enabled || !this.inputs.length;
  }
}
