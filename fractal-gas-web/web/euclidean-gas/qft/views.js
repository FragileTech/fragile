// HTML builders of the QFT Simulator workbench. They return strings and touch
// no DOM, so they can be checked from node; `initTabs` is the only DOM helper.
import {
  chartSVG,
  legendHTML,
  escapeXML as esc,
  format,
} from "../lecture/plots.js";
import {
  GEVP_BASIS,
  VOCABULARY,
  catalogFamilies,
  chartTable,
  gevpReady,
} from "./model.js";

const cell = (value) =>
  value === null || value === undefined
    ? "—"
    : typeof value === "number"
      ? format(value)
      : String(value);

export function optionsHTML(pairs, selected) {
  return pairs
    .map(
      ([value, label]) =>
        '<option value="' +
        esc(value) +
        '"' +
        (String(value) === String(selected) ? " selected" : "") +
        ">" +
        esc(label) +
        "</option>",
    )
    .join("");
}

// Notes are Rust's interpretation of its own numbers: escaped, never edited.
export function notesHTML(
  notes = [],
  heading = "Notes from the Rust analysis",
) {
  if (!notes.length) return "";
  return (
    '<aside class="notes" aria-label="' +
    esc(heading) +
    '"><h3>' +
    esc(heading) +
    "</h3><ul>" +
    notes
      .map((note) =>
        typeof note === "string"
          ? "<li>" + esc(note) + "</li>"
          : '<li><span class="note-owner">' +
            esc(note.owner) +
            "</span>" +
            esc(note.text) +
            "</li>",
      )
      .join("") +
    "</ul></aside>"
  );
}

export function tableHTML(table, { caption = "", limit = 400 } = {}) {
  if (!table?.columns?.length) return "";
  const rows = table.rows.slice(0, limit);
  return (
    '<div class="table-scroll" tabindex="0" role="region" aria-label="' +
    esc(caption || "Numeric table") +
    '"><table>' +
    (caption ? "<caption>" + esc(caption) + "</caption>" : "") +
    "<thead><tr>" +
    table.columns.map((c) => '<th scope="col">' + esc(c) + "</th>").join("") +
    "</tr></thead><tbody>" +
    (rows.length
      ? rows
          .map(
            (row) =>
              "<tr>" +
              row.map((v) => "<td>" + esc(cell(v)) + "</td>").join("") +
              "</tr>",
          )
          .join("")
      : '<tr><td colspan="' +
        table.columns.length +
        '">No rows reported.</td></tr>') +
    "</tbody></table></div>" +
    (table.rows.length > limit
      ? '<p class="footnote">Showing ' +
        limit +
        " of " +
        table.rows.length +
        " rows; export the evidence for the full data.</p>"
      : "")
  );
}

// A chart card: SVG, legend, "SVG" export button and the numeric fallback.
export function chartCardsHTML(charts, { width = 640, group = "chart" } = {}) {
  return charts
    .map(
      (chart, index) =>
        '<section class="chart"><div class="chart-heading"><h3>' +
        esc(chart.title) +
        '</h3><button type="button" class="svg-export" data-export="' +
        esc(group) +
        ":" +
        index +
        '" aria-label="Save ' +
        esc(chart.title) +
        ' as SVG">SVG ↓</button></div>' +
        chartSVG(chart, { width }) +
        '<div class="legend">' +
        legendHTML(chart) +
        '</div><details class="numeric"><summary>Numeric values</summary>' +
        tableHTML(chartTable(chart), { caption: chart.title, limit: 200 }) +
        "</details></section>",
    )
    .join("");
}

// ---------------------------------------------------------------- setup

export function variantOptionsHTML(variants = [], selected) {
  return variants
    .map(
      (v) =>
        '<option value="' +
        esc(v.name) +
        '"' +
        (v.implemented ? "" : " disabled") +
        (v.name === selected ? " selected" : "") +
        ">" +
        esc(v.title) +
        (v.implemented ? "" : " (book only)") +
        "</option>",
    )
    .join("");
}
export function variantListHTML(variants = []) {
  return (
    '<dl class="variants">' +
    variants
      .map(
        (v) =>
          "<dt>" +
          esc(v.title) +
          ' <span class="tag' +
          (v.implemented ? "" : " tag-muted") +
          '">' +
          (v.implemented ? "implemented" : "book only") +
          "</span></dt><dd>" +
          esc(v.summary || "") +
          (v.book_label
            ? ' <span class="book-label">' + esc(v.book_label) + "</span>"
            : "") +
          "</dd>",
      )
      .join("") +
    "</dl>"
  );
}

export function colorFieldsHTML(color) {
  if (color.kind === "recorded_field")
    return ["stage", "amplitude", "phase"]
      .map(
        (key) =>
          '<label class="control"><span>Recorded ' +
          key +
          '</span><input type="text" name="color-' +
          key +
          '" value="' +
          esc(color[key] ?? "") +
          '" autocomplete="off"></label>',
      )
      .join("");
  const alignment = color.alignment?.kind ?? "preceding_kick";
  return (
    '<label class="control"><span>Colour alignment</span><select name="alignment">' +
    optionsHTML(VOCABULARY.alignment, alignment) +
    "</select></label>" +
    (alignment === "matched_kick"
      ? '<label class="control"><span>Matched B stage</span><select name="stage">' +
        optionsHTML(VOCABULARY.stage, color.alignment.stage ?? "b1") +
        "</select></label>"
      : "")
  );
}

function requirementText(requirements) {
  if (!requirements) return "";
  const records = (requirements.records || []).join(", ");
  return (
    (records || "no record") +
    (requirements.dimension != null ? " · d = " + requirements.dimension : "")
  );
}

const NORMALIZATION_LABEL = {
  valid_count: "sum of valid element weights",
  fixed_n: "population size N",
};

export function channelListHTML(catalog, selected, availability) {
  const chosen = new Set(selected);
  return catalogFamilies(catalog)
    .map(
      ({ family, channels }) =>
        '<fieldset class="family"><legend>' +
        esc(family) +
        "</legend><ul>" +
        channels
          .map((entry, index) => {
            // `CatalogEntry` = {id, spec, kind, family, standard,
            // availability, signature, assignment}. The signature is the one
            // of `Capabilities::nominal()` and is null for a channel that
            // does not exist at all, so every read of it is optional.
            const signature = entry.signature || {};
            const text = signature.descriptor || {};
            const state = availability.get(entry.id);
            const blocked = state?.available === false;
            const reasonId = "reason-" + family + "-" + index;
            return (
              '<li class="channel' +
              (blocked ? " unavailable" : "") +
              '"><label><input type="checkbox" name="channel" value="' +
              esc(entry.id) +
              '"' +
              (chosen.has(entry.id) && !blocked ? " checked" : "") +
              (blocked
                ? ' disabled aria-describedby="' + esc(reasonId) + '"'
                : "") +
              "><code>" +
              esc(entry.id) +
              '</code></label><code class="definition">' +
              esc(text.definition || "") +
              '</code><div class="channel-meta">' +
              (text.book_label
                ? "<span>Book: " + esc(text.book_label) + "</span>"
                : "<span>Not a book operator</span>") +
              (entry.kind
                ? "<span>Elements: " + esc(entry.kind) + "</span>"
                : "") +
              "<span>Exchange: " +
              esc(signature.exchange ?? "—") +
              "</span>" +
              (text.spatial_parity
                ? "<span>Spatial parity: " +
                  esc(text.spatial_parity) +
                  "</span>"
                : "") +
              (signature.requires
                ? "<span>Requires: " +
                  esc(requirementText(signature.requires)) +
                  "</span>"
                : "") +
              (signature.normalization
                ? "<span>Frame normalisation: " +
                  esc(
                    NORMALIZATION_LABEL[signature.normalization] ??
                      signature.normalization,
                  ) +
                  "</span>"
                : "") +
              (signature.correlatable === false
                ? "<span>No correlator</span>"
                : "") +
              (entry.assignment
                ? "<span>Assigned: " + esc(entry.assignment) + "</span>"
                : "") +
              "</div>" +
              (text.note
                ? '<p class="channel-note">' + esc(text.note) + "</p>"
                : "") +
              (blocked
                ? '<p class="channel-reason" id="' +
                  esc(reasonId) +
                  '">Unavailable: ' +
                  esc(state.reason) +
                  "</p>"
                : state
                  ? '<p class="channel-ok">Available for this configuration</p>'
                  : "") +
              "</li>"
            );
          })
          .join("") +
        "</ul></fieldset>",
    )
    .join("");
}

export function capabilitySummaryHTML(response, missing) {
  if (!response) return "";
  const c = response.capabilities || {};
  const facts = [
    ["Position dimension", c.dimension],
    ["Mutual distance pairing", c.mutual_distance],
    ["Mutual cloning pairing", c.mutual_cloning],
    ["Euclidean-time axis", c.euclidean_axis ?? "none declared"],
    ["Distance kernel width ε_d", c.distance_kernel_width ?? "—"],
    ["Cloning kernel width ε_c", c.cloning_kernel_width ?? "—"],
    ["Dense viscosity", c.dense_viscosity],
    ["Integrator step dt", c.time_step ?? "—"],
    ["Updates per recording chunk", response.chunk],
  ].filter(([, value]) => value !== undefined);
  return (
    '<dl class="facts">' +
    facts
      .map(
        ([label, value]) =>
          "<div><dt>" +
          esc(label) +
          "</dt><dd>" +
          esc(cell(value)) +
          "</dd></div>",
      )
      .join("") +
    "</dl>" +
    (missing.length
      ? tableHTML(
          { columns: ["Missing record", "Reason"], rows: missing },
          { caption: "Records this configuration does not provide" },
        )
      : "") +
    notesHTML(response.notes || [], "Capability notes from Rust")
  );
}

// ------------------------------------------------------------- analysis

export function analysisControlsHTML(controls) {
  const select = (name, label, pairs, value) =>
    '<label class="control"><span>' +
    esc(label) +
    '</span><select name="' +
    name +
    '">' +
    optionsHTML(pairs, value) +
    "</select></label>";
  const number = (name, label, value, min) =>
    '<label class="control"><span>' +
    esc(label) +
    '</span><input type="number" name="' +
    name +
    '" min="' +
    min +
    '" step="1" value="' +
    esc(value) +
    '"></label>';
  return (
    select("estimator", "Estimator", VOCABULARY.estimator, controls.estimator) +
    select(
      "resampling",
      "Resampling",
      VOCABULARY.resampling,
      controls.resampling,
    ) +
    (controls.resampling === "uncorrelated"
      ? ""
      : select(
          "block",
          "Block size",
          [
            ["auto", "Automatic (at least 2 τ_int)"],
            ["fixed", "Fixed"],
          ],
          controls.block,
        ) +
        (controls.block === "fixed"
          ? number("blockFrames", "Frames per block", controls.blockFrames, 1)
          : "")) +
    (controls.resampling === "bootstrap"
      ? number("samples", "Bootstrap samples", controls.samples, 2) +
        number("resampleSeed", "Bootstrap seed", controls.resampleSeed, 0)
      : "") +
    select("combine", "Replicas", VOCABULARY.combine, controls.combine) +
    select(
      "effectiveRate",
      "Effective rate",
      VOCABULARY.effectiveRate,
      controls.effectiveRate,
    ) +
    select("timeUnit", "Time unit", VOCABULARY.timeUnit, controls.timeUnit) +
    '<label class="check"><input type="checkbox" name="connected"' +
    (controls.connected ? " checked" : "") +
    "> Subtract the disconnected part</label>"
  );
}

// `analysed` is the number of channels the report carries a correlator for:
// Rust solves a basis of GEVP_BASIS.min..=GEVP_BASIS.max of them, so outside
// that range the control is disabled and says which bound was missed.
export function fitControlsHTML(controls, analysed = 0) {
  const ready = gevpReady(analysed);
  return (
    '<label class="control"><span>Fit method</span><select name="fit">' +
    optionsHTML(VOCABULARY.fit, controls.fit) +
    '</select></label><label class="check"><input type="checkbox" name="stability"' +
    (controls.stability ? " checked" : "") +
    '> Stability scan (Rust default grid)</label><label class="check"><input type="checkbox" name="gevp"' +
    (controls.gevp && ready ? " checked" : "") +
    (ready ? "" : " disabled") +
    "> GEVP over the analysed channels" +
    (ready
      ? ""
      : " (Rust solves " +
        GEVP_BASIS.min +
        " to " +
        GEVP_BASIS.max +
        " channels; " +
        analysed +
        " analysed)") +
    "</label>"
  );
}

export function assignmentsHTML(keys, references, controls) {
  const names = [["", "— not assigned —"], ...references.map((r) => [r, r])];
  return (
    '<fieldset class="mapping"><legend>Hypothesis mapping · channel → reference</legend>' +
    '<p class="footnote">The assignment is an input you choose, not a result of the run.</p>' +
    '<div class="control-grid">' +
    keys
      .map(
        (key) =>
          '<label class="control"><span><code>' +
          esc(key) +
          '</code></span><select name="assign" data-channel="' +
          esc(key) +
          '">' +
          optionsHTML(names, controls.assignments[key] ?? "") +
          "</select></label>",
      )
      .join("") +
    '</div></fieldset><fieldset class="mapping"><legend>Anchors · references that set the scale</legend><div class="anchor-list">' +
    references
      .map(
        (name) =>
          '<label class="check"><input type="checkbox" name="anchor" value="' +
          esc(name) +
          '"' +
          (controls.anchors.includes(name) ? " checked" : "") +
          "> " +
          esc(name) +
          "</label>",
      )
      .join("") +
    "</div></fieldset>"
  );
}

export function comparisonHTML(tables) {
  if (!tables)
    return '<p class="observation">Rust reported no comparison for this analysis.</p>';
  return (
    '<p class="hypothesis-label">' +
    esc(tables.label) +
    "</p>" +
    tableHTML(tables.reference, {
      caption: "Reference table · " + tables.label,
    }) +
    tables.anchors
      .map(
        (a) =>
          "<h3>Anchor: " +
          esc(a.anchor) +
          '</h3><p class="footnote">Reference units per lattice unit: ' +
          esc(a.scale) +
          "</p>" +
          tableHTML(a.table, {
            caption:
              "Predictions anchored on " + a.anchor + " · " + tables.label,
          }),
      )
      .join("") +
    "<h3>Ratios</h3>" +
    tableHTML(tables.ratios, { caption: "Ratio table · " + tables.label }) +
    "<h3>Anchor spread</h3>" +
    tableHTML(tables.spread, { caption: "Spread of predictions over anchors" })
  );
}

export function couplingsHTML(tables) {
  if (!tables) return "";
  return (
    "<h3>Algorithmic scales</h3>" +
    tableHTML(tables.scales, { caption: "Algorithmic scales" }) +
    "<h3>Couplings</h3>" +
    tableHTML(tables.couplings, { caption: "Couplings derived by the book" }) +
    "<h3>Inversion · calibration inputs, not measurements</h3>" +
    tableHTML(tables.inversion, {
      caption: "Standard Model inputs mapped to gas parameters",
    })
  );
}

// ----------------------------------------------------------------- tabs

// WAI-ARIA tabs with automatic activation and a roving tabindex.
export function initTabs(tablist, onSelect) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const panel = (tab) =>
    tablist.ownerDocument.getElementById(tab.getAttribute("aria-controls"));
  function select(tab, focus) {
    for (const other of tabs) {
      const on = other === tab;
      other.setAttribute("aria-selected", String(on));
      other.tabIndex = on ? 0 : -1;
      panel(other).hidden = !on;
    }
    if (focus) tab.focus();
    onSelect?.(tab.dataset.tab);
  }
  tablist.addEventListener("click", (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab) select(tab, false);
  });
  tablist.addEventListener("keydown", (event) => {
    const current = tabs.indexOf(event.target.closest('[role="tab"]'));
    if (current < 0) return;
    const target = {
      ArrowRight: (current + 1) % tabs.length,
      ArrowLeft: (current - 1 + tabs.length) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
    }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    select(tabs[target], true);
  });
  return {
    select: (name, focus = false) => {
      const tab = tabs.find((t) => t.dataset.tab === name);
      if (tab) select(tab, focus);
    },
  };
}
