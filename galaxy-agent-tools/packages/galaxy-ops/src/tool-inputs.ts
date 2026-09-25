/**
 * Pure helpers for diagnosing and scaffolding Galaxy tool inputs.
 *
 * No network, no Galaxy client, no global state. Faithful port of
 * mcp-server-galaxy-py/src/galaxy_mcp/ops/tool_inputs.py -- the Python is
 * authoritative for key formats and placeholder literals.
 */

const MAX_OPTIONS = 25;

/** Galaxy options are [label, value, selected] triples. Extract just the value. */
function optionValues(p: Record<string, unknown>): unknown[] {
  const options = (p["options"] as unknown[] | null | undefined) ?? [];
  return (options as unknown[]).slice(0, MAX_OPTIONS).map((o) =>
    Array.isArray(o) && o.length > 1 ? o[1] : o,
  );
}

function optionsTruncated(p: Record<string, unknown>): boolean {
  const options = (p["options"] as unknown[] | null | undefined) ?? [];
  return options.length > MAX_OPTIONS;
}

function placeholder(p: Record<string, unknown>): unknown {
  const ptype = p["type"] as string | undefined;
  if (ptype === "data") return { src: "hda", id: "<dataset_id>" };
  if (ptype === "data_collection") return { src: "hdca", id: "<collection_id>" };
  if (ptype === "select") {
    const choices = optionValues(p);
    return choices.length > 0 ? choices[0] : "<choice>";
  }
  if (ptype === "boolean") return false;
  if (ptype === "integer") return 0;
  if (ptype === "float") return 0;
  return "<value>";
}

function fillParam(p: Record<string, unknown>, prefix: string, out: Record<string, unknown>): void {
  const name = p["name"] as string | undefined;
  if (name == null) return;
  const key = `${prefix}${name}`;
  const ptype = p["type"] as string | undefined;

  if (ptype === "repeat") {
    const children = (p["inputs"] as Record<string, unknown>[] | undefined) ?? [];
    for (const child of children) fillParam(child, `${key}_0|`, out);
  } else if (ptype === "section") {
    const children = (p["inputs"] as Record<string, unknown>[] | undefined) ?? [];
    for (const child of children) fillParam(child, `${key}|`, out);
  } else if (ptype === "conditional") {
    const tp = (p["test_param"] as Record<string, unknown> | undefined) ?? {};
    const tpName = tp["name"] as string | undefined;
    const cases = (p["cases"] as Record<string, unknown>[] | undefined) ?? [];
    const first = cases[0] ?? null;
    const selValue = first != null ? (first["value"] as unknown) : "<choice>";
    if (tpName) out[`${key}|${tpName}`] = selValue;
    if (first) {
      const firstInputs = (first["inputs"] as Record<string, unknown>[] | undefined) ?? [];
      for (const child of firstInputs) fillParam(child, `${key}|`, out);
    }
  } else {
    out[key] = placeholder(p);
  }
}

/**
 * Build a ready-to-fill flattened `inputs` skeleton from a tool's `inputs` array.
 *
 * Data params -> `{"src": "hda", "id": "<dataset_id>"}`, selects -> first choice,
 * conditionals -> first-case selector + that branch's params, repeats -> one
 * `name_0|...` instance, sections -> `name|...`.
 */
export function buildInputTemplate(inputs: unknown[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of inputs) {
    if (p != null && typeof p === "object") {
      fillParam(p as Record<string, unknown>, "", out);
    }
  }
  return out;
}

function summarizeParam(p: Record<string, unknown>): Record<string, unknown> {
  const ptype = (p["type"] as string | undefined) ?? (p["model_class"] as string | undefined);
  const out: Record<string, unknown> = { name: p["name"], type: ptype };
  if (p["optional"] != null) out["optional"] = p["optional"];

  if (ptype === "repeat") {
    out["repeat_key_hint"] = `${p["name"]}_0|<param>`;
    const children = (p["inputs"] as Record<string, unknown>[] | undefined) ?? [];
    out["children"] = children.map(summarizeParam);
  } else if (ptype === "section") {
    out["section_key_hint"] = `${p["name"]}|<param>`;
    const children = (p["inputs"] as Record<string, unknown>[] | undefined) ?? [];
    out["children"] = children.map(summarizeParam);
  } else if (ptype === "conditional") {
    const tp = (p["test_param"] as Record<string, unknown> | undefined) ?? {};
    const selector: Record<string, unknown> = {
      name: tp["name"],
      type: tp["type"],
      choices: optionValues(tp),
      key_hint: `${p["name"]}|${tp["name"]}`,
    };
    if (optionsTruncated(tp)) selector["choices_truncated"] = true;
    out["selector"] = selector;
    const cases = (p["cases"] as Record<string, unknown>[] | undefined) ?? [];
    out["cases"] = cases.map((c) => ({
      when: c["value"],
      params: ((c["inputs"] as Record<string, unknown>[] | undefined) ?? []).map(summarizeParam),
    }));
  } else if (ptype === "select") {
    out["choices"] = optionValues(p);
    if (optionsTruncated(p)) out["choices_truncated"] = true;
  }

  return out;
}

/**
 * Compact a tool's io_details `inputs` array into a model-friendly parameter list.
 *
 * Preserves the nesting that matters for building flattened input keys
 * (repeats -> `name_0|param`, conditionals -> `name|selector`, sections ->
 * `name|param`) without the full Galaxy schema noise.
 */
export function summarizeToolInputs(inputs: unknown[]): unknown[] {
  return inputs
    .filter((p): p is Record<string, unknown> => p != null && typeof p === "object")
    .map(summarizeParam);
}
