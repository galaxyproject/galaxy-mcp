import { describe, it, expect } from "vitest";
import { buildInputTemplate, summarizeToolInputs } from "../src/tool-inputs";

/**
 * Fixture exercising: data, select (with options), boolean, integer,
 * repeat, section, and conditional params.
 *
 * Galaxy options are [label, value, selected] triples.
 */
const TOOL_INPUTS = [
  // simple data param
  { name: "input_file", type: "data" },
  // select with 3 options
  {
    name: "format",
    type: "select",
    options: [
      ["FASTQ", "fastq", true],
      ["FASTA", "fasta", false],
      ["BAM", "bam", false],
    ],
  },
  // boolean and integer
  { name: "paired", type: "boolean" },
  { name: "threads", type: "integer" },
  // repeat with one child
  {
    name: "adapters",
    type: "repeat",
    inputs: [{ name: "seq", type: "data" }],
  },
  // section with two children
  {
    name: "advanced",
    type: "section",
    inputs: [
      { name: "min_length", type: "integer" },
      { name: "db", type: "data" },
    ],
  },
  // conditional: test_param is a select; first case has one child
  {
    name: "quality_filter",
    type: "conditional",
    test_param: {
      name: "method",
      type: "select",
      options: [
        ["Phred", "phred", true],
        ["Solexa", "solexa", false],
      ],
    },
    cases: [
      {
        value: "phred",
        inputs: [{ name: "cutoff", type: "integer" }],
      },
      {
        value: "solexa",
        inputs: [{ name: "score", type: "float" }],
      },
    ],
  },
];

describe("buildInputTemplate", () => {
  it("produces exact key paths and placeholder values", () => {
    const tpl = buildInputTemplate(TOOL_INPUTS);

    // data
    expect(tpl["input_file"]).toEqual({ src: "hda", id: "<dataset_id>" });

    // select -> first option value
    expect(tpl["format"]).toBe("fastq");

    // boolean -> false
    expect(tpl["paired"]).toBe(false);

    // integer -> 0
    expect(tpl["threads"]).toBe(0);

    // repeat child under name_0| prefix
    expect(tpl["adapters_0|seq"]).toEqual({ src: "hda", id: "<dataset_id>" });
    // the parent key itself is NOT emitted
    expect("adapters" in tpl).toBe(false);

    // section children under name| prefix
    expect(tpl["advanced|min_length"]).toBe(0);
    expect(tpl["advanced|db"]).toEqual({ src: "hda", id: "<dataset_id>" });
    expect("advanced" in tpl).toBe(false);

    // conditional: selector key + first case children
    expect(tpl["quality_filter|method"]).toBe("phred");
    expect(tpl["quality_filter|cutoff"]).toBe(0);
    // second case children NOT emitted
    expect("quality_filter|score" in tpl).toBe(false);
    expect("quality_filter" in tpl).toBe(false);
  });

  it("float placeholder is 0 (matching Python 0.0)", () => {
    const tpl = buildInputTemplate([{ name: "x", type: "float" }]);
    expect(tpl["x"]).toBe(0);
  });

  it("unknown type gets <value> placeholder", () => {
    const tpl = buildInputTemplate([{ name: "misc", type: "color" }]);
    expect(tpl["misc"]).toBe("<value>");
  });

  it("data_collection gets hdca placeholder", () => {
    const tpl = buildInputTemplate([{ name: "col", type: "data_collection" }]);
    expect(tpl["col"]).toEqual({ src: "hdca", id: "<collection_id>" });
  });

  it("select with no options falls back to <choice>", () => {
    const tpl = buildInputTemplate([{ name: "sel", type: "select", options: [] }]);
    expect(tpl["sel"]).toBe("<choice>");
  });

  it("select with no options field falls back to <choice>", () => {
    const tpl = buildInputTemplate([{ name: "sel", type: "select" }]);
    expect(tpl["sel"]).toBe("<choice>");
  });

  it("conditional with no cases emits nothing beyond selector", () => {
    const tpl = buildInputTemplate([
      {
        name: "cond",
        type: "conditional",
        test_param: { name: "mode", type: "select", options: [] },
        cases: [],
      },
    ]);
    // test_param has no options -> sel_value is "<choice>"
    expect(tpl["cond|mode"]).toBe("<choice>");
  });

  it("returns empty object for empty inputs array", () => {
    expect(buildInputTemplate([])).toEqual({});
  });

  it("skips params with no name", () => {
    const tpl = buildInputTemplate([{ type: "data" }]);
    expect(Object.keys(tpl)).toHaveLength(0);
  });
});

describe("summarizeToolInputs", () => {
  it("produces compact summary structure", () => {
    const summary = summarizeToolInputs(TOOL_INPUTS);
    expect(summary).toHaveLength(7);

    // data param
    expect(summary[0]).toEqual({ name: "input_file", type: "data" });

    // select: choices present, first three values
    const sel = summary[1] as Record<string, unknown>;
    expect(sel["name"]).toBe("format");
    expect(sel["type"]).toBe("select");
    expect(sel["choices"]).toEqual(["fastq", "fasta", "bam"]);
    expect("choices_truncated" in sel).toBe(false);

    // boolean
    expect(summary[2]).toEqual({ name: "paired", type: "boolean" });

    // integer
    expect(summary[3]).toEqual({ name: "threads", type: "integer" });

    // repeat
    const rep = summary[4] as Record<string, unknown>;
    expect(rep["name"]).toBe("adapters");
    expect(rep["type"]).toBe("repeat");
    expect(rep["repeat_key_hint"]).toBe("adapters_0|<param>");
    expect(rep["children"]).toEqual([{ name: "seq", type: "data" }]);

    // section
    const sec = summary[5] as Record<string, unknown>;
    expect(sec["name"]).toBe("advanced");
    expect(sec["type"]).toBe("section");
    expect(sec["section_key_hint"]).toBe("advanced|<param>");
    expect(sec["children"]).toEqual([
      { name: "min_length", type: "integer" },
      { name: "db", type: "data" },
    ]);

    // conditional
    const cond = summary[6] as Record<string, unknown>;
    expect(cond["name"]).toBe("quality_filter");
    expect(cond["type"]).toBe("conditional");
    const selector = cond["selector"] as Record<string, unknown>;
    expect(selector["name"]).toBe("method");
    expect(selector["type"]).toBe("select");
    expect(selector["choices"]).toEqual(["phred", "solexa"]);
    expect(selector["key_hint"]).toBe("quality_filter|method");
    expect("choices_truncated" in selector).toBe(false);
    const cases = cond["cases"] as Array<Record<string, unknown>>;
    expect(cases).toHaveLength(2);
    expect(cases[0]).toEqual({ when: "phred", params: [{ name: "cutoff", type: "integer" }] });
    expect(cases[1]).toEqual({ when: "solexa", params: [{ name: "score", type: "float" }] });
  });

  it("caps select choices at 25 and sets choices_truncated", () => {
    // 26 options -> capped to 25, truncated flag set
    const manyOptions = Array.from({ length: 26 }, (_, i) => [`Label ${i}`, `val${i}`, false]);
    const summary = summarizeToolInputs([
      { name: "big_sel", type: "select", options: manyOptions },
    ]);
    const s = summary[0] as Record<string, unknown>;
    expect((s["choices"] as unknown[]).length).toBe(25);
    expect(s["choices_truncated"]).toBe(true);
  });

  it("does not set choices_truncated when exactly 25 options", () => {
    const options = Array.from({ length: 25 }, (_, i) => [`L${i}`, `v${i}`, false]);
    const summary = summarizeToolInputs([{ name: "sel", type: "select", options }]);
    const s = summary[0] as Record<string, unknown>;
    expect((s["choices"] as unknown[]).length).toBe(25);
    expect("choices_truncated" in s).toBe(false);
  });

  it("caps conditional selector choices and sets choices_truncated", () => {
    const manyOptions = Array.from({ length: 26 }, (_, i) => [`L${i}`, `v${i}`, false]);
    const summary = summarizeToolInputs([
      {
        name: "cond",
        type: "conditional",
        test_param: { name: "sel", type: "select", options: manyOptions },
        cases: [],
      },
    ]);
    const cond = summary[0] as Record<string, unknown>;
    const selector = cond["selector"] as Record<string, unknown>;
    expect((selector["choices"] as unknown[]).length).toBe(25);
    expect(selector["choices_truncated"]).toBe(true);
  });

  it("includes optional field when present", () => {
    const summary = summarizeToolInputs([{ name: "x", type: "data", optional: true }]);
    expect((summary[0] as Record<string, unknown>)["optional"]).toBe(true);
  });

  it("omits optional field when absent", () => {
    const summary = summarizeToolInputs([{ name: "x", type: "data" }]);
    expect("optional" in (summary[0] as Record<string, unknown>)).toBe(false);
  });

  it("returns empty array for empty inputs", () => {
    expect(summarizeToolInputs([])).toEqual([]);
  });
});
