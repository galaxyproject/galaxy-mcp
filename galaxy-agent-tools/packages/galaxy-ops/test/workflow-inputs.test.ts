import { describe, it, expect } from "vitest";
import {
  normalizeRunModel,
  normalizeGaSteps,
  buildWorkflowInputTemplate,
  buildGuide,
  findLegacyWarnings,
} from "../src/workflow-inputs";

// ---------------------------------------------------------------------------
// Golden fixtures
// ---------------------------------------------------------------------------

/**
 * Minimal style=run model fixture: two data inputs and one parameter input,
 * steps as a dict (the common case). Derived from Python normalizeRunModel
 * expectations by reading galaxy_mcp/ops/workflow_inputs.py.
 */
const RUN_MODEL_FIXTURE: Record<string, unknown> = {
  has_upgrade_messages: false,
  step_version_changes: [],
  steps: {
    "0": {
      step_type: "data_input",
      step_index: 0,
      step_label: "Input reads",
      uuid: "uuid-data-0",
      inputs: [
        {
          extensions: ["fastq", "fastqsanger"],
          acceptable_extensions: ["fastq", "fastqsanger", "fastq.gz"],
          optional: false,
        },
      ],
    },
    "1": {
      step_type: "data_collection_input",
      step_index: 1,
      step_label: "Paired reads",
      uuid: "uuid-col-1",
      inputs: [
        {
          extensions: ["fastq"],
          acceptable_extensions: [],
          collection_type: "list:paired",
          optional: false,
        },
      ],
    },
    "2": {
      step_type: "parameter_input",
      step_index: 2,
      step_label: "Reference genome",
      uuid: "uuid-param-2",
      parameter_type: "text",
      inputs: [
        {
          optional: true,
          parameter_type: "text",
          options: [
            ["hg38", "hg38", true],
            ["mm10", "mm10", false],
          ],
        },
      ],
    },
  },
};

/**
 * Corresponding .ga definition fixture for the same three-step workflow.
 * Steps use string tool_state (JSON-encoded), the common .ga export format.
 */
const GA_DEFINITION_FIXTURE: Record<string, unknown> = {
  steps: {
    "0": {
      id: 0,
      type: "data_input",
      label: "Input reads",
      uuid: "uuid-data-0",
      tool_state: JSON.stringify({ format: ["fastq", "fastqsanger"], optional: false }),
    },
    "1": {
      id: 1,
      type: "data_collection_input",
      label: "Paired reads",
      uuid: "uuid-col-1",
      tool_state: JSON.stringify({ collection_type: "list:paired", optional: false }),
    },
    "2": {
      id: 2,
      type: "parameter_input",
      label: "Reference genome",
      uuid: "uuid-param-2",
      tool_state: JSON.stringify({
        parameter_type: "text",
        restrictions: ["hg38", "mm10"],
        optional: true,
      }),
    },
  },
};

// ---------------------------------------------------------------------------
// normalizeRunModel
// ---------------------------------------------------------------------------

describe("normalizeRunModel", () => {
  it("produces exactly 3 slots in step_index order", () => {
    const slots = normalizeRunModel(RUN_MODEL_FIXTURE);
    expect(slots).toHaveLength(3);
    expect(slots.map((s) => s.step_index)).toEqual([0, 1, 2]);
  });

  it("data slot has correct shape", () => {
    const [slot] = normalizeRunModel(RUN_MODEL_FIXTURE);
    expect(slot).toEqual({
      step_index: 0,
      step_uuid: "uuid-data-0",
      label: "Input reads",
      input_type: "data",
      src: "hda",
      accepted_formats: ["fastq", "fastqsanger"],
      acceptable_extensions: ["fastq", "fastqsanger", "fastq.gz"],
      collection_type: null,
      parameter_type: null,
      optional: false,
      options: [],
    });
  });

  it("data_collection slot has collection_type and src=hdca", () => {
    const slots = normalizeRunModel(RUN_MODEL_FIXTURE);
    expect(slots[1].input_type).toBe("data_collection");
    expect(slots[1].src).toBe("hdca");
    expect(slots[1].collection_type).toBe("list:paired");
    expect(slots[1].accepted_formats).toEqual(["fastq"]);
  });

  it("parameter slot has options from triples and parameter_type", () => {
    const slots = normalizeRunModel(RUN_MODEL_FIXTURE);
    const p = slots[2];
    expect(p.input_type).toBe("parameter");
    expect(p.src).toBeNull();
    expect(p.parameter_type).toBe("text");
    expect(p.optional).toBe(true);
    expect(p.options).toEqual([
      { label: "hg38", value: "hg38" },
      { label: "mm10", value: "mm10" },
    ]);
  });

  it("uses step_label over param.label over fallback", () => {
    const model: Record<string, unknown> = {
      steps: {
        "0": {
          step_type: "data_input",
          step_index: 0,
          // no step_label -> falls back to param.label
          inputs: [{ label: "Param Label", extensions: [] }],
        },
        "1": {
          step_type: "data_input",
          order_index: 1,
          // neither -> fallback
          inputs: [{}],
        },
      },
    };
    const slots = normalizeRunModel(model);
    expect(slots[0].label).toBe("Param Label");
    expect(slots[1].label).toBe("Input dataset (step 1)");
  });

  it("skips steps with non-integer index", () => {
    const model: Record<string, unknown> = {
      steps: {
        "0": { step_type: "data_input", step_index: 0, inputs: [{ extensions: [] }] },
        abc: { step_type: "data_input", step_index: null, inputs: [{ extensions: [] }] },
      },
    };
    expect(normalizeRunModel(model)).toHaveLength(1);
  });

  it("handles steps as an array", () => {
    const model: Record<string, unknown> = {
      steps: [
        { step_type: "data_input", step_index: 0, inputs: [{ extensions: ["bam"] }] },
        { type: "tool" }, // non-input -- skipped
      ],
    };
    const slots = normalizeRunModel(model);
    expect(slots).toHaveLength(1);
    expect(slots[0].accepted_formats).toEqual(["bam"]);
  });

  it("collection_types array fallback", () => {
    const model: Record<string, unknown> = {
      steps: {
        "0": {
          step_type: "data_collection_input",
          step_index: 0,
          inputs: [{ collection_types: ["list", "paired"], extensions: [] }],
        },
      },
    };
    const [slot] = normalizeRunModel(model);
    expect(slot.collection_type).toBe("list");
  });
});

// ---------------------------------------------------------------------------
// normalizeGaSteps
// ---------------------------------------------------------------------------

describe("normalizeGaSteps", () => {
  it("produces exactly 3 slots in key order", () => {
    const slots = normalizeGaSteps(GA_DEFINITION_FIXTURE);
    expect(slots).toHaveLength(3);
    expect(slots.map((s) => s.step_index)).toEqual([0, 1, 2]);
  });

  it("data slot has correct shape (no acceptable_extensions -- .ga path)", () => {
    const [slot] = normalizeGaSteps(GA_DEFINITION_FIXTURE);
    expect(slot).toEqual({
      step_index: 0,
      step_uuid: "uuid-data-0",
      label: "Input reads",
      input_type: "data",
      src: "hda",
      accepted_formats: ["fastq", "fastqsanger"],
      acceptable_extensions: [], // always empty on ga path
      collection_type: null,
      parameter_type: null,
      optional: false,
      options: [],
    });
  });

  it("data_collection slot has collection_type", () => {
    const slots = normalizeGaSteps(GA_DEFINITION_FIXTURE);
    expect(slots[1].collection_type).toBe("list:paired");
  });

  it("parameter slot gets options from restrictions", () => {
    const slots = normalizeGaSteps(GA_DEFINITION_FIXTURE);
    const p = slots[2];
    expect(p.input_type).toBe("parameter");
    expect(p.parameter_type).toBe("text");
    expect(p.optional).toBe(true);
    expect(p.options).toEqual([
      { label: "hg38", value: "hg38" },
      { label: "mm10", value: "mm10" },
    ]);
  });

  it("uses label from step, falls back to type+index if absent", () => {
    const def: Record<string, unknown> = {
      steps: {
        "0": {
          id: 0,
          type: "parameter_input",
          // no label
          tool_state: "{}",
        },
      },
    };
    const [slot] = normalizeGaSteps(def);
    expect(slot.label).toBe("Input parameter (step 0)");
  });

  it("skips non-input step types", () => {
    const def: Record<string, unknown> = {
      steps: {
        "0": { type: "tool", tool_id: "fastqc", tool_state: "{}" },
        "1": { type: "data_input", label: "Input", tool_state: "{}" },
      },
    };
    const slots = normalizeGaSteps(def);
    expect(slots).toHaveLength(1);
    expect(slots[0].step_index).toBe(1);
  });

  it("skips non-numeric step keys", () => {
    const def: Record<string, unknown> = {
      steps: {
        "0": { type: "data_input", label: "Input", tool_state: "{}" },
        nonNumeric: { type: "data_input", label: "Bad", tool_state: "{}" },
      },
    };
    expect(normalizeGaSteps(def)).toHaveLength(1);
  });

  it("parses string tool_state (the common .ga export format)", () => {
    const def: Record<string, unknown> = {
      steps: {
        "0": {
          type: "data_input",
          label: "Bed",
          tool_state: '{"format": "bed", "optional": true}',
        },
      },
    };
    const [slot] = normalizeGaSteps(def);
    expect(slot.accepted_formats).toEqual(["bed"]); // scalar -> asList wraps it
    expect(slot.optional).toBe(true);
  });

  it("returns empty when steps is absent", () => {
    expect(normalizeGaSteps({})).toEqual([]);
    expect(normalizeGaSteps({ steps: null })).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// findLegacyWarnings
// ---------------------------------------------------------------------------

const RUNTIME_VALUE_STEP = {
  type: "tool",
  label: "Trim tool",
  tool_state: JSON.stringify({
    quality: { __class__: "RuntimeValue" },
    paired: false,
  }),
};

describe("findLegacyWarnings", () => {
  it("flags tool steps with a RuntimeValue in tool_state", () => {
    const def = { steps: { "5": RUNTIME_VALUE_STEP } };
    const warnings = findLegacyWarnings(def);
    expect(warnings).toHaveLength(1);
    expect(warnings[0].kind).toBe("legacy_runtime_value");
    expect(warnings[0].message).toContain("Trim tool");
    expect(warnings[0].message).toContain("RuntimeValue");
  });

  it("does not flag parameter_input steps (formal inputs are fine)", () => {
    const def = {
      steps: {
        "0": {
          type: "parameter_input",
          label: "My Param",
          tool_state: JSON.stringify({ __class__: "RuntimeValue" }),
        },
      },
    };
    expect(findLegacyWarnings(def)).toHaveLength(0);
  });

  it("detects nested RuntimeValue inside arrays", () => {
    const def = {
      steps: {
        "2": {
          type: "tool",
          tool_id: "bwa",
          tool_state: JSON.stringify({
            inputs: [{ __class__: "RuntimeValue" }],
          }),
        },
      },
    };
    expect(findLegacyWarnings(def)).toHaveLength(1);
    expect(findLegacyWarnings(def)[0].message).toContain("bwa");
  });

  it("uses tool_id as fallback label when label is absent", () => {
    const def = {
      steps: {
        "3": {
          type: "tool",
          tool_id: "trimmomatic",
          tool_state: JSON.stringify({ quality: { __class__: "RuntimeValue" } }),
        },
      },
    };
    expect(findLegacyWarnings(def)[0].message).toContain("trimmomatic");
  });

  it("uses step key as last-resort label", () => {
    const def = {
      steps: {
        "7": {
          type: "tool",
          tool_state: JSON.stringify({ x: { __class__: "RuntimeValue" } }),
        },
      },
    };
    expect(findLegacyWarnings(def)[0].message).toContain("step 7");
  });

  it("returns empty for a clean workflow", () => {
    const def = {
      steps: {
        "0": { type: "data_input", tool_state: "{}" },
        "1": { type: "tool", tool_state: '{"quality": 20}' },
      },
    };
    expect(findLegacyWarnings(def)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// buildWorkflowInputTemplate
// ---------------------------------------------------------------------------

describe("buildWorkflowInputTemplate", () => {
  it("composes slots + warnings + guide into the expected shape", () => {
    const slots = normalizeRunModel(RUN_MODEL_FIXTURE);
    const warnings = [{ kind: "legacy_runtime_value", message: "some warning" }];
    const guide = { summary: "A workflow.", annotation: "", provenance: {} };

    const tmpl = buildWorkflowInputTemplate(slots, warnings, guide, false);

    // inputs_template keyed by step_index
    expect(tmpl.inputs_template).toEqual({
      "0": { src: "hda", id: "<dataset_id>" },
      "1": { src: "hdca", id: "<collection_id>" },
      "2": "<value>",
    });
    expect(tmpl.inputs_by).toBe("step_index|step_uuid");
    expect(tmpl.warnings).toEqual(warnings);
    expect(tmpl.guide).toEqual(guide);
    expect(tmpl.slots).toHaveLength(3);
  });

  it("omits guide key when guide is null", () => {
    const slots = normalizeRunModel(RUN_MODEL_FIXTURE);
    const tmpl = buildWorkflowInputTemplate(slots, [], null, false);
    expect("guide" in tmpl).toBe(false);
  });

  it("defaults warnings to [] when null/undefined", () => {
    const tmpl = buildWorkflowInputTemplate([], null, null, false);
    expect(tmpl.warnings).toEqual([]);
  });

  it("drops acceptable_extensions from displayed slots", () => {
    const slots = normalizeRunModel(RUN_MODEL_FIXTURE);
    const tmpl = buildWorkflowInputTemplate(slots, [], null, false);
    for (const s of tmpl.slots) {
      expect("acceptable_extensions" in s).toBe(false);
    }
  });

  it("drops options from non-parameter slots but keeps option_count for parameter slots", () => {
    const slots = normalizeRunModel(RUN_MODEL_FIXTURE);
    const tmpl = buildWorkflowInputTemplate(slots, [], null, false);
    // data slot: no options key
    expect("options" in tmpl.slots[0]).toBe(false);
    // parameter slot with 2 options: inline (below cap), option_count present
    expect((tmpl.slots[2] as any).option_count).toBe(2);
    expect((tmpl.slots[2] as any).options).toHaveLength(2);
  });

  it("caps large option sets at OPTIONS_SAMPLE=15 when verbose=false", () => {
    // Build a parameter slot with 30 options
    const manyOptions = Array.from({ length: 30 }, (_, i) => ({
      label: `opt${i}`,
      value: `opt${i}`,
    }));
    const slot = normalizeGaSteps({
      steps: {
        "0": {
          type: "parameter_input",
          label: "Big select",
          tool_state: JSON.stringify({
            restrictions: Array.from({ length: 30 }, (_, i) => `opt${i}`),
          }),
        },
      },
    })[0];
    // Manually patch options to verify the cap logic
    slot.options = manyOptions;

    const tmpl = buildWorkflowInputTemplate([slot], [], null, false);
    const displayed = tmpl.slots[0] as any;
    expect(displayed.option_count).toBe(30);
    expect(displayed.options).toHaveLength(15);
    expect(displayed.options_note).toContain("showing 15 of 30");
  });

  it("shows full options when verbose=true even beyond cap", () => {
    const manyOptions = Array.from({ length: 30 }, (_, i) => ({
      label: `opt${i}`,
      value: `opt${i}`,
    }));
    const slots = normalizeGaSteps({
      steps: {
        "0": {
          type: "parameter_input",
          label: "Big select",
          tool_state: JSON.stringify({
            restrictions: Array.from({ length: 30 }, (_, i) => `opt${i}`),
          }),
        },
      },
    });
    slots[0].options = manyOptions;
    const tmpl = buildWorkflowInputTemplate(slots, [], null, true);
    expect((tmpl.slots[0] as any).options).toHaveLength(30);
    expect("options_note" in tmpl.slots[0]).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// buildGuide
// ---------------------------------------------------------------------------

const SHOW_WORKFLOW_FIXTURE: Record<string, unknown> = {
  version: 3,
  annotation: "An RNA-seq workflow",
  readme: "# RNA-seq\n\nThis workflow trims and aligns reads.",
  source_metadata: {
    trs_tool_id: "#workflow/github.com/foo/bar",
    trs_url: "https://dockstore.org/api/ga4gh/trs/v2/tools/%23workflow%2Fgithub.com%2Ffoo%2Fbar",
  },
};

describe("buildGuide", () => {
  it("produces a guide with summary, annotation, and provenance", () => {
    const guide = buildGuide(SHOW_WORKFLOW_FIXTURE, RUN_MODEL_FIXTURE, false);
    expect(guide.annotation).toBe("An RNA-seq workflow");
    expect(guide.provenance.version).toBe(3);
    expect(guide.provenance.source.trs_id).toBe("#workflow/github.com/foo/bar");
  });

  it("cleans the readme summary when verbose=false", () => {
    const guide = buildGuide(SHOW_WORKFLOW_FIXTURE, RUN_MODEL_FIXTURE, false);
    // Cleaned: strips the # header, collapses whitespace
    expect(guide.summary).toBe("This workflow trims and aligns reads.");
    expect(guide.summary).not.toContain("# RNA-seq");
  });

  it("returns full readme when verbose=true", () => {
    const guide = buildGuide(SHOW_WORKFLOW_FIXTURE, RUN_MODEL_FIXTURE, true);
    expect(guide.summary).toBe(SHOW_WORKFLOW_FIXTURE["readme"]);
  });

  it("falls back to annotation when readme cleans to empty", () => {
    const show = { ...SHOW_WORKFLOW_FIXTURE, readme: "# Just headers\n##Another" };
    const guide = buildGuide(show, RUN_MODEL_FIXTURE, false);
    expect(guide.summary).toBe("An RNA-seq workflow");
  });

  it("includes freshness from run_model when provided", () => {
    const guide = buildGuide(SHOW_WORKFLOW_FIXTURE, RUN_MODEL_FIXTURE, false);
    expect(guide.provenance.freshness).toBeDefined();
    expect(guide.provenance.freshness!.has_upgrade_messages).toBe(false);
    expect(guide.provenance.freshness!.step_version_changes).toEqual([]);
  });

  it("adds notes and omits freshness when run_model is null (ga-fallback path)", () => {
    const guide = buildGuide(SHOW_WORKFLOW_FIXTURE, null, false);
    expect(guide.provenance.freshness).toBeUndefined();
    expect(guide.notes).toBeDefined();
    expect(guide.notes![0]).toContain("history_id");
  });

  it("handles missing source_metadata gracefully", () => {
    const show = { version: 1, annotation: "test" };
    const guide = buildGuide(show, null, false);
    expect(guide.provenance.source.trs_id).toBeNull();
    expect(guide.provenance.source.trs_url).toBeNull();
  });

  it("falls through to empty summary when no readme/help/annotation", () => {
    const guide = buildGuide({}, null, false);
    expect(guide.summary).toBe("");
  });
});
