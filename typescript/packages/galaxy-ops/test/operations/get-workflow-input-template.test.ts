import { describe, it, expect } from "vitest";
import { z } from "zod";
import {
  getWorkflowInputTemplateOp,
  getWorkflowInputTemplate,
  resolveWorkflowSlots,
} from "../../src/operations/get-workflow-input-template";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError, GalaxyValidationError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Minimal style=run model with two input steps. */
const RUN_MODEL = {
  has_upgrade_messages: false,
  step_version_changes: [],
  steps: {
    "0": {
      step_type: "data_input",
      step_index: 0,
      step_label: "Reads",
      uuid: "uuid-0",
      inputs: [{ extensions: ["fastq"], acceptable_extensions: ["fastq", "fastq.gz"] }],
    },
    "1": {
      step_type: "parameter_input",
      step_index: 1,
      step_label: "Reference",
      uuid: "uuid-1",
      parameter_type: "text",
      inputs: [
        {
          optional: false,
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

/** Minimal .ga definition fixture with one data input. */
const GA_DEFINITION = {
  steps: {
    "0": {
      type: "data_input",
      label: "Input reads",
      uuid: "uuid-ga-0",
      tool_state: JSON.stringify({ format: ["fastq"], optional: false }),
    },
    "1": {
      type: "tool",
      tool_id: "trimmomatic",
      tool_state: JSON.stringify({ quality: { __class__: "RuntimeValue" } }),
    },
  },
};

/** show_workflow response fixture. */
const SHOW_WF = {
  id: "wf1",
  name: "My Workflow",
  version: 2,
  annotation: "Reads -> aligned BAM",
  readme: "# Reads aligner\n\nAligns FASTQ reads.",
  source_metadata: {},
};

// ---------------------------------------------------------------------------
// resolveWorkflowSlots
// ---------------------------------------------------------------------------

describe("resolveWorkflowSlots", () => {
  it("uses style=run path when it returns a non-empty slot list", async () => {
    let downloadCalled = 0;
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/workflows/{workflow_id}/download");
        downloadCalled++;
        const q = init?.params?.query ?? {};
        if (q.style === "run") {
          return { data: RUN_MODEL, response: { status: 200 } };
        }
        // fallback path should not be reached
        return { data: GA_DEFINITION, response: { status: 200 } };
      },
    });
    const result = await resolveWorkflowSlots(ctxWith(client), "wf1");
    expect(result.provenance).toBe("style=run");
    expect(result.slots).toHaveLength(2);
    expect(result.runModel).not.toBeNull();
    expect(downloadCalled).toBe(1); // only style=run called
  });

  it("falls back to .ga when style=run returns empty slots", async () => {
    let downloadCallCount = 0;
    const client = mockClient({
      GET: (_path, init) => {
        downloadCallCount++;
        const q = init?.params?.query ?? {};
        if (q.style === "run") {
          // Return a run model with no input steps -> empty slots -> triggers fallback
          return { data: { steps: { "0": { step_type: "tool" } } }, response: { status: 200 } };
        }
        // .ga fallback
        return { data: GA_DEFINITION, response: { status: 200 } };
      },
    });
    const result = await resolveWorkflowSlots(ctxWith(client), "wf1");
    expect(result.provenance).toBe("ga-fallback");
    expect(result.slots).toHaveLength(1); // GA_DEFINITION has one data_input
    expect(result.runModel).toBeNull();
    expect(downloadCallCount).toBe(2);
  });

  it("falls back to .ga when style=run throws (e.g. 404)", async () => {
    const client = mockClient({
      GET: (_path, init) => {
        const q = init?.params?.query ?? {};
        if (q.style === "run") {
          return { error: "not found", response: { status: 404 } };
        }
        return { data: GA_DEFINITION, response: { status: 200 } };
      },
    });
    const result = await resolveWorkflowSlots(ctxWith(client), "wf1");
    expect(result.provenance).toBe("ga-fallback");
  });

  it("passes historyId in the query when provided", async () => {
    const client = mockClient({
      GET: (_path, init) => {
        const q = init?.params?.query ?? {};
        expect(q.history_id).toBe("h99");
        return { data: RUN_MODEL, response: { status: 200 } };
      },
    });
    const result = await resolveWorkflowSlots(ctxWith(client), "wf1", "h99");
    expect(result.provenance).toBe("style=run");
  });

  it("sends instance=false in the style=run query", async () => {
    const client = mockClient({
      GET: (_path, init) => {
        const q = init?.params?.query ?? {};
        if (q.style === "run") {
          expect(q.instance).toBe(false);
        }
        return { data: RUN_MODEL, response: { status: 200 } };
      },
    });
    await resolveWorkflowSlots(ctxWith(client), "wf1");
  });
});

// ---------------------------------------------------------------------------
// getWorkflowInputTemplate (the full op)
// ---------------------------------------------------------------------------

describe("get_workflow_input_template", () => {
  /**
   * Build a mock client that handles all three download patterns:
   *   1. style=run -> returns RUN_MODEL
   *   2. no style -> .ga for legacy warnings -> returns GA_DEFINITION
   *   3. GET /api/workflows/{workflow_id} (no /download) -> SHOW_WF
   */
  function buildHappyClient(overrides?: Partial<{ runModel: unknown; gaDefinition: unknown; showWf: unknown }>) {
    return mockClient({
      GET: (path, init) => {
        if (path === "/api/workflows/{workflow_id}/download") {
          const q = init?.params?.query ?? {};
          if (q.style === "run") {
            return { data: overrides?.runModel ?? RUN_MODEL, response: { status: 200 } };
          }
          return { data: overrides?.gaDefinition ?? GA_DEFINITION, response: { status: 200 } };
        }
        if (path === "/api/workflows/{workflow_id}") {
          return { data: overrides?.showWf ?? SHOW_WF, response: { status: 200 } };
        }
        return { error: "unexpected", response: { status: 500 } };
      },
    });
  }

  it("returns a template with slots, inputs_template, guide, and warnings", async () => {
    const out = await getWorkflowInputTemplate({ workflowId: "wf1" }, ctxWith(buildHappyClient()));

    expect(out.inputs_template).toBeDefined();
    expect(out.slots).toHaveLength(2); // from RUN_MODEL
    expect(out.inputs_by).toBe("step_index|step_uuid");

    // guide should be present (best-effort show_workflow succeeded)
    expect(out.guide).toBeDefined();
    expect((out.guide as any).summary).toContain("Aligns FASTQ reads");

    // warnings: GA_DEFINITION has a tool step with RuntimeValue -> 1 warning
    expect((out.warnings as any[]).length).toBe(1);
    expect((out.warnings as any[])[0].kind).toBe("legacy_runtime_value");
  });

  it("inputs_template uses step_index keys and correct placeholder shapes", async () => {
    const out = await getWorkflowInputTemplate({ workflowId: "wf1" }, ctxWith(buildHappyClient()));
    expect(out.inputs_template["0"]).toEqual({ src: "hda", id: "<dataset_id>" });
    expect(out.inputs_template["1"]).toBe("<value>");
  });

  it("uses ga-fallback when style=run produces empty slots", async () => {
    const client = mockClient({
      GET: (path, init) => {
        if (path === "/api/workflows/{workflow_id}/download") {
          const q = init?.params?.query ?? {};
          if (q.style === "run") {
            return { data: { steps: {} }, response: { status: 200 } };
          }
          return { data: GA_DEFINITION, response: { status: 200 } };
        }
        return { data: SHOW_WF, response: { status: 200 } };
      },
    });
    const out = await getWorkflowInputTemplate({ workflowId: "wf1" }, ctxWith(client));
    expect(out.slots).toHaveLength(1); // GA_DEFINITION data_input
    // guide.notes present on ga-fallback path
    expect((out.guide as any).notes).toBeDefined();
  });

  it("still returns a template when show_workflow fails (guide docs best-effort)", async () => {
    const client = mockClient({
      GET: (path, init) => {
        if (path === "/api/workflows/{workflow_id}/download") {
          const q = init?.params?.query ?? {};
          if (q.style === "run") return { data: RUN_MODEL, response: { status: 200 } };
          return { data: GA_DEFINITION, response: { status: 200 } };
        }
        // show_workflow returns 500 -> guide should be built with empty dict
        return { error: "server error", response: { status: 500 } };
      },
    });
    const out = await getWorkflowInputTemplate({ workflowId: "wf1" }, ctxWith(client));
    expect(out.slots).toHaveLength(2);
    // guide falls through with empty show dict -> still present
    expect(out.guide).toBeDefined();
  });

  it("propagates GalaxyNotFoundError when both style=run and .ga return 404", async () => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "not found" }, response: { status: 404 } }),
    });
    await expect(
      getWorkflowInputTemplate({ workflowId: "missing" }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });

  it("project message is the other server's, source and invoke hint included", () => {
    const fakeOut = {
      inputs_template: { "0": { src: "hda", id: "<dataset_id>" } },
      slots: [{}],
      inputs_by: "step_index|step_uuid",
      warnings: [],
    };
    const msg = getWorkflowInputTemplateOp.project!(fakeOut as any, { workflowId: "wf42" });
    expect(msg.message).toBe(
      "Built an input template for workflow 'wf42' (1 slot(s), source: style=run). " +
        "Fill inputs_template and invoke with inputs_by='step_index|step_uuid'.",
    );
  });

  it("reads the source off the guide when nobody collected the fact", () => {
    // buildGuide adds `notes` exactly when there was no run model, which is exactly
    // the .ga fallback -- so a hand projection still names the right source.
    const fakeOut = {
      inputs_template: {},
      slots: [],
      inputs_by: "step_index|step_uuid",
      warnings: [],
      guide: { summary: "", notes: ["options resolve at run time"] },
    };
    const msg = getWorkflowInputTemplateOp.project!(fakeOut as any, { workflowId: "wf42" });
    expect(msg.message).toContain("(0 slot(s), source: ga-fallback)");
  });

  it("verbose=true leaves option lists uncapped and returns full readme", async () => {
    // Build a run model with a parameter having 30 options
    const bigOptions = Array.from({ length: 30 }, (_, i) => [`opt${i}`, `opt${i}`, false]);
    const bigRunModel = {
      ...RUN_MODEL,
      steps: {
        "0": {
          step_type: "parameter_input",
          step_index: 0,
          step_label: "Big select",
          uuid: "uuid-big",
          inputs: [{ options: bigOptions, optional: false, parameter_type: "text" }],
        },
      },
    };
    const client = mockClient({
      GET: (path, init) => {
        if (path === "/api/workflows/{workflow_id}/download") {
          const q = init?.params?.query ?? {};
          if (q.style === "run") return { data: bigRunModel, response: { status: 200 } };
          return { data: { steps: {} }, response: { status: 200 } };
        }
        return { data: SHOW_WF, response: { status: 200 } };
      },
    });
    const out = await getWorkflowInputTemplate({ workflowId: "wf1", verbose: true }, ctxWith(client));
    const slot = out.slots[0] as any;
    expect(slot.options).toHaveLength(30);
    expect("options_note" in slot).toBe(false);
    // summary should be full readme (not cleaned)
    expect((out.guide as any).summary).toContain("# Reads aligner");
  });
});

// ---------------------------------------------------------------------------
// version: one stored version for every read
// ---------------------------------------------------------------------------

describe("get_workflow_input_template at a stored version", () => {
  /** Every request this op makes, with the query it carried, in order. */
  function recordingClient(opts: { runModel?: unknown } = {}) {
    const seen: Array<{ path: string; query: Record<string, unknown> }> = [];
    const client = mockClient({
      GET: (path, init) => {
        const query = (init?.params?.query ?? {}) as Record<string, unknown>;
        seen.push({ path, query });
        if (path === "/api/workflows/{workflow_id}/download") {
          if (query.style === "run") {
            return { data: opts.runModel ?? RUN_MODEL, response: { status: 200 } };
          }
          return { data: GA_DEFINITION, response: { status: 200 } };
        }
        return { data: { ...SHOW_WF, version: 0 }, response: { status: 200 } };
      },
    });
    return { client, seen };
  }

  it("sends the version on the run model, the .ga export and the show", async () => {
    const { client, seen } = recordingClient();
    const out = await getWorkflowInputTemplate({ workflowId: "wf1", version: 0 }, ctxWith(client));
    expect(seen.map((r) => r.query.version)).toEqual([0, 0, 0]);
    expect(seen[0]!.query).toEqual({ style: "run", instance: false, version: 0 });
    expect((out.guide as any).provenance.version).toBe(0);
  });

  it("sends no version key at all when none was asked for", async () => {
    const { client, seen } = recordingClient();
    await getWorkflowInputTemplate({ workflowId: "wf1" }, ctxWith(client));
    for (const r of seen) expect("version" in r.query).toBe(false);
  });

  it("treats an explicit null as left out", async () => {
    const { client, seen } = recordingClient();
    await getWorkflowInputTemplate({ workflowId: "wf1", version: null }, ctxWith(client));
    for (const r of seen) expect("version" in r.query).toBe(false);
  });

  it("pins the .ga fallback to the same version", async () => {
    const { client, seen } = recordingClient({ runModel: { steps: {} } });
    const out = await getWorkflowInputTemplate({ workflowId: "wf1", version: 2 }, ctxWith(client));
    expect(out.slots).toHaveLength(1);
    // run model, fallback export, warnings export, show -- all four at version 2
    expect(seen).toHaveLength(4);
    for (const r of seen) expect(r.query.version).toBe(2);
  });

  it("refuses a negative version before sending anything", async () => {
    const { client, seen } = recordingClient();
    await expect(
      getWorkflowInputTemplate({ workflowId: "wf1", version: -1 }, ctxWith(client)),
    ).rejects.toThrow(
      new GalaxyValidationError(
        "version must be 0 or greater (got -1); 0 is the oldest stored version and get_workflow_details counts up from there",
      ),
    );
    expect(seen).toHaveLength(0);
  });

  it("names the version in the failure context only when one was asked for", () => {
    const context = getWorkflowInputTemplateOp.failure!.context!;
    expect(context({ workflowId: "wf1", version: 7 } as any)).toEqual({ workflow_id: "wf1", version: 7 });
    expect(context({ workflowId: "wf1" } as any)).toEqual({ workflow_id: "wf1" });
    expect(context({ workflowId: "wf1", version: null } as any)).toEqual({ workflow_id: "wf1" });
  });

  it("schema: version is an optional integer that accepts null and refuses fractions", () => {
    const schema = z.object(getWorkflowInputTemplateOp.input);
    expect(schema.parse({ workflowId: "wf1" }).version).toBeUndefined();
    expect(schema.parse({ workflowId: "wf1", version: null }).version).toBeNull();
    expect(schema.parse({ workflowId: "wf1", version: 3 }).version).toBe(3);
    expect(() => schema.parse({ workflowId: "wf1", version: 1.5 })).toThrow();
    expect(() => schema.parse({ workflowId: "wf1", version: "0" })).toThrow();
  });
});
