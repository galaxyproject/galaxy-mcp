import { describe, it, expect } from "vitest";
import {
  getToolInputTemplateOp,
  getToolInputTemplate,
} from "../../src/operations/get-tool-input-template";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/** Small tool-info fixture: data param + a boolean + a select. */
const TOOL_INFO_FIXTURE = {
  id: "cat1",
  name: "Concatenate",
  version: "1.0",
  inputs: [
    { name: "input1", type: "data" },
    { name: "header", type: "boolean" },
    {
      name: "out_format",
      type: "select",
      options: [
        ["FASTQ", "fastq", true],
        ["FASTA", "fasta", false],
      ],
    },
  ],
};

describe("get_tool_input_template", () => {
  it("fetches with io_details=true and returns inputs_template + parameters", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/tools/{tool_id}");
        expect(init.params.path.tool_id).toBe("cat1");
        expect(init.params.query.io_details).toBe(true);
        expect(init.params.query.link_details).toBe(false);
        return { data: TOOL_INFO_FIXTURE, response: { status: 200 } };
      },
    });
    const out = await getToolInputTemplate({ toolId: "cat1" }, ctxWith(client));

    expect(out.tool_id).toBe("cat1");

    // inputs_template: flat key -> placeholder
    expect(out.inputs_template["input1"]).toEqual({ src: "hda", id: "<dataset_id>" });
    expect(out.inputs_template["header"]).toBe(false);
    expect(out.inputs_template["out_format"]).toBe("fastq");

    // parameters: compact summary, 3 top-level
    expect(out.parameters).toHaveLength(3);
    const params = out.parameters as Array<Record<string, unknown>>;
    expect(params[0]).toEqual({ name: "input1", type: "data" });
    expect(params[1]).toEqual({ name: "header", type: "boolean" });
    expect(params[2]).toMatchObject({ name: "out_format", type: "select", choices: ["fastq", "fasta"] });
  });

  it("handles a tool with no inputs", async () => {
    const client = mockClient({
      GET: () => ({ data: { id: "empty", name: "Empty" }, response: { status: 200 } }),
    });
    const out = await getToolInputTemplate({ toolId: "empty" }, ctxWith(client));
    expect(out.inputs_template).toEqual({});
    expect(out.parameters).toEqual([]);
  });

  it("throws GalaxyNotFoundError on 404", async () => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "not found" }, response: { status: 404 } }),
    });
    await expect(
      getToolInputTemplate({ toolId: "nope" }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });

  it("project message is the other server's, placeholder hint and all", () => {
    const out = {
      tool_id: "cat1",
      inputs_template: {},
      parameters: [{}, {}, {}],
    };
    const msg = getToolInputTemplateOp.project!(out as any, { toolId: "cat1" });
    expect(msg.message).toBe(
      "Built an input template for tool 'cat1'. Replace placeholders " +
        "(e.g. <dataset_id>) and pass the result as `inputs` to run_tool.",
    );
  });
});
