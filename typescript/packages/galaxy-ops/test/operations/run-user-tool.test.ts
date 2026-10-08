import { describe, it, expect } from "vitest";
import { runUserToolOp, runUserTool } from "../../src/operations/run-user-tool";
import { GalaxyNotFoundError } from "../../src/errors";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const TOOL_UUID = "61d15277-a911-45ef-aa66-5385146578cc";
const HISTORY_ID = "abc123";
const INPUTS = { input1: { src: "hda", id: "ds1" } };

describe("run_user_tool", () => {
  it("is a write op", () => {
    expect(runUserToolOp.readOnly).toBe(false);
  });

  it("fetches tool info then POSTs to /api/tools with correct body", async () => {
    let getCount = 0;
    let postBody: Record<string, unknown> | undefined;

    const client = mockClient({
      GET: (path, init) => {
        getCount++;
        expect(path).toBe("/api/unprivileged_tools/{tool_uuid}");
        expect(init.params.path.tool_uuid).toBe(TOOL_UUID);
        return {
          data: { tool_id: "my_filter/0.1.0", representation: { version: "1.2.3" } },
          response: { status: 200 },
        };
      },
      POST: (path, init) => {
        expect(path).toBe("/api/tools");
        postBody = init.body as Record<string, unknown>;
        return { data: { outputs: [], jobs: [{ id: "j1" }] }, response: { status: 200 } };
      },
    });

    const out = await runUserTool({ historyId: HISTORY_ID, toolUuid: TOOL_UUID, inputs: INPUTS }, ctxWith(client));
    expect(getCount).toBe(1);
    expect(postBody).toMatchObject({
      tool_uuid: TOOL_UUID,
      tool_version: "1.2.3",
      history_id: HISTORY_ID,
      inputs: INPUTS,
      input_format: "legacy",
    });
    expect(out.jobs).toHaveLength(1);
  });

  it("defaults tool_version to '0.1.0' when representation has no version", async () => {
    let capturedVersion: unknown;
    const client = mockClient({
      GET: () => ({
        data: { tool_id: "my_filter/0.1.0", representation: {} },
        response: { status: 200 },
      }),
      POST: (_path, init) => {
        capturedVersion = (init.body as any).tool_version;
        return { data: { outputs: [], jobs: [] }, response: { status: 200 } };
      },
    });
    await runUserTool({ historyId: HISTORY_ID, toolUuid: TOOL_UUID, inputs: {} }, ctxWith(client));
    expect(capturedVersion).toBe("0.1.0");
  });

  it("defaults tool_version to '0.1.0' when representation is absent", async () => {
    let capturedVersion: unknown;
    const client = mockClient({
      GET: () => ({
        data: { tool_id: "my_filter/0.1.0" },
        response: { status: 200 },
      }),
      POST: (_path, init) => {
        capturedVersion = (init.body as any).tool_version;
        return { data: { outputs: [], jobs: [] }, response: { status: 200 } };
      },
    });
    await runUserTool({ historyId: HISTORY_ID, toolUuid: TOOL_UUID, inputs: {} }, ctxWith(client));
    expect(capturedVersion).toBe("0.1.0");
  });

  it("names the tool_id the lookup found, beside the uuid and the history", async () => {
    // The tool_id is the one thing in the other server's sentence that the arguments
    // do not carry, so it rides the per-call facts collector runWithEnvelope sets up.
    const client = mockClient({
      GET: () => ({
        data: {
          tool_id: "row_filter",
          representation: { version: "1.0", inputs: [{ name: "in", type: "data" }] },
        },
        response: { status: 200 },
      }),
      POST: () => ({ data: { outputs: [], jobs: [] }, response: { status: 200 } }),
    });
    const out = await runWithEnvelope(
      runUserToolOp as never,
      { historyId: HISTORY_ID, toolUuid: TOOL_UUID, inputs: {} },
      ctxWith(client),
    );
    expect(out.message).toBe(
      `Started user tool 'row_filter' (UUID: ${TOOL_UUID}) in history '${HISTORY_ID}'`,
    );
  });

  // A representation with no parameter list is not checkable, and the other server says so
  // in the success message rather than implying the inputs were vetted.
  it("says the inputs went unchecked when the representation carries no parameter list", async () => {
    const client = mockClient({
      GET: () => ({
        data: { tool_id: "row_filter", representation: { version: "1.0" } },
        response: { status: 200 },
      }),
      POST: () => ({ data: { outputs: [], jobs: [] }, response: { status: 200 } }),
    });
    const out = await runWithEnvelope(
      runUserToolOp as never,
      { historyId: HISTORY_ID, toolUuid: TOOL_UUID, inputs: {} },
      ctxWith(client),
    );
    expect(out.message).toBe(
      `Started user tool 'row_filter' (UUID: ${TOOL_UUID}) in history '${HISTORY_ID}' ` +
        "(inputs not pre-checked: the definition of 'row_filter' arrived without a parameter list)",
    );
  });

  it("throws GalaxyNotFoundError when lookup returns 200 without tool_id (no POST fired)", async () => {
    let postCalled = false;
    const client = mockClient({
      GET: () => ({
        data: { representation: { version: "1.0" } },
        response: { status: 200 },
      }),
      POST: () => {
        postCalled = true;
        return { data: {}, response: { status: 200 } };
      },
    });
    await expect(
      runUserTool({ historyId: HISTORY_ID, toolUuid: TOOL_UUID, inputs: {} }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyNotFoundError);
    expect(postCalled).toBe(false);
  });

  /**
   * A refused user-tool run gets the input explanation, whatever the reply happens to say.
   *
   * The other server's run_user_tool has no credentials branch at all: a 400 goes straight
   * to the enriched input sentence, and the word "credentials" in Galaxy's reply changes
   * nothing. The branch is run_tool's, and it was firing for both -- so a user-tool run
   * refused with "Invalid credentials" was told to configure stored credentials for a tool
   * that has no credentials path.
   */
  it("explains the inputs when Galaxy's refusal mentions credentials, as the other server does", async () => {
    const client = mockClient({
      GET: () => ({
        data: {
          tool_id: "row_filter",
          representation: { version: "1.0", inputs: [{ name: "in", type: "data" }] },
        },
        response: { status: 200 },
      }),
      POST: () => ({ error: { err_msg: "Invalid credentials" }, response: { status: 400 } }),
    });
    const out = await runWithEnvelope(
      runUserToolOp as never,
      { historyId: HISTORY_ID, toolUuid: TOOL_UUID, inputs: INPUTS },
      ctxWith(client),
    );
    expect(out.success).toBe(false);
    expect(out.message).toContain("Run user tool failed:");
    expect(out.message).toContain("do not match the parameter schema for tool 'row_filter'");
    expect(out.message).not.toContain("Run tool failed:");
    expect(out.message).not.toContain("credentials were found for tool");
    expect(out.message).not.toContain("stored credentials");
  });

  it("throws when GET fails", async () => {
    const client = mockClient({
      GET: () => ({ error: "not found", response: { status: 404 } }),
    });
    await expect(
      runUserTool({ historyId: HISTORY_ID, toolUuid: "bad-uuid", inputs: {} }, ctxWith(client)),
    ).rejects.toThrow();
  });
});
