import { describe, it, expect } from "vitest";
import { z } from "zod";
import { getToolDetailsOp, getToolDetails } from "../../src/operations/get-tool-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_tool_details", () => {
  it("fetches a tool via the legacy endpoint", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/tools/{tool_id}");
        expect(init.params.path.tool_id).toBe("cat1");
        // No version asked for, so none is sent: the request is the one it always was.
        expect(init.params.query).toEqual({ io_details: false });
        return { data: { id: "cat1", name: "Concatenate", version: "1.0" }, response: { status: 200 } };
      },
    });
    const out = await getToolDetails({ toolId: "cat1" }, ctxWith(client));
    expect(out.id).toBe("cat1");
    expect(out.name).toBe("Concatenate");
  });
  it("throws NotFound on 404", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(getToolDetails({ toolId: "nope" }, ctxWith(client))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });

  describe("at a pinned version", () => {
    it("sends tool_version in the query and names it in the message", async () => {
      const client = mockClient({
        GET: (_path, init) => {
          expect(init.params.query).toEqual({ io_details: true, tool_version: "1.0.2" });
          return { data: { id: "Cut1", name: "Cut", version: "1.0.2" }, response: { status: 200 } };
        },
      });
      const out = await getToolDetails({ toolId: "Cut1", ioDetails: true, toolVersion: "1.0.2" }, ctxWith(client));
      expect(out.version).toBe("1.0.2");
      expect(getToolDetailsOp.project?.(out, { toolId: "Cut1", ioDetails: true, toolVersion: "1.0.2" }, {})).toEqual({
        message: "Retrieved details for tool 'Cut1' at version 1.0.2",
      });
    });
    it("treats an explicit null as no version", async () => {
      const client = mockClient({
        GET: (_path, init) => {
          expect(init.params.query).toEqual({ io_details: false });
          return { data: { id: "cat1", name: "Concatenate", version: "1.0" }, response: { status: 200 } };
        },
      });
      await getToolDetails({ toolId: "cat1", toolVersion: null }, ctxWith(client));
      expect(getToolDetailsOp.project?.({ id: "cat1", name: "Concatenate" }, { toolId: "cat1", ioDetails: false, toolVersion: null }, {})).toEqual({
        message: "Retrieved details for tool 'cat1'",
      });
    });
    it("refuses a record for another version, as not found, after the request", async () => {
      const client = mockClient({
        GET: () => ({ data: { id: "cat1", name: "Concatenate datasets", version: "1.0.0" }, response: { status: 200 } }),
      });
      const err = await getToolDetails({ toolId: "cat1", toolVersion: "9.9.9" }, ctxWith(client)).catch((e) => e);
      expect(err).toBeInstanceOf(GalaxyNotFoundError);
      expect(err.message).toBe(
        "Galaxy described version 1.0.0 of tool 'cat1' rather than the 9.9.9 asked for, so that version is not installed on this server. " +
          "Call get_tool_details('cat1') without tool_version to see the version Galaxy serves for this id.",
      );
      // A refusal of ours, not a reply's failure: no facts for the sentence builder to reword.
      expect(err.http).toBeUndefined();
    });
    it("passes a record that names no version through", async () => {
      const client = mockClient({
        GET: () => ({ data: { id: "cat1", name: "Concatenate datasets" }, response: { status: 200 } }),
      });
      const out = await getToolDetails({ toolId: "cat1", toolVersion: "1.0.0" }, ctxWith(client));
      expect(out.id).toBe("cat1");
    });
    it("names the version in the failure context only when one was asked for", () => {
      const context = getToolDetailsOp.failure?.context;
      expect(context?.({ toolId: "cat1", ioDetails: false })).toEqual({ tool_id: "cat1", io_details: false });
      expect(context?.({ toolId: "cat1", ioDetails: false, toolVersion: "1.0.0" })).toEqual({
        tool_id: "cat1",
        io_details: false,
        tool_version: "1.0.0",
      });
    });
  });

  describe("input schema", () => {
    const schema = z.object(getToolDetailsOp.input);
    it("defaults ioDetails and leaves toolVersion unset", () => {
      expect(schema.parse({ toolId: "cat1" })).toEqual({ toolId: "cat1", ioDetails: false });
    });
    it("accepts null for toolVersion and refuses it for ioDetails", () => {
      expect(schema.parse({ toolId: "cat1", toolVersion: null }).toolVersion).toBeNull();
      expect(schema.safeParse({ toolId: "cat1", ioDetails: null }).success).toBe(false);
    });
    it("refuses a non-string toolVersion", () => {
      expect(schema.safeParse({ toolId: "cat1", toolVersion: 1 }).success).toBe(false);
    });
  });
});
