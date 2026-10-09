import { describe, it, expect } from "vitest";
import { z } from "zod";
import { getUserToolOp, getUserTool } from "../../src/operations/get-user-tool";
import { GalaxyAuthError, GalaxyNotFoundError } from "../../src/errors";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const UUID = "61d15277-a911-45ef-aa66-5385146578cc";
const RECORD = {
  id: "ut000001",
  uuid: UUID,
  tool_id: "row_filter",
  active: true,
  hidden: false,
  tool_format: null,
  create_time: "2026-01-02T03:04:05.000000",
  representation: { class: "GalaxyUserTool", id: "row_filter", version: "0.1.0" },
} as any;

describe("get_user_tool", () => {
  it("is named for the Python tool and reads only", () => {
    expect(getUserToolOp.name).toBe("get_user_tool");
    expect(getUserToolOp.readOnly).toBeUndefined();
    expect(getUserToolOp.requires).toBeUndefined();
  });

  it("GETs /api/unprivileged_tools/{uuid} and returns the record", async () => {
    let calls = 0;
    const client = mockClient({
      GET: (path, init) => {
        calls++;
        expect(path).toBe("/api/unprivileged_tools/{uuid}");
        expect(init.params.path.uuid).toBe(UUID);
        expect(init.params.query).toBeUndefined();
        return { data: RECORD, response: { status: 200 } };
      },
    });
    const out = await getUserTool({ uuid: UUID }, ctxWith(client));
    expect(calls).toBe(1);
    expect(out.tool_id).toBe("row_filter");
    expect(out.representation).toEqual(RECORD.representation);
  });

  it("projects the tool id and uuid into the message, and nothing else", () => {
    expect(getUserToolOp.project!(RECORD, { uuid: UUID }, {} as any)).toEqual({
      message: `Retrieved user-defined tool 'row_filter' (UUID: ${UUID})`,
    });
  });

  it("names the uuid when the record carries no tool id", () => {
    const bare = { ...RECORD, tool_id: null };
    expect(getUserToolOp.project!(bare, { uuid: UUID }, {} as any)).toEqual({
      message: `Retrieved user-defined tool '${UUID}' (UUID: ${UUID})`,
    });
  });

  it("requires the uuid", () => {
    const schema = z.object(getUserToolOp.input);
    expect(schema.safeParse({}).success).toBe(false);
    expect(schema.safeParse({ uuid: null }).success).toBe(false);
    expect(schema.parse({ uuid: UUID })).toEqual({ uuid: UUID });
  });

  it.each([
    "row_filter",
    "61d15277a91145efaa665385146578cc",
    "61d15277-a911-45ef-aa66-5385146578c",
    `{${UUID}}`,
    "",
  ])("refuses %j as not found before sending anything", async (bad) => {
    let calls = 0;
    const client = mockClient({
      GET: () => {
        calls++;
        return { data: RECORD, response: { status: 200 } };
      },
    });
    const err = await getUserTool({ uuid: bad }, ctxWith(client)).catch((e) => e);
    expect(err).toBeInstanceOf(GalaxyNotFoundError);
    expect(err.message).toBe(
      `No user-defined tool found with UUID '${bad}': that is not a UUID. A user tool's UUID ` +
        "is 32 hex digits in 8-4-4-4-12 groups, as list_user_tools() reports it. " +
        "Nothing was sent to Galaxy.",
    );
    expect(err.http).toBeUndefined();
    expect(calls).toBe(0);
  });

  it("accepts upper-case hex, which is still the same uuid", async () => {
    const upper = UUID.toUpperCase();
    const client = mockClient({
      GET: (_path, init) => {
        expect(init.params.path.uuid).toBe(upper);
        return { data: RECORD, response: { status: 200 } };
      },
    });
    await expect(getUserTool({ uuid: upper }, ctxWith(client))).resolves.toBeDefined();
  });

  it("throws GalaxyNotFoundError on 404", async () => {
    const client = mockClient({
      GET: () => ({
        error: { err_msg: "History not found", err_code: 404001 },
        response: { status: 404 },
      }),
    });
    await expect(getUserTool({ uuid: UUID }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });

  it("throws GalaxyAuthError on 403", async () => {
    const client = mockClient({
      GET: () => ({
        error: { err_msg: "not accessible", err_code: 403002 },
        response: { status: 403 },
      }),
    });
    await expect(getUserTool({ uuid: UUID }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyAuthError,
    );
  });
});
