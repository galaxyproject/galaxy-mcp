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

  // Galaxy 26.0+ answers these with a 400 "Invalid UUID format" (older ones a 500), neither of
  // which is a 404 -- so anything Python's uuid.UUID() would refuse is refused here as not found.
  it.each([
    "row_filter",
    "61d15277-a911-45ef-aa66-5385146578c",
    "61d15277-a911-45ef-aa66-5385146578cg",
    "urn:uuid:row_filter",
    "{61d15277-a911-45ef-aa66-5385146578c}",
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
        "is 32 hex digits, as list_user_tools() reports it (hyphenated, bare, braced " +
        "or urn:uuid: spellings all name the same tool). Nothing was sent to Galaxy.",
    );
    expect(err.http).toBeUndefined();
    expect(calls).toBe(0);
  });

  // Galaxy validates with uuid.UUID() and compares on the 32 hex digits, so every spelling it
  // accepts finds the tool -- and delete_user_tool / run_user_tool already send them through,
  // so this has to read back what they touch. Sent as given, like those two.
  it.each([
    "61d15277a91145efaa665385146578cc",
    `{${UUID}}`,
    `urn:uuid:${UUID}`,
    UUID.toUpperCase(),
  ])("sends %j through unchanged, as uuid.UUID() accepts it", async (spelling) => {
    let calls = 0;
    const client = mockClient({
      GET: (_path, init) => {
        calls++;
        expect(init.params.path.uuid).toBe(spelling);
        return { data: RECORD, response: { status: 200 } };
      },
    });
    const out = await getUserTool({ uuid: spelling }, ctxWith(client));
    expect(calls).toBe(1);
    expect(out.tool_id).toBe("row_filter");
    expect(getUserToolOp.project!(out, { uuid: spelling }, {} as any)).toEqual({
      message: `Retrieved user-defined tool 'row_filter' (UUID: ${spelling})`,
    });
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
