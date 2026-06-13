import { describe, it, expect } from "vitest";
import { z } from "zod";
import { runWithEnvelope } from "../../src/operations/registry";
import type { Operation } from "../../src/operations/types";
import { GalaxyNotFoundError } from "../../src/errors";
import { createGalaxyContext } from "../../src/context";

const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K" });

const okOp: Operation<{ id: typeof z.string }, { value: string }> = {
  name: "ok_op",
  domain: "connection",
  summary: "returns value",
  input: { id: z.string() },
  run: async (i) => ({ value: i.id }),
  project: (o) => ({ message: `got ${o.value}` }),
};

const failOp: Operation<Record<string, never>, never> = {
  name: "fail_op",
  domain: "connection",
  summary: "throws",
  input: {},
  run: async () => {
    throw new GalaxyNotFoundError("nope");
  },
};

const bugOp: Operation<Record<string, never>, never> = {
  name: "bug_op",
  domain: "connection",
  summary: "throws a non-Galaxy error (a real bug)",
  input: {},
  run: async () => {
    throw new TypeError("undefined is not a function");
  },
};

describe("runWithEnvelope", () => {
  it("wraps success and applies project()", async () => {
    const r = await runWithEnvelope(okOp as any, { id: "abc" }, ctx);
    expect(r).toEqual({ data: { value: "abc" }, success: true, message: "got abc" });
  });
  it("catches a typed error into success=false + message", async () => {
    const r = await runWithEnvelope(failOp as any, {}, ctx);
    expect(r.success).toBe(false);
    expect(r.message).toContain("nope");
    expect(r.data).toBeUndefined();
  });
  it("rethrows a non-Galaxy error instead of swallowing it (bugs must surface)", async () => {
    await expect(runWithEnvelope(bugOp as any, {}, ctx)).rejects.toThrow(TypeError);
  });
  it("surfaces the typed error's kind on the envelope", async () => {
    const r = await runWithEnvelope(failOp as any, {}, ctx); // failOp throws GalaxyNotFoundError
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("not_found");
  });
});
