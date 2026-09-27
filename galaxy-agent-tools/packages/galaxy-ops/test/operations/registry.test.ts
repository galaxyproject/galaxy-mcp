import { describe, it, expect } from "vitest";
import { z } from "zod";
import { runWithEnvelope, spellParamNames } from "../../src/operations/registry";
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
    // count and pagination are always sent, null where the op reports neither:
    // that is what the Python model serialises to, and a missing key is a
    // different answer from "this tool has no count".
    expect(r).toEqual({
      data: { value: "abc" },
      success: true,
      message: "got abc",
      count: null,
      pagination: null,
    });
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

/**
 * The op's key is the one spelling written down; every surface respells it on the way out.
 * What matters here is where the rewriting STOPS -- these sentences are full of words that
 * are also parameter names somewhere, and a substitution that took them would be a bug.
 */
describe("spellParamNames", () => {
  const input = {
    sectionId: z.string(),
    historyId: z.string(),
    name: z.string(),
    inputs: z.record(z.string(), z.unknown()),
  };
  const shout = (text: string) => spellParamNames(text, input, (key) => `<${key}>`);

  it("respells the op's own camelCase parameters", () => {
    expect(shout("Pass sectionId to list one section's tools.")).toBe(
      "Pass <sectionId> to list one section's tools.",
    );
    expect(shout("ignored if historyId is provided")).toBe("ignored if <historyId> is provided");
  });

  it("leaves the single-word parameters alone, because they are also English", () => {
    expect(shout("List the histories (id, name, counts) and their inputs.")).toBe(
      "List the histories (id, name, counts) and their inputs.",
    );
  });

  it("leaves a camelCase word that is not a parameter of this op alone", () => {
    expect(shout("data refs as {src:'hda',id}, someNestedKey and all")).toBe(
      "data refs as {src:'hda',id}, someNestedKey and all",
    );
  });

  it("matches whole words only", () => {
    expect(shout("historyIds and prehistoryId are not historyId")).toBe(
      "historyIds and prehistoryId are not <historyId>",
    );
  });

  it("hands back text it has nothing to do with, unchanged", () => {
    expect(spellParamNames("nothing here", { limit: z.number() }, () => "!")).toBe("nothing here");
  });
});
