import { describe, it, expect, expectTypeOf } from "vitest";
import { z } from "zod";
import type { Operation, InputOf } from "../../src/operations/types";

const shape = { id: z.string(), verbose: z.boolean().optional() };

describe("Operation contract", () => {
  it("derives the run input type from the raw shape", () => {
    type In = InputOf<typeof shape>;
    expectTypeOf<In>().toEqualTypeOf<{ id: string; verbose?: boolean }>();
  });

  it("an operation carries name/domain/summary/input/run", () => {
    const op: Operation<typeof shape, { ok: true }> = {
      name: "demo",
      domain: "connection",
      summary: "demo op",
      input: shape,
      run: async () => ({ ok: true }),
    };
    expect(op.name).toBe("demo");
    expect(Object.keys(op.input)).toEqual(["id", "verbose"]);
  });

  it("Operation accepts an optional readOnly hint", () => {
    const op: Operation<Record<string, never>, number> = {
      name: "x", domain: "connection", summary: "s", input: {}, readOnly: false,
      run: async () => 1,
    };
    expect(op.readOnly).toBe(false);
  });
});
