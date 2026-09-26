import { describe, it, expect } from "vitest";
import { z } from "zod";
import { Command } from "commander";
import { getInvocationsOp, searchToolsByNameOp } from "@galaxyproject/galaxy-ops";
import { classifyField, buildInput } from "../src/flags";
import { applyInputs } from "../src/flags-apply";

describe("flags mapping", () => {
  it("classifies field kinds from a zod raw shape", () => {
    expect(classifyField(z.string())).toBe("positional");
    expect(classifyField(z.string().optional())).toBe("option");
    expect(classifyField(z.coerce.number().optional())).toBe("option");
    expect(classifyField(z.boolean().optional())).toBe("boolean");
    expect(classifyField(z.record(z.string(), z.unknown()))).toBe("json");
  });

  it("buildInput reassembles positionals + options + parses --inputs json", () => {
    const shape = {
      historyId: z.string(),
      inputs: z.record(z.string(), z.unknown()),
      limit: z.coerce.number().optional(),
    };
    const parsed = buildInput(shape, ["h1"], { inputs: '{"a":1}', limit: "5" });
    expect(parsed.success).toBe(true);
    expect(parsed.success && parsed.data).toEqual({ historyId: "h1", inputs: { a: 1 }, limit: 5 });
  });

  it("turns a numeric flag into a number for the ops that ask for one", () => {
    // Commander only ever hands over strings, and the listing schemas want plain integers, so
    // without this `galaxy-cli search_tools_by_name bwa --limit 5` is a usage error.
    const parsed = buildInput(searchToolsByNameOp.input, ["bwa"], { limit: "5", offset: "10" });
    expect(parsed.success && parsed.data).toMatchObject({ query: "bwa", limit: 5, offset: 10 });

    const invocations = buildInput(getInvocationsOp.input, [], { limit: "5" });
    expect(invocations.success && invocations.data).toMatchObject({ limit: 5 });
  });

  it("leaves a non-number alone so the schema can say what is wrong with it", () => {
    // Converting these would hand the op a 0 or a NaN and lose the reason.
    for (const bad of ["", "abc", "  "]) {
      const parsed = buildInput(getInvocationsOp.input, [], { limit: bad });
      expect(parsed.success).toBe(false);
    }
    // A non-integer converts fine and is then refused by the schema, which is the right layer.
    expect(buildInput(getInvocationsOp.input, [], { limit: "2.5" }).success).toBe(false);
  });

  it("reports a usage error for bad input", () => {
    const shape = { historyId: z.string() };
    const parsed = buildInput(shape, [], {});
    expect(parsed.success).toBe(false);
  });

  it("offers both values for boolean inputs", () => {
    const command = new Command();
    applyInputs(command, { input: { visible: z.boolean().optional() } } as any);
    command.parse(["node", "test", "--no-visible"]);
    expect(command.opts()).toMatchObject({ visible: false });
  });
});
