import { describe, it, expect } from "vitest";
import { z } from "zod";
import { Command } from "commander";
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
