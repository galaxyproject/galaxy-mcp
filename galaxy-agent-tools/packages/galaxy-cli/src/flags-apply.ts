import type { Command } from "commander";
import type { AnyOperation } from "@galaxyproject/galaxy-ops";
import { type ZodTypeAny } from "zod";
import { classifyField, flagName } from "./flags";

export function applyInputs(cmd: Command, op: AnyOperation): void {
  for (const [key, schema] of Object.entries(op.input) as [string, ZodTypeAny][]) {
    const kind = classifyField(schema);
    const flag = flagName(key);
    if (kind === "positional") cmd.argument(`<${key}>`, describe(schema) ?? key);
    else if (kind === "boolean") cmd.option(`--${flag}`, describe(schema) ?? key);
    else if (kind === "json") cmd.option(`--${flag} <json>`, `${describe(schema) ?? key} (JSON or @file.json)`);
    else cmd.option(`--${flag} <value>`, describe(schema) ?? key);
  }
}

function describe(schema: ZodTypeAny): string | undefined {
  return (schema as unknown as { description?: string }).description;
}
