import type { Command } from "commander";
import type { AnyOperation } from "@galaxyproject/galaxy-ops";
import { type ZodTypeAny } from "zod";
import { classifyField, flagName, inCliNames } from "./flags";

export function applyInputs(cmd: Command, op: AnyOperation): void {
  for (const [key, schema] of Object.entries(op.input) as [string, ZodTypeAny][]) {
    const kind = classifyField(schema);
    const flag = flagName(key);
    const help = describe(schema, op) ?? key;
    if (kind === "positional") cmd.argument(`<${key}>`, help);
    else if (kind === "boolean") {
      cmd.option(`--${flag}`, help);
      cmd.option(`--no-${flag}`, `Set ${flag} to false.`);
    }
    else if (kind === "json") cmd.option(`--${flag} <json>`, `${help} (JSON or @file.json)`);
    // Variadic: `--packages a b` and `--packages a --packages b` both build the same list.
    else if (kind === "array") cmd.option(`--${flag} <value...>`, help);
    else cmd.option(`--${flag} <value>`, help);
  }
}

/** A field's help, with any parameter it names spelled the way this CLI takes it. */
function describe(schema: ZodTypeAny, op: AnyOperation): string | undefined {
  const text = (schema as unknown as { description?: string }).description;
  return text === undefined ? undefined : inCliNames(text, op.input);
}
