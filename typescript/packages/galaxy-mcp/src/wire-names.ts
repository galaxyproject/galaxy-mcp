import { spellParamNames } from "@galaxyproject/galaxy-ops";
import { z, type ZodObject, type ZodRawShape, type ZodType } from "zod";

/**
 * The names a tool call writes, and how they get back to the names an op reads.
 *
 * The ops are a TypeScript library, so their inputs are camelCase; the Python MCP server
 * takes the same parameters under Python names. A model calling `get_history_details`
 * should not have to know which of the two servers answered, so this surface -- and only
 * this surface -- advertises and accepts the Python spelling. `@galaxyproject/galaxy-ops`
 * and anything that imports an op directly still pass camelCase, and the CLI's flags come
 * off the same camelCase keys and are unchanged.
 *
 * The conversion is mechanical, checked against the Python surface manifest by the parity
 * test rather than asserted here: every parameter of every tool converts to exactly the
 * name the other server declares, so there is no table of exceptions to keep.
 */
export function toSnakeCase(name: string): string {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

export interface WireShape {
  /** The op's own schemas, under the names a caller writes. */
  shape: ZodRawShape;
  /**
   * The same shape as a closed object: what the tool advertises and what its arguments
   * are validated against.
   */
  object: ZodObject<ZodRawShape>;
  /** Wire name -> the key the op reads, for turning a validated call back into an input. */
  toInput: ReadonlyMap<string, string>;
}

const quoted = (name: string): string => JSON.stringify(name);

/**
 * What to say about names a tool does not take.
 *
 * Worth saying properly, because the rename is exactly the mistake a caller will make: a
 * payload written for 0.1.x, or copied out of the ops library, arrives one underscore away
 * from working. So a key whose Python spelling IS a parameter of this tool is answered with
 * that spelling rather than with a bare refusal.
 */
export function unknownKeyMessage(declared: ReadonlySet<string>, keys: readonly string[]): string {
  const say = (key: string) => {
    const wire = toSnakeCase(key);
    return wire !== key && declared.has(wire)
      ? `${quoted(key)} -- did you mean ${quoted(wire)}?`
      : quoted(key);
  };
  const what = keys.length > 1 ? "parameters" : "parameter";
  return `Unrecognized ${what}: ${keys.map(say).join(", ")}`;
}

/**
 * Anything a client reads, with the parameters it names spelled the way it has to send them.
 *
 * A description that says "pass sectionId" is an instruction this surface now refuses, so the
 * text goes through the same rename the schema does. Applied to the tool's own line and to
 * every parameter's, because a client sees both.
 */
export function inWireNames(text: string, input: ZodRawShape): string {
  return spellParamNames(text, input, toSnakeCase);
}

function describedInWireNames(schema: ZodRawShape[string], input: ZodRawShape): ZodRawShape[string] {
  const text = (schema as { description?: string }).description;
  if (text === undefined) return schema;
  const spelled = inWireNames(text, input);
  // `describe` clones, so the op's own schema keeps the sentence the op was written with.
  return spelled === text ? schema : (schema as unknown as ZodType).describe(spelled);
}

/**
 * One op's input shape as this surface advertises it.
 *
 * Only the top level is renamed. What is inside an object-valued parameter -- a workflow's
 * inputs, a tool's parameters, a user tool's representation -- is user data on its way to
 * Galaxy, and renaming a key in there would corrupt the call.
 *
 * The object is closed (`additionalProperties: false`, as every Python tool declares), so a
 * name this tool does not take is refused rather than dropped. Dropping it is the dangerous
 * answer now that the names have changed: `upload_file_from_url` with the old `historyId`
 * would have uploaded into the default history, and `get_histories` with a misspelled filter
 * would have listed everything. Closing the object is not the whole of that promise: two keys
 * never reach it, and `refuseDroppedKeys` in lax-transport.ts answers for those.
 *
 * Two parameters converging on one name would leave a tool quietly missing a parameter, so
 * it stops the server being built rather than the call being made.
 */
export function wireShape(op: { name: string; input: ZodRawShape }): WireShape {
  const renamed: [string, ZodRawShape[string]][] = [];
  const toInput = new Map<string, string>();
  for (const [key, schema] of Object.entries(op.input)) {
    const wire = toSnakeCase(key);
    const taken = toInput.get(wire);
    if (taken !== undefined) {
      throw new Error(
        `${op.name}: "${key}" and "${taken}" are both spelled "${wire}" on the wire, so one of ` +
          "them could never be called",
      );
    }
    toInput.set(wire, key);
    renamed.push([wire, describedInWireNames(schema, op.input)]);
  }
  const shape: ZodRawShape = Object.fromEntries(renamed);
  const declared = new Set(toInput.keys());
  const object = z.strictObject(shape, {
    // Only the issues this object raises reach here, and only one of them is ours: an
    // undefined or otherwise unusable argument container is zod's to describe, and
    // returning nothing leaves its own message in place.
    error: (issue: { code?: string; keys?: unknown }) =>
      issue.code === "unrecognized_keys" && Array.isArray(issue.keys)
        ? unknownKeyMessage(declared, issue.keys as string[])
        : undefined,
  });
  return { shape, object, toInput };
}

/**
 * A validated call's arguments under the keys the op reads.
 *
 * Built from the mapping rather than from what arrived, so a key the shape does not declare
 * cannot ride along -- though by here there can be none, since the object above refuses
 * them. `Object.fromEntries` rather than assignment, so a parameter that happened to be
 * called `__proto__` would be a property and not a prototype.
 */
export function toOperationInput(
  args: Record<string, unknown>,
  toInput: ReadonlyMap<string, string>,
): Record<string, unknown> {
  return Object.fromEntries(
    [...toInput]
      .filter(([wire]) => Object.hasOwn(args, wire))
      .map(([wire, key]) => [key, args[wire]]),
  );
}
