import type { ZodRawShape } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyError } from "../errors";
import type { AnyOperation, GalaxyResult, InputOf, Operation } from "./types";

/** Wrap an op for a surface: catch typed errors, apply project() metadata. */
export async function runWithEnvelope<Shape extends ZodRawShape, O>(
  op: Operation<Shape, O>,
  input: InputOf<Shape>,
  ctx: GalaxyContext,
): Promise<GalaxyResult<O>> {
  try {
    const data = await op.run(input, ctx);
    const meta = op.project?.(data, input) ?? {};
    return { data, success: true, ...meta };
  } catch (err) {
    if (err instanceof GalaxyError) {
      return { data: undefined as unknown as O, success: false, message: err.message, errorKind: err.kind };
    }
    throw err; // non-Galaxy errors are bugs -- let them surface
  }
}

/** The v1 registry. Populated as ops land (Tasks 8, 12, 15). */
export const allOperations: AnyOperation[] = [];

export function register(op: AnyOperation): AnyOperation {
  allOperations.push(op);
  return op;
}
