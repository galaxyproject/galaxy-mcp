import type { AnyOperation, Operation, GalaxyResult } from "../../src/operations/types";
import type { ZodRawShape } from "zod";

/**
 * What the model actually sees for one tool call, before the surface trims it.
 *
 * `galaxy-mcp`'s `toolResult` sends `content[0].text = JSON.stringify(result)`,
 * so the text is the whole envelope -- `data` plus `success`, `message` and
 * `pagination` -- and measuring `run()`'s bare return value understates it.
 * galaxy-ops cannot import that function, so the shape is rebuilt here and
 * `galaxy-mcp/test/surface.test.ts` pins the real one against this description:
 * one text block, nothing else.
 *
 * It is NOT the size of the JSON-RPC frame. The client decodes the frame before
 * any output guard runs, so the transport's quote escaping never reaches the
 * model and must not be counted. The text is also compact rather than indented,
 * which makes it a single line: the line half of the usual 50 KB / 2000 line
 * guard cannot bind here, so bytes are the only limit.
 *
 * The budget itself is `OUTPUT_BUDGET_BYTES`, which lives in the source rather
 * than here because the surface enforces it; this is only how a test measures a
 * page the surface has not trimmed yet.
 */
export function mcpPayloadBytes<Shape extends ZodRawShape, O>(
  op: Operation<Shape, O>,
  data: O,
  input: unknown,
): number {
  const envelope: GalaxyResult<O> = {
    data,
    success: true,
    ...(op.project?.(data, input as never) ?? {}),
  };
  return new TextEncoder().encode(JSON.stringify(envelope)).length;
}

/** Same, for an op referenced through the heterogeneous registry type. */
export const mcpBytesOf = (op: AnyOperation, data: unknown, input: unknown): number =>
  mcpPayloadBytes(op as Operation<ZodRawShape, unknown>, data, input);
