/**
 * What a failed request left behind, filed against the reply it failed on.
 *
 * The facts and the error are separated on purpose. A throw site knows a request failed and
 * with what status; only the operation knows which sentence the Python server would have
 * said about it, and only the surface knows whether that sentence is going out as an MCP
 * error or as a CLI envelope. So the middleware files the facts, the throw site attaches
 * them to a typed error, and the boundary words the sentence.
 *
 * A WeakMap keyed on the Response, rather than a field on it: the object belongs to the
 * runtime, two concurrent calls each have their own, and nothing has to be cleaned up.
 */
import type { HttpFailureFacts } from "./python-failure";

const facts = new WeakMap<object, HttpFailureFacts>();

export function rememberHttpFailure(response: object, what: HttpFailureFacts): void {
  facts.set(response, what);
}

export function httpFailureFacts(response: object | undefined): HttpFailureFacts | undefined {
  return response ? facts.get(response) : undefined;
}
