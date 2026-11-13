/**
 * Whether a string can be a token: printable ASCII and nothing else, so no whitespace, control
 * or format characters, curly quotes or non-ASCII.
 *
 * A token goes into an HTTP header, and `auth token` hands it to a shell. Real Galaxy API keys are
 * plain ASCII, so anything else is a broken paste or a hostile server, and refusing it before it
 * is stored or sent means nothing downstream has to cope with a character that cannot get there.
 * Callers must not echo the token in their message.
 */
export function isPlainToken(token: string): boolean {
  return /^[\x21-\x7e]+$/.test(token);
}
