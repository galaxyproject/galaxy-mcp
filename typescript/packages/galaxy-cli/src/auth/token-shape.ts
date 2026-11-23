/**
 * Whether a string can be a token: printable ASCII and nothing else, so no whitespace, control
 * or format characters, curly quotes or non-ASCII.
 *
 * A token goes into an HTTP header and, for `auth env`, into a shell, where it is quoted for
 * sh, fish and PowerShell. Real Galaxy API keys are plain ASCII, so
 * anything else is a broken paste or a hostile server, and refusing it before it is stored or
 * sent means none of those quoting rules has to be right for a character that cannot get
 * there. Callers must not echo the token in their message.
 */
export function isPlainToken(token: string): boolean {
  return /^[\x21-\x7e]+$/.test(token);
}
