/**
 * server.py, _is_encoded_id / _is_this_record: what the two reads by id check before
 * sending and after receiving, so that a 200 is answered for the record asked for.
 *
 * An id travels as one path segment, and the fetch API (like requests on the other
 * surface) folds a value such as `.` or `../histories` into a different path --
 * `/api/jobs/` or `/api/histories` -- which answers 200 with a list that is then not the
 * record asked for. Galaxy's encoded ids are hex, and it refuses anything else with a 400
 * before looking anything up, which the reads by id already report as not found; so the
 * shape check changes no answer, only whether the request is made. The sentences are the
 * Python server's, byte for byte, so change both.
 */
const ENCODED_ID = /^[0-9a-fA-F]+$/;

export function isEncodedId(value: string): boolean {
  return ENCODED_ID.test(value);
}

export function notAGalaxyId(noun: string, value: string, listing: string): string {
  return (
    `${noun} ID '${value}' not found: that is not a Galaxy id. Galaxy's ids are hex ` +
    `strings, as ${listing} reports them. Nothing was sent to Galaxy.`
  );
}

/**
 * Galaxy writes its ids in lowercase hex and decodes either case, so a caller's uppercase
 * spelling of the same id is the same record.
 */
export function isThisRecord(payload: unknown, recordId: string): boolean {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
  const answered = (payload as { id?: unknown }).id;
  return typeof answered === "string" && answered.toLowerCase() === recordId.toLowerCase();
}

export function notThatRecord(noun: string, value: string): string {
  return (
    `Galaxy answered the read of ${noun} '${value}' with something that is not that ` +
    `${noun}'s record (no matching id), so it was not returned.`
  );
}
