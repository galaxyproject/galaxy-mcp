import { httpError, GalaxyNotFoundError } from "../errors";

/**
 * server.py, _job_lookup_failed: a job asked for by id that Galaxy answers 400 or 404 to is
 * not found, whichever of the two it was. Galaxy refuses an id it cannot decode with a 400
 * before looking anything up, and the caller holding the id -- an agent asking whether a run
 * it started still exists -- cannot act on the difference. The facts stay on the error so
 * the sentence is still worded from the reply. Shared by get_job_details and get_job_logs,
 * which read the same record by the same id and differ only in what they keep of it.
 */
export function jobLookupFailure(response: { status: number } | undefined, error: unknown): Error {
  const failure = httpError(response, error);
  if (failure.http?.status === 400) {
    const notFound = new GalaxyNotFoundError(failure.message);
    notFound.http = failure.http;
    return notFound;
  }
  return failure;
}

/** server.py, _job_lookup_failed: the one sentence a 400 and a 404 on a job id share. */
export function jobNotFoundSentence(jobId: string): string {
  return (
    `Job ID '${jobId}' not found or not accessible. ` +
    "Make sure the job exists and you have permission to view it."
  );
}
