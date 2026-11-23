import { version } from "../../package.json";

/** What the auth code calls itself on the wire, so a Galaxy admin can tell `gxy` from other clients. */
export const USER_AGENT = `gxy/${version} (galaxy-cli)`;
