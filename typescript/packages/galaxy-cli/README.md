# @galaxyproject/galaxy-cli

Galaxy agent operations on the command line. One subcommand per operation --
histories, datasets, tools, workflows, pages, and the IWC catalog -- with table / JSON
output and meaningful exit codes. Built on
[`@galaxyproject/galaxy-ops`](https://www.npmjs.com/package/@galaxyproject/galaxy-ops).

## Install

```bash
npm install -g @galaxyproject/galaxy-cli   # provides `galaxy-cli` and `gxy` (the same program, shorter name)
# or run without installing:
npx @galaxyproject/galaxy-cli --help
```

Requires Node.js `>=22.19`.

## Usage

Set your [Galaxy](https://galaxyproject.org/) connection, then run a command:

```bash
export GALAXY_URL=https://usegalaxy.org/
export GALAXY_API_KEY=your-api-key

galaxy-cli get_user
galaxy-cli --format json get_histories --name rnaseq
galaxy-cli get_workflow_input_template <workflowId>
galaxy-cli invoke_workflow <workflowId> --inputs @inputs.json --history-name "WF run"
```

Credentials come from the `--url`/`--api-key` flags, the `GALAXY_URL`/`GALAXY_API_KEY`
environment variables, a stored login (`gxy auth login`), a `.env` file in the current
directory, or a planemo profile (`--profile <name>`); see "Which key and which Galaxy" below. Output is a table by
default; `--format json` prints the full result envelope for scripting. Exit
codes follow `sysexits.h` conventions (e.g. `77` for auth failures).

Run `galaxy-cli --help`, or `galaxy-cli <command> --help`, for the full list of
commands and their arguments.

## Logging in

`gxy auth` stores a login so you don't have to put an API key in a shell variable or a
command line. `gxy` and `galaxy-cli` are the same program.

```bash
printf %s "$KEY" | gxy auth login --host https://usegalaxy.org --with-token   # store an API key
gxy auth status                                                                # stored logins, checked against Galaxy
gxy get_user                                                                   # operations use the stored login, no flags
gxy auth token                                                                 # print the stored token
gxy auth logout                                                                # delete the stored login
```

| Command | Does |
| --- | --- |
| `auth login` | store an API key read from stdin (`--with-token`, required); `--insecure-storage` keeps the token in `hosts.yml` instead of the system keyring |
| `auth status` | show stored logins and check each one (`--json`, `--show-token`) |
| `auth token` | print the token on stdout (`--json` for an object with its expiry) |
| `auth env` | print `GALAXY_URL` and `GALAXY_API_KEY` assignments, for `eval` |
| `auth headers` | print `{"x-api-key": ...}` as JSON, for an MCP `headersHelper` |
| `auth logout` | delete the stored login; a pasted API key is not revoked |

Create an API key in Galaxy's user preferences and pipe it in:
`printf %s "$KEY" | gxy auth login --host <url> --with-token`. At a terminal, `--with-token` asks
for the key without echoing it. If stdin is a terminal but stderr is redirected, where the prompt
would go unseen, it refuses (exit 64) and asks you to pipe the key in. `gxy auth login` without
`--with-token` says the same and exits 64. `auth logout` only forgets a key, because it may be in use
elsewhere: it keeps working until you revoke it in Galaxy's user preferences, and `auth logout` says
so.

### Where logins are kept

Tokens live in the system keyring when there is one, otherwise in `hosts.yml` in the config
directory (`$GXY_CONFIG_DIR`, else `$XDG_CONFIG_HOME/gxy`, else `~/.config/gxy`; a relative path in
either variable is ignored), readable only by you. A token that carries admin rights is only ever
stored in the system keyring: with no keyring, `gxy auth login` refuses and stores nothing, unless
you add `--insecure-storage` to put it in `hosts.yml`.

`gxy` refuses a `hosts.yml` that other users can write to (it would not know what they put in it),
that is a symlink or belongs to someone else, and one it cannot make sense of rather than overwrite
it. A file or config directory that others can read or enter is narrowed to 600 or 700 with a
warning. An entry written by a different version of `gxy` (a storage or method this one does not
know) is skipped with a warning and kept in the file as it is, so it does not stop your other
logins from working; asking for that very Galaxy fails and says it was written by a newer version,
rather than that you are not logged in. Each refusal exits 65 (not 77, which would suggest logging in
again, and that meets the same refusal), names the file and what to do, and changes nothing.

If the credential store cannot be read (a corrupt `hosts.yml`, a config directory that is a
symlink, a login a newer `gxy` wrote for the Galaxy in question, a token in a system keyring that
cannot be reached), an operation with `--api-key`, a
profile key or `GALAXY_API_KEY` does not read it at all. One that has only a `.env` key still runs,
with a warning on each run; any other fails, and the error names the problem. Fix or remove the
store, or set `GXY_CONFIG_DIR` somewhere it can be read, to stop it.

The system keyring is asked through a short-lived helper process that is killed if it does not
answer, so a dead D-Bus costs at most one wait per command before the keyring is reported as
unavailable: 5 seconds, or 60 when `gxy` is writing to a terminal, so that you have time to type a
password into a keyring unlock prompt (it says "Waiting for the system keyring" when that is what
it is doing). The warning when a token goes to `hosts.yml` instead says why. `GXY_NO_KEYRING=1`
skips the keyring entirely. If the keyring cannot be read, `auth logout` still removes the stored
record (it says the secret may remain in the keyring, and exits 1) and `auth status`
reports that host as "token unavailable" and exits 1. Logging in again over a login that is in the
keyring, when `gxy` cannot use the keyring to remove the old secret (`GXY_NO_KEYRING`, or a keyring
that does not answer), stores the new login and warns that the old secret may remain, naming the
entry (`gxy:<url>`) to remove with your keyring's own tool.

### Which key and which Galaxy an operation uses

The key, first match wins: `--api-key`, then the key of the `--profile` you name, then
`GALAXY_API_KEY` in the environment, then the stored login for the Galaxy being used, then
`GALAXY_API_KEY` in `.env`. The command line beats the environment, which beats files, as in the
aws CLI. Two things differ from before stored logins existed: a `--profile` with a key now beats
`GALAXY_API_KEY` and a `.env` key (it used to rank below both), and a stored login ranks above a
`.env` key. `--api-key` is deprecated (it shows in shell history).

The Galaxy: `--url`, then the `--profile` URL (when the profile has a `galaxy_url`: a profile is a
pair, so its key goes to its own Galaxy), then `GALAXY_URL` in the environment, then `GALAXY_URL` in
`.env`, then the only stored login. Before stored logins existed, `GALAXY_URL` and the `.env` URL
beat the profile's.

A key that is not the stored login's is only ever sent to a Galaxy that something named, never to
the only stored login by default: with a key and no Galaxy named, `gxy` stops with "No Galaxy
credentials" and asks for the Galaxy. A profile's key goes to its own URL, or to the Galaxy you
named with `--url` or `GALAXY_URL` when it has none, and never to one that a `.env` chose. A stored
token is only sent to the host it was issued for: name another Galaxy with no key for it, and the
error says which Galaxies you are logged in to and how to log in to that one.

One case is unchanged from before stored logins existed: a key from `--api-key` or `GALAXY_API_KEY`
is sent to whatever Galaxy is resolved, including one that only a `.env` names. So if you have
`GALAXY_API_KEY` set and run `gxy` in a directory whose `.env` has a `GALAXY_URL`, that key goes to
that Galaxy. The notice on stderr (`Using GALAXY_URL from <dir>/.env.`) is the only warning, and
`--quiet` hides it.

A `.env` comes with the directory you are in, often a clone of someone else's repository, so an
operation that uses its `GALAXY_API_KEY` or its `GALAXY_URL` says so on stderr, naming the file, and
says when a stored login outranked its key. `--quiet` silences both.

`gxy auth status` shows, for each login, which of these an operation run with the same flags and
environment would use for that Galaxy (`--api-key`, `--profile <name>`, `GALAXY_API_KEY`,
`stored login` or `.env`), says at the top when something outranks the stored logins, and reports the
same as `key_source` in `--json` (`flag`, `profile`, `env`, `stored`, `dotenv`). When an operation
with those flags would go to another Galaxy (a `--profile` or `GALAXY_URL` that names one), or would
stop with an error (a key and no Galaxy named, or several logins and none named), it names no
source: the line says `none` and why, and `key_source` is `null`.

The `auth` commands (except `login`) pick their Galaxy by this same order, so `gxy auth status` and
`gxy get_user` cannot disagree about which Galaxy a set of flags and variables means; `--host` is
the same as `--url`. They never use a key from
`--api-key`, `GALAXY_API_KEY`, `.env` or a profile, and warn once on stderr if `--api-key` is given
(and, for `auth login` only, `--profile`); `auth status` does not warn, as it reports on them. `auth login` never takes its host from `.env` or a
profile (a repository must not choose where your key goes): give `--host` or set `GALAXY_URL`.

### What is checked before a key is sent

Before a key goes anywhere, `gxy` checks that the address answers as a Galaxy (`/api/version`),
and it never follows a redirect with a key attached: a redirect is reported as a wrong address,
with the place it points to. Plain `http://` is refused (exit 76) for any host but this machine
(`localhost`, `127.0.0.0/8`, `::1`), because a token would cross the network in the clear.

`auth login` and `auth status` check a Galaxy with two requests (a version probe, then the user
lookup). In `auth login` each gives up after 15 seconds; `auth status` allows each Galaxy 10 seconds
in all, and checks every stored login at once, printing them in stored order. Ctrl-C stops a check
at once.

`auth status` exits 1 when any login is not usable right now (expired, its
token unreadable, refused by Galaxy, or Galaxy did not answer), 0 when all are fine, and 77 when
nothing is stored. With `--json` it exits 0 whenever it prints the list (each entry's `status` says
how its check went); a failure before there is a list, such as an unreadable `hosts.yml`, still
exits non-zero.

| Variable | Does |
| --- | --- |
| `GXY_CONFIG_DIR` | Where `hosts.yml` lives (see above). A relative path is ignored, so the file does not follow the working directory |
| `GXY_NO_KEYRING` | `1` never touches the system keyring: tokens go to `hosts.yml`, with a warning. Use it on a box whose keyring is broken or locked, to skip the wait. A login already stored in the keyring then cannot be read (`auth token` exits 77, `auth status` shows "token unavailable"), and `auth logout` removes its record but leaves the secret in the keyring |

Behind a proxy, set `NODE_USE_ENV_PROXY=1` along with `HTTPS_PROXY` (and `NO_PROXY` for a Galaxy on
your own network). Node's `fetch` ignores the proxy variables without it, and reads it only at
startup, so it has to be in the environment before `gxy` runs. It needs Node 22.21+ or 24+. If
`gxy` cannot reach the Galaxy while a proxy variable is set and the flag is not, it says so (not when
the Galaxy answered with an error, which a proxy being ignored would not cause). For a Galaxy with a
certificate from a private CA, set `NODE_EXTRA_CA_CERTS=/path/to/ca.pem`. Login never offers to
switch off certificate checks.

```bash
printf %s "$KEY" | NODE_USE_ENV_PROXY=1 HTTPS_PROXY=http://proxy.example.org:3128 NO_PROXY=localhost gxy auth login --host https://galaxy.example.org --with-token
```

### Keeping tokens away from agents

`gxy auth env` puts the token in the environment of every process started from that shell,
and `gxy auth token` prints it where a transcript can keep it. Agents should run
`gxy <operation>` and let it read the stored login, rather than handling tokens themselves.
In Claude Code you can say so in `.claude/settings.json`:

```json
{
  "permissions": {
    "deny": ["Bash(gxy auth token*)", "Bash(gxy auth env*)", "Read(~/.config/gxy/**)"]
  }
}
```

These rules match the command text; they are not a sandbox. A different spelling of the
same command, or a script that runs it, is not caught.

### Galaxy's MCP endpoint

To give Claude Code the stored login for a Galaxy's `/api/mcp` endpoint without putting a key
in the config file:

```json
{
  "mcpServers": {
    "galaxy": {
      "type": "http",
      "url": "https://usegalaxy.org/api/mcp",
      "headersHelper": "gxy auth headers"
    }
  }
}
```

## Documentation

See the [TypeScript workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/typescript#readme)
for the full operation list, output formats, and exit-code reference.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
