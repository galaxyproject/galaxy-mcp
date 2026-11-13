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
gxy get_user                                                                   # operations use the stored login, no flags
gxy auth token                                                                 # print the stored token
gxy auth logout                                                                # delete the stored login
```

| Command | Does |
| --- | --- |
| `auth login` | store an API key read from stdin (`--with-token`, required); `--insecure-storage` allows an admin key |
| `auth token` | print the token on stdout (`--json` for an object with its expiry) |
| `auth logout` | delete the stored login; a pasted API key is not revoked |

Create an API key in Galaxy's user preferences and pipe it in:
`printf %s "$KEY" | gxy auth login --host <url> --with-token`. At a terminal, `--with-token` asks
for the key without echoing it. If stdin is a terminal but stderr is redirected, where the prompt
would go unseen, it refuses (exit 64) and asks you to pipe the key in. `gxy auth login` without
`--with-token` says the same and exits 64. `auth logout` only forgets a key, because it may be in use
elsewhere: it keeps working until you revoke it in Galaxy's user preferences, and `auth logout` says
so.

### Where logins are kept

Logins live in `hosts.yml` in the config directory (`$GXY_CONFIG_DIR`, else
`$XDG_CONFIG_HOME/gxy`, else `~/.config/gxy`; a relative path in either variable is ignored), token
and all, in plain text, readable only by you. A key that carries admin rights is the whole server,
so `gxy auth login` refuses it and stores nothing (exit 77), unless you add `--insecure-storage` to
put it in `hosts.yml`.

`gxy` refuses a `hosts.yml` that other users can write to (it would not know what they put in it),
that is a symlink or belongs to someone else, and one it cannot make sense of rather than overwrite
it. A file or config directory that others can read or enter is narrowed to 600 or 700 with a
warning. An entry written by a different version of `gxy` (a storage or method this one does not
know) is skipped with a warning and kept in the file as it is, so it does not stop your other
logins from working; asking for that very Galaxy fails and says it was written by a newer version,
rather than that you are not logged in. Each refusal exits 65 (not 77, which would suggest logging in
again, and that meets the same refusal), names the file and what to do, and changes nothing.

If the credential store cannot be read (a corrupt `hosts.yml`, a config directory that is a
symlink, a login a newer `gxy` wrote for the Galaxy in question), an operation with `--api-key`, a
profile key or `GALAXY_API_KEY` does not read it at all. One that has only a `.env` key still runs,
with a warning on each run; any other fails, and the error names the problem. Fix or remove the
store, or set `GXY_CONFIG_DIR` somewhere it can be read, to stop it.

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

The `auth` commands (except `login`) pick their Galaxy by this same order, so `gxy auth token` and
`gxy get_user` cannot disagree about which Galaxy a set of flags and variables means; `--host` is
the same as `--url`. They never use a key from
`--api-key`, `GALAXY_API_KEY`, `.env` or a profile, and warn once on stderr if `--api-key` is given
(and, for `auth login` only, `--profile`). `auth login` never takes its host from `.env` or a
profile (a repository must not choose where your key goes): give `--host` or set `GALAXY_URL`.

### What is checked before a key is sent

Before a key goes anywhere, `gxy` checks that the address answers as a Galaxy (`/api/version`),
and it never follows a redirect with a key attached: a redirect is reported as a wrong address,
with the place it points to. Plain `http://` is refused (exit 76) for any host but this machine
(`localhost`, `127.0.0.0/8`, `::1`), because a token would cross the network in the clear.

`auth login` checks a Galaxy with two requests (a version probe, then the user lookup), and each
gives up after 15 seconds. Ctrl-C stops a check at once.

| Variable | Does |
| --- | --- |
| `GXY_CONFIG_DIR` | Where `hosts.yml` lives (see above). A relative path is ignored, so the file does not follow the working directory |

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

## Documentation

See the [TypeScript workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/typescript#readme)
for the full operation list, output formats, and exit-code reference.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
