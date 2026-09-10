
import os
import pwinput

import girder_client


# Names of the environment variables read by GirderSink to authenticate. The
# actual secret values are never passed around as nipype traits (which would
# get pickled to disk in node caches / crash files): they are only ever read
# from the current process' environment at execution time (_list_outputs).
# See shivai.utils.girder_utils._resolve_girder_credentials for how these
# variables get populated (via `pwinput`, or programmatically for automated
# callers).
GIRDER_API_KEY_ENV = "SHIVAI_GIRDER_API_KEY"
GIRDER_USERNAME_ENV = "SHIVAI_GIRDER_USERNAME"
GIRDER_PASSWORD_ENV = "SHIVAI_GIRDER_PASSWORD"


def _disable_girder_ssl_verification(gc: girder_client.GirderClient):
    """
    Make a girder_client.GirderClient use an insecure requests.Session (no SSL certificate
    verification), for debugging against a Girder server with a self-signed/invalid
    certificate. Also silences the resulting urllib3 InsecureRequestWarning, since the user
    already explicitly opted into this (via `verify_ssl: false` in the config).
    """
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = requests.Session()
    session.verify = False
    # GirderClient has no public API to disable SSL verification, but it uses this session
    # for every request (including authenticate()) if one is set - see `_requestFunc`.
    gc._session = session


def _check_girder_auth(girder_host, auth_method, api_key=None, username=None, password=None, verify_ssl=True):
    """
    Perform a lightweight, fail-fast authentication check against the Girder server,
    so that connectivity/authentication issues (unreachable host, wrong URL, invalid or
    expired API key/credentials) are reported immediately, before the (potentially long)
    workflow is built and run - rather than only failing much later, in a GirderSink node.

    If `verify_ssl` is false, SSL certificate verification is disabled (for quick debugging
    against a server with a self-signed/invalid certificate only - see `datasink.py`'s
    `_disable_girder_ssl_verification`).
    """

    gc = girder_client.GirderClient(apiUrl=girder_host)
    if not verify_ssl:
        _disable_girder_ssl_verification(gc)
    try:
        if auth_method == 'api_key':
            gc.authenticate(apiKey=api_key)
        else:
            gc.authenticate(username=username, password=password)
    except girder_client.AuthenticationError as exc:
        raise RuntimeError(
            f"GirderSink: authentication to '{girder_host}' failed (auth_method='{auth_method}'): "
            "check the supplied API key/username/password."
        ) from exc
    except girder_client.HttpError as exc:
        raise RuntimeError(
            f"GirderSink: could not reach/authenticate to the Girder server at '{girder_host}': {exc}"
        ) from exc
    except Exception as exc:  # e.g. requests.exceptions.ConnectionError for an unreachable host
        raise RuntimeError(
            f"GirderSink: could not connect to the Girder server at '{girder_host}': {exc}"
        ) from exc


def _resolve_girder_credentials(girder_host, auth_method, girder_api_key=None, girder_username=None, girder_password=None, verify_ssl=True):
    """
    Resolve the Girder credentials needed by the GirderSink nodes, check that they actually
    work against `girder_host` (failing fast if not), and store them **only** in this
    process' environment (never as a nipype trait), so that they are never pickled to disk
    in nipype's working-directory cache or crash files.

    `verify_ssl=False` disables SSL certificate verification for this check (and is also
    threaded down to the GirderSink nodes) - use only for quick debugging against a server
    with a self-signed/invalid certificate.

    Resolution order (first match wins) for each secret:
      1. Value passed directly as a function argument (for programmatic/library callers,
         e.g. an automated process calling `shiva()` directly from Python).
      2. Value already present in this process' environment (e.g. set by an automated/CI
         caller before invoking the `shiva` command).
      3. Interactive prompt: masked input via `pwinput` for secrets, plain input for the
         (non-secret) username.

    Note: since nipype's MultiProc plugin uses `fork`-based multiprocessing on Linux, the
    child processes inherit this process' environment, so the env vars set here are visible
    to the GirderSink node when it runs. This will NOT work for plugins that submit jobs to a
    separate environment (e.g. SLURM) unless that plugin is configured to forward the
    SHIVAI_GIRDER_* environment variables to the submitted jobs.
    """
    if auth_method == 'api_key':
        api_key = girder_api_key or os.environ.get(GIRDER_API_KEY_ENV) or os.environ.get('GIRDER_API_KEY')
        if not api_key:
            api_key = pwinput.pwinput(prompt='Girder API key: ')
        _check_girder_auth(girder_host, auth_method, api_key=api_key, verify_ssl=verify_ssl)
        os.environ[GIRDER_API_KEY_ENV] = api_key
    elif auth_method == 'password':
        username = girder_username or os.environ.get(GIRDER_USERNAME_ENV) or os.environ.get('GIRDER_USERNAME')
        if not username:
            username = input('Girder username: ')
        password = girder_password or os.environ.get(GIRDER_PASSWORD_ENV) or os.environ.get('GIRDER_PASSWORD')
        if not password:
            password = pwinput.pwinput(prompt='Girder password: ')
        _check_girder_auth(girder_host, auth_method, username=username, password=password, verify_ssl=verify_ssl)
        os.environ[GIRDER_USERNAME_ENV] = username
        os.environ[GIRDER_PASSWORD_ENV] = password
    else:
        raise ValueError(f"Unknown Girder auth_method '{auth_method}', expected 'api_key' or 'password'")
