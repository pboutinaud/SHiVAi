'''
DataSink interface classes overloaded to append data to csv if already exists in the sink.
Also add an "Append date" column for the lines appended to the csv.
For PDF files already existing, it saves the new file with a number added at the end of the
filename (e.g. "report_2.pdf" if "report.pdf" already exists)

All copy-pasted from _list_outputs of DataSink. The modified part can be found under the
two "# NOTE: OVERLOADED" tags

The problem is that if there are several processes trying to append at the same time, some
of the data may be lost...
'''

import os
import shutil
import string
from nipype.interfaces.io import DataSink, DataSinkInputSpec, copytree
from nipype.utils.misc import str2bool
from nipype import config, logging
from nipype.interfaces.base import isdefined, traits, Str
from nipype.utils.filemanip import copyfile, ensure_list
from shivai.utils.girder_utils import _disable_girder_ssl_verification, GIRDER_API_KEY_ENV, GIRDER_USERNAME_ENV, GIRDER_PASSWORD_ENV
import pandas as pd
from datetime import date
iflogger = logging.getLogger("nipype.interface")


class DataSink_CSV_and_PDF_safe(DataSink):
    # List outputs, main run routine
    def _list_outputs(self):
        """Execute this module."""

        # Init variables
        outputs = self.output_spec().get()
        out_files = []
        # Use hardlink
        use_hardlink = str2bool(config.get("execution", "try_hard_link_datasink"))

        # Set local output directory if specified
        if isdefined(self.inputs.local_copy):
            outdir = self.inputs.local_copy
        else:
            outdir = self.inputs.base_directory
            # If base directory isn't given, assume current directory
            if not isdefined(outdir):
                outdir = "."

        # Check if base directory reflects S3 bucket upload
        s3_flag, bucket_name = self._check_s3_base_dir()
        if s3_flag:
            s3dir = self.inputs.base_directory
            # If user overrides bucket object, use that
            if self.inputs.bucket:
                bucket = self.inputs.bucket
            # Otherwise fetch bucket object using name
            else:
                try:
                    bucket = self._fetch_bucket(bucket_name)
                # If encountering an exception during bucket access, set output
                # base directory to a local folder
                except Exception as exc:
                    s3dir = "<N/A>"
                    if not isdefined(self.inputs.local_copy):
                        local_out_exception = os.path.join(
                            os.path.expanduser("~"), "s3_datasink_" + bucket_name
                        )
                        outdir = local_out_exception
                    # Log local copying directory
                    iflogger.info(
                        "Access to S3 failed! Storing outputs locally at: "
                        "%s\nError: %s",
                        outdir,
                        exc,
                    )
        else:
            s3dir = "<N/A>"

        # If container input is given, append that to outdir
        if isdefined(self.inputs.container):
            outdir = os.path.join(outdir, self.inputs.container)
            s3dir = os.path.join(s3dir, self.inputs.container)

        # If sinking to local folder
        if outdir != s3dir:
            outdir = os.path.abspath(outdir)
            # Create the directory if it doesn't exist
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except OSError as inst:
                    if "File exists" in inst.strerror:
                        pass
                    else:
                        raise (inst)

        # Iterate through outputs attributes {key : path(s)}
        for key, files in list(self.inputs._outputs.items()):
            if not isdefined(files):
                continue
            iflogger.debug("key: %s files: %s", key, str(files))
            files = ensure_list(files)
            tempoutdir = outdir
            if s3_flag:
                s3tempoutdir = s3dir
            for d in key.split("."):
                if d[0] == "@":
                    continue
                tempoutdir = os.path.join(tempoutdir, d)
                if s3_flag:
                    s3tempoutdir = os.path.join(s3tempoutdir, d)

            # flattening list
            if isinstance(files, list):
                if isinstance(files[0], list):
                    files = [item for sublist in files for item in sublist]

            # Iterate through passed-in source files
            for src in ensure_list(files):
                # Format src and dst files
                src = os.path.abspath(src)
                if not os.path.isfile(src):
                    src = os.path.join(src, "")
                dst = self._get_dst(src)
                if s3_flag:
                    s3dst = os.path.join(s3tempoutdir, dst)
                    s3dst = self._substitute(s3dst)
                dst = os.path.join(tempoutdir, dst)
                dst = self._substitute(dst)
                path, _ = os.path.split(dst)

                # If we're uploading to S3
                if s3_flag:
                    self._upload_to_s3(bucket, src, s3dst)
                    out_files.append(s3dst)
                # Otherwise, copy locally src -> dst
                if not s3_flag or isdefined(self.inputs.local_copy):
                    # Create output directory if it doesn't exist
                    if not os.path.exists(path):
                        try:
                            os.makedirs(path)
                        except OSError as inst:
                            if "File exists" in inst.strerror:
                                pass
                            else:
                                raise (inst)
                    # If src is a file, copy it to dst
                    if os.path.isfile(src):
                        # NOTE: OVERLOADED Start
                        if os.path.exists(dst) and os.path.splitext(dst)[-1] == '.csv':
                            iflogger.debug("append_csv: %s %s", src, dst)
                            dst_ori_df = pd.read_csv(dst)
                            src_df = pd.read_csv(src)
                            src_df['Append date'] = date.today()
                            dst_df = pd.concat([dst_ori_df, src_df])
                            dst_df.to_csv(dst, index=False)
                        elif os.path.exists(dst) and os.path.splitext(dst)[-1] == '.pdf':
                            n = 2
                            dst_bn, ext = os.path.splitext(dst)
                            savename = dst_bn + f'_{n}' + ext
                            while os.path.exists(savename):
                                n += 1
                                savename = dst_bn + f'_{n}' + ext
                            iflogger.debug("copyfile: %s %s", src, savename)
                            copyfile(
                                src,
                                savename,
                                copy=True,
                                hashmethod="content",
                                use_hardlink=use_hardlink,
                            )
                        else:
                            iflogger.debug("copyfile: %s %s", src, dst)
                            copyfile(
                                src,
                                dst,
                                copy=True,
                                hashmethod="content",
                                use_hardlink=use_hardlink,
                            )
                        out_files.append(dst)
                        # NOTE: OVERLOADED End
                    # If src is a directory, copy entire contents to dst dir
                    elif os.path.isdir(src):
                        if os.path.exists(dst) and self.inputs.remove_dest_dir:
                            iflogger.debug("removing: %s", dst)
                            shutil.rmtree(dst)
                        iflogger.debug("copydir: %s %s", src, dst)
                        copytree(src, dst)
                        out_files.append(dst)

        # Return outputs dictionary
        outputs["out_file"] = out_files

        return outputs




class GirderSinkInputSpec(DataSinkInputSpec):
    """Input spec for GirderSink. Extends nipype's DataSinkInputSpec (base_directory,
    container, substitutions, parameterization, etc.) so that GirderSink can reconstruct
    the exact same relative path the local DataSink would use (see `_get_dst`/`_substitute`),
    on top of which we add the Girder-specific connection/routing traits below.
    """

    host = Str(mandatory=True, desc="Girder API URL (e.g. https://girder.example.com/api/v1)")

    auth_method = traits.Enum(
        "api_key", "password", usedefault=True,
        desc="Which credentials to read from the environment: 'api_key' or 'password' (username+password)"
    )

    verify_ssl = traits.Bool(
        True, usedefault=True,
        desc=(
            "Whether to verify the Girder server's SSL certificate. Only set to false for quick "
            "debugging against a server with a self-signed/invalid certificate: this disables all "
            "certificate verification and should not be used against a Girder server with sensitive "
            "data over an untrusted network."
        )
    )

    # NOTE: these are only the *names* of environment variables, never the secret
    # values themselves, so nothing sensitive is ever stored as a node input.
    api_key_env = Str(GIRDER_API_KEY_ENV, usedefault=True,
                      desc="Name of the environment variable holding the Girder API key")
    username_env = Str(GIRDER_USERNAME_ENV, usedefault=True,
                       desc="Name of the environment variable holding the Girder username")
    password_env = Str(GIRDER_PASSWORD_ENV, usedefault=True,
                       desc="Name of the environment variable holding the Girder password")

    mapping = traits.Dict(
        Str, mandatory=True,
        desc=(
            "Description of the Girder upload tree (collection/root_folder_id, "
            "subjectwise_folders, global_folders, overwrite, create_missing_folders). "
            "Built from the 'girder' section of the shivai config YAML - see the "
            "README's 'Uploading results to Girder' section for the full format."
        )
    )

    subject_id = Str(
        mandatory=True,
        desc="Subject id, substituted for the '$subject_id' placeholder in the mapping's path templates"
    )


class GirderSink(DataSink):
    """Sink that uploads workflow outputs to a Girder server.

    Similar in spirit to nipype's XNATSink, but instead of a fixed
    project/subject/experiment hierarchy, the destination Girder folder for each output
    key is derived from a path template read from the `mapping` dict (built from the
    'girder' section of the shivai config YAML). See the README's 'Uploading results to
    Girder' section for the full format and an example with every possible key.

    GirderSink subclasses nipype's DataSink (rather than starting from a bare IOBase) so
    that it can reuse `_get_dst`/`_substitute` to compute the exact same relative path the
    local DataSink uses for the same inputs - this relative path is recorded as Girder item
    metadata ("original_path") on every uploaded file, so uploads stay traceable back to the
    local `results/` layout. `_list_outputs` is fully overridden: GirderSink never writes to
    `self.inputs.base_directory` itself, only uses it (and `container`/`substitutions`/
    `parameterization`) to compute that relative path.

    Missing folders in the destination Girder hierarchy are created on demand (unless
    `create_missing_folders` is set to false in the mapping). If `overwrite` is false
    (default) and a same-named item already exists in the destination folder, the upload
    is skipped (with a warning) rather than duplicated.

    Credentials are never passed as node inputs (nipype pickles node inputs to disk in its
    working-directory cache and in crash files, which would otherwise leak the secret).
    Instead, `auth_method`/`api_key_env`/`username_env`/`password_env` only carry the *name*
    of an environment variable; the actual secret is read from `os.environ` at execution
    time. See `shivai.utils.girder_utils._resolve_girder_credentials`.
    """

    input_spec = GirderSinkInputSpec
    _pkg = "girder_client"

    @staticmethod
    def _mapping_key(key):
        """Strip the '@...' dot-segments from a DataSink output key, so that e.g.
        'segmentations.pvs_segmentation' and 'segmentations.pvs_segmentation.@foldwise'
        both resolve to the same mapping entry/Girder folder - matching how DataSink
        itself groups them into the same local subdirectory.
        """
        return ".".join(d for d in key.split(".") if not d.startswith("@"))

    def _local_relative_path(self, key, src):
        """Reconstruct the path (relative to base_directory) that the local
        DataSink_CSV_and_PDF_safe sink would use for this (key, src) pair, by reusing
        DataSink's own `_get_dst`/`_substitute` helpers.
        """
        base_directory = self.inputs.base_directory
        tempoutdir = os.path.abspath(base_directory)
        if isdefined(self.inputs.container):
            tempoutdir = os.path.join(tempoutdir, self.inputs.container)
        for d in key.split("."):
            if d.startswith("@"):
                continue
            tempoutdir = os.path.join(tempoutdir, d)
        src = os.path.abspath(src)
        dst = self._get_dst(src)
        dst = os.path.join(tempoutdir, dst)
        dst = self._substitute(dst)
        return os.path.relpath(dst, os.path.abspath(base_directory))

    def _resolve_root(self, gc, mapping, create_missing_folders):
        root_folder_id = mapping.get("root_folder_id")
        if root_folder_id:
            return root_folder_id, "folder"
        collection = mapping.get("collection")
        if not collection:
            raise ValueError(
                "GirderSink: the 'girder' config section must set either 'collection' or 'root_folder_id'"
            )
        resource = gc.resourceLookup(f"/collection/{collection}", test=True)
        if resource:
            return resource["_id"], "collection"
        if not create_missing_folders:
            raise ValueError(
                f"GirderSink: Girder collection '{collection}' not found, and "
                "'create_missing_folders' is false in the config"
            )
        created = gc.createCollection(collection)
        return created["_id"], "collection"

    def _resolve_folder(self, gc, root_id, root_type, relative_path, create_missing_folders, cache):
        parent_id, parent_type = root_id, root_type
        for component in [c for c in relative_path.split("/") if c]:
            cache_key = (parent_id, component)
            if cache_key in cache:
                parent_id = cache[cache_key]
                parent_type = "folder"
                continue
            if create_missing_folders:
                folder = gc.createFolder(
                    parent_id, component, parentType=parent_type, reuseExisting=True
                )
            else:
                folder = next(
                    (f for f in gc.listFolder(parent_id, parentFolderType=parent_type, name=component)),
                    None,
                )
                if folder is None:
                    raise ValueError(
                        f"GirderSink: folder '{component}' not found under parent "
                        f"'{parent_id}', and 'create_missing_folders' is false in the config"
                    )
            cache[cache_key] = folder["_id"]
            parent_id = folder["_id"]
            parent_type = "folder"
        return parent_id

    def _item_exists(self, gc, folder_id, filename):
        return next((i for i in gc.listItem(folder_id, name=filename)), None) is not None

    def _upload_file(self, gc, folder_id, src, relative_original_path, overwrite):
        filename = os.path.basename(src)
        if not overwrite and self._item_exists(gc, folder_id, filename):
            iflogger.warning(
                "GirderSink: item '%s' already exists in folder '%s', skipping upload (overwrite=false)",
                filename, folder_id
            )
            return None
        uploaded = gc.uploadFileToFolder(folder_id, src)
        item_id = uploaded.get("itemId", uploaded.get("_id"))
        if item_id:
            gc.addMetadataToItem(item_id, {"original_path": relative_original_path})
        return uploaded

    def _list_outputs(self):
        """Execute this module."""
        import girder_client

        outputs = self.output_spec().get()
        out_files = []

        # Keys starting with '_' are ignored, in case any comment-style key ends up in here
        mapping = {k: v for k, v in self.inputs.mapping.items() if not k.startswith("_")}
        overwrite = bool(mapping.get("overwrite", False))
        create_missing_folders = bool(mapping.get("create_missing_folders", True))
        subjectwise_folders = mapping.get("subjectwise_folders", {}) or {}
        global_folders = mapping.get("global_folders", {}) or {}

        # Setup Girder connection. Credentials are resolved from this process'
        # environment only (never stored as a trait) - see the module-level
        # note above `GIRDER_API_KEY_ENV` for why.
        gc = girder_client.GirderClient(apiUrl=self.inputs.host)
        if not self.inputs.verify_ssl:
            _disable_girder_ssl_verification(gc)
        if self.inputs.auth_method == "api_key":
            api_key = os.environ.get(self.inputs.api_key_env)
            if not api_key:
                raise ValueError(
                    f"GirderSink: environment variable '{self.inputs.api_key_env}' is not set, "
                    "cannot authenticate to Girder (auth_method='api_key')"
                )
            gc.authenticate(apiKey=api_key)
        else:
            username = os.environ.get(self.inputs.username_env)
            password = os.environ.get(self.inputs.password_env)
            if not username or not password:
                raise ValueError(
                    f"GirderSink: environment variables '{self.inputs.username_env}'/'{self.inputs.password_env}' "
                    "are not both set, cannot authenticate to Girder (auth_method='password')"
                )
            gc.authenticate(username=username, password=password)

        root_id, root_type = self._resolve_root(gc, mapping, create_missing_folders)
        folder_cache = {}

        # Iterate through outputs attributes {key : path(s)}
        for key, files in list(self.inputs._outputs.items()):
            if not isdefined(files):
                continue

            mapping_key = self._mapping_key(key)
            folder_template = subjectwise_folders.get(mapping_key, global_folders.get(mapping_key))
            if folder_template is None:
                iflogger.warning(
                    "GirderSink: no mapping found for output key '%s' (subject '%s'), skipping upload",
                    key, self.inputs.subject_id
                )
                continue
            relative_folder = string.Template(folder_template).safe_substitute(subject_id=self.inputs.subject_id)
            folder_id = self._resolve_folder(gc, root_id, root_type, relative_folder, create_missing_folders, folder_cache)

            files = ensure_list(files)
            # flattening list
            if isinstance(files, list) and files and isinstance(files[0], list):
                files = [item for sublist in files for item in sublist]

            for src in ensure_list(files):
                src = os.path.abspath(src)
                relative_original_path = self._local_relative_path(key, src)
                if os.path.isfile(src):
                    iflogger.debug("girder upload file: %s -> folder %s", src, folder_id)
                    uploaded = self._upload_file(gc, folder_id, src, relative_original_path, overwrite)
                    if uploaded is not None:
                        out_files.append(uploaded)
                elif os.path.isdir(src):
                    for root, _, filenames in os.walk(src):
                        for fname in filenames:
                            fsrc = os.path.join(root, fname)
                            rel_original_path = self._local_relative_path(key, fsrc)
                            iflogger.debug("girder upload file: %s -> folder %s", fsrc, folder_id)
                            uploaded = self._upload_file(gc, folder_id, fsrc, rel_original_path, overwrite)
                            if uploaded is not None:
                                out_files.append(uploaded)

        outputs["out_file"] = out_files
        return outputs
