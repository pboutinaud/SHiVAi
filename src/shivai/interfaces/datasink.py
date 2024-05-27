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
from nipype.interfaces.io import DataSink, copytree
from nipype.utils.misc import str2bool
from nipype import config, logging
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import copyfile, ensure_list
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
