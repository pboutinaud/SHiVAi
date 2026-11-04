"""Test the standalone download wrapper without containers or network access."""
import argparse
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / 'apptainer' / 'download_models.py'
spec = importlib.util.spec_from_file_location('download_models', SCRIPT)
assert spec is not None and spec.loader is not None
download_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_models)


class DownloadModelsTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch('sys.stdout', new_callable=io.StringIO))
        self.stderr = self.enterContext(patch('sys.stderr', new_callable=io.StringIO))
        self.run = self.enterContext(patch.object(download_models.subprocess, 'run'))
        self.run.return_value.returncode = 0

    def test_docker_creates_output_and_forwards_helper_arguments(self):
        output = self.directory / 'models with spaces'
        result = download_models.main([
            '--docker', 'vnozais/shivai:latest', '-m', 'PVS', 'WMH',
            '-o', str(output), '-v', 'V5',
        ])
        self.assertEqual(result, 0)
        self.assertTrue(output.is_dir())
        self.run.assert_called_once()
        command = self.run.call_args.args[0]
        target = str(output.resolve()) if download_models.os.name == 'posix' else '/mnt/model'
        self.assertEqual(command[:3], ['docker', 'run', '--rm'])
        self.assertEqual(command[command.index('-v') + 1], f'{output.resolve()}:{target}:rw')
        self.assertEqual(command[command.index('-w') + 1], target)
        self.assertEqual(command[command.index('-e') + 1], f'HOME={target}')
        self.assertEqual(command[command.index('vnozais/shivai:latest') + 1:], [
            'shiva_download', '--models', 'PVS', 'WMH', '--output-dir', target,
            '--version', 'V5',
        ])
        self.assertEqual(self.run.call_args.kwargs, {})

    def test_posix_docker_maps_user_and_preserves_output_path(self):
        args = argparse.Namespace(docker='shivai:latest', apptainer=None,
                                  models=['all'], version=None)
        with patch.object(download_models.os, 'name', 'posix'), \
                patch.object(download_models.os, 'getuid', return_value=123, create=True), \
                patch.object(download_models.os, 'getgid', return_value=456, create=True):
            command = download_models.build_exec_command(args, '/home/user/my models')
        self.assertEqual(command[3:5], ['--user', '123:456'])
        self.assertEqual(command[-1], '/home/user/my models')

    def test_windows_docker_omits_unix_user_mapping(self):
        args = argparse.Namespace(docker='shivai:latest', apptainer=None,
                                  models=['brainmask'], version=None)
        with patch.object(download_models.os, 'name', 'nt'):
            command = download_models.build_exec_command(args, r'C:\my models')
        self.assertNotIn('--user', command)
        self.assertIn(r'C:\my models:/mnt/model:rw', command)
        self.assertEqual(command[-1], '/mnt/model')

    def test_apptainer_command_uses_image_and_writable_bind(self):
        args = argparse.Namespace(docker=None, apptainer='/images/my shivai.sif',
                                  models=['PVS'], version='V4')
        with patch.object(download_models.os, 'name', 'posix'):
            command = download_models.build_exec_command(args, '/home/user/models')
        self.assertEqual(command, [
            'apptainer', 'exec', '--bind', '/home/user/models:/home/user/models:rw',
            '--pwd', '/home/user/models', '/images/my shivai.sif',
            'shiva_download', '--models', 'PVS', '--output-dir', '/home/user/models',
            '--version', 'V4',
        ])

    def test_apptainer_resolves_image_and_runs_helper(self):
        image = self.directory / 'my shivai.sif'
        image.touch()
        output = self.directory / 'models'
        with patch.object(download_models, 'os') as host_os:
            host_os.name = 'posix'
            result = download_models.main([
                '--apptainer', str(image), '-m', 'PVS', '-o', str(output),
            ])
        self.assertEqual(result, 0)
        self.assertTrue(output.is_dir())
        command = self.run.call_args.args[0]
        self.assertEqual(command[:2], ['apptainer', 'exec'])
        self.assertIn(str(image.resolve()), command)
        self.assertEqual(command[-1], str(output.resolve()))

    def test_missing_apptainer_image_is_rejected(self):
        with patch.object(download_models, 'os') as host_os:
            host_os.name = 'posix'
            with self.assertRaises(SystemExit) as exc:
                download_models.main([
                    '--apptainer', str(self.directory / 'missing.sif'),
                    '-m', 'all', '-o', str(self.directory),
                ])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn('Apptainer image does not exist', self.stderr.getvalue())
        self.run.assert_not_called()

    def test_native_windows_apptainer_is_rejected(self):
        with patch.object(download_models, 'os') as host_os:
            host_os.name = 'nt'
            with self.assertRaises(SystemExit) as exc:
                download_models.main([
                    '--apptainer', 'shivai.sif', '-m', 'all', '-o', str(self.directory),
                ])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn('WSL', self.stderr.getvalue())
        self.run.assert_not_called()

    def test_runtime_is_required_and_mutually_exclusive(self):
        for runtime in ([], ['--docker', 'shivai', '--apptainer', 'shivai.sif']):
            with self.subTest(runtime=runtime), self.assertRaises(SystemExit) as exc:
                download_models.main([*runtime, '-m', 'all', '-o', str(self.directory)])
            self.assertEqual(exc.exception.code, 2)
        self.run.assert_not_called()

    def test_picker_selection_and_cancellation(self):
        with patch.object(download_models, 'pick_output_dir', return_value=str(self.directory)) as picker:
            self.assertEqual(download_models.main(['--docker', 'shivai', '-m', 'all']), 0)
            picker.assert_called_once_with()
        self.run.reset_mock()
        with patch.object(download_models, 'pick_output_dir', return_value=''):
            self.assertEqual(download_models.main(['--docker', 'shivai', '-m', 'all']), 0)
        self.run.assert_not_called()

    def test_picker_error_is_reported_without_starting_container(self):
        with patch.object(download_models, 'pick_output_dir',
                          side_effect=RuntimeError('Cannot open the folder picker')), \
                self.assertRaises(SystemExit) as exc:
            download_models.main(['--docker', 'shivai', '-m', 'all'])
        self.assertEqual(exc.exception.code, 1)
        self.assertIn('Cannot open the folder picker', self.stderr.getvalue())
        self.run.assert_not_called()

    def test_host_picker_destroys_window_on_selection_or_error(self):
        tk = MagicMock()
        tk.TclError = RuntimeError
        dialog = MagicMock()
        with patch.dict('sys.modules', {'tkinter': tk, 'tkinter.filedialog': dialog}):
            tk.filedialog = dialog
            dialog.askdirectory.return_value = str(self.directory)
            self.assertEqual(download_models.pick_output_dir(), str(self.directory))
            root = tk.Tk.return_value
            dialog.askdirectory.assert_called_once_with(
                parent=root, title='Download SHiVAi models to')
            root.destroy.assert_called_once_with()
            root.destroy.reset_mock()
            dialog.askdirectory.side_effect = RuntimeError('no display')
            with self.assertRaisesRegex(RuntimeError, 'provide -o'):
                download_models.pick_output_dir()
            root.destroy.assert_called_once_with()

    def test_output_file_is_rejected(self):
        output = self.directory / 'not-a-directory'
        output.touch()
        with self.assertRaises(SystemExit) as exc:
            download_models.main(['--docker', 'shivai', '-m', 'all', '-o', str(output)])
        self.assertEqual(exc.exception.code, 1)
        self.assertIn('error:', self.stderr.getvalue())
        self.run.assert_not_called()

    def test_container_failure_status_is_preserved(self):
        self.run.return_value.returncode = 7
        self.assertEqual(download_models.main([
            '--docker', 'shivai', '-m', 'all', '-o', str(self.directory),
        ]), 7)

    def test_missing_runtime_is_reported(self):
        self.run.side_effect = FileNotFoundError('docker executable not found')
        with self.assertRaises(SystemExit) as exc:
            download_models.main([
                '--docker', 'shivai', '-m', 'all', '-o', str(self.directory),
            ])
        self.assertEqual(exc.exception.code, 1)
        self.assertIn('docker executable not found', self.stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
