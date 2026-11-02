"""Test per-file download progress without accessing the network."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from tqdm import tqdm

from shivai.scripts import download_helper


class DirectoryPickerTests(unittest.TestCase):
    def test_cancelled_picker_uses_readable_colors_only_on_x11(self):
        for windowing_system in ('x11', 'win32', 'aqua'):
            with self.subTest(windowing_system=windowing_system), \
                    patch('sys.argv', ['download_helper', '--models', 'PVS']), \
                    patch.object(download_helper.tk, 'Tk') as tk_factory, \
                    patch.object(download_helper.filedialog, 'askdirectory', return_value='') as picker, \
                    patch.object(download_helper.requests, 'Session') as session_factory:
                root = tk_factory.return_value
                root.tk.call.return_value = windowing_system

                download_helper.main()

                root.withdraw.assert_called_once_with()
                root.tk.call.assert_called_once_with('tk', 'windowingsystem')
                if windowing_system == 'x11':
                    root.option_add.assert_any_call(
                        '*foreground', '#202020', 80)
                    root.option_add.assert_any_call(
                        '*background', '#eeeeee', 80)
                    root.option_add.assert_any_call(
                        '*selectForeground', '#ffffff', 80)
                    root.option_add.assert_any_call(
                        '*selectBackground', '#315a86', 80)
                else:
                    root.option_add.assert_not_called()
                picker.assert_called_once_with(parent=root)
                root.destroy.assert_called_once_with()
                session_factory.assert_not_called()

    def test_picker_error_destroys_hidden_window(self):
        with patch('sys.argv', ['download_helper', '--models', 'PVS']), \
                patch.object(download_helper.tk, 'Tk') as tk_factory, \
                patch.object(download_helper.filedialog, 'askdirectory', side_effect=RuntimeError('picker failed')):
            with self.assertRaisesRegex(RuntimeError, 'picker failed'):
                download_helper.main()

            tk_factory.return_value.destroy.assert_called_once_with()


class DownloadProgressTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.stderr = io.StringIO()
        self.stdout = self.enterContext(patch('sys.stdout', new_callable=io.StringIO))
        self.bars = []
        self.enterContext(patch.object(download_helper, 'tqdm', side_effect=self.make_bar))
        session_factory = self.enterContext(patch.object(download_helper.requests, 'Session'))
        self.session = session_factory.return_value.__enter__.return_value
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as zip_ref:
            zip_ref.writestr('model/weights.bin', b'model weights')
        self.payload = archive.getvalue()

    def make_bar(self, **kwargs):
        bar = tqdm(file=self.stderr, mininterval=0, ascii=True, **kwargs)
        self.bars.append(bar)
        return bar

    def response(self, content_length=True):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {'Content-Type': 'application/zip'}
        if content_length:
            response.headers['Content-Length'] = str(len(self.payload))
        response.iter_content.return_value = [self.payload[:10], b'', self.payload[10:]]
        return response

    def run_download(self, *models):
        with patch('sys.argv', [
            'download_helper', '--output-dir', str(self.output_dir), '--models', *models,
        ]):
            download_helper.main()

    def test_separate_progress_for_each_file_and_extraction(self):
        first, second = self.response(), self.response()
        self.session.get.side_effect = [MagicMock(), first, MagicMock(), second]

        self.run_download('PVS', 'WMH')

        filenames = ['T1-PVS.zip', 'T1.FLAIR-WMH.zip']
        self.assertEqual([bar.desc for bar in self.bars], filenames)
        for bar, filename in zip(self.bars, filenames):
            self.assertEqual(bar.n, len(self.payload))
            self.assertEqual(bar.total, len(self.payload))
            self.assertTrue(bar.disable)
            self.assertEqual((self.output_dir / filename).read_bytes(), self.payload)
            self.assertIn(filename, self.stderr.getvalue())
        self.assertIn('100%', self.stderr.getvalue())
        self.assertEqual(
            (self.output_dir / 'model' / 'weights.bin').read_bytes(), b'model weights'
        )
        first.iter_content.assert_called_once_with(chunk_size=10 * 1024)
        first.raise_for_status.assert_called_once_with()

    def test_unknown_or_zero_size_tracks_bytes_without_percentage(self):
        for size in (None, '0'):
            with self.subTest(content_length=size):
                response = self.response(content_length=False)
                if size is not None:
                    response.headers['Content-Length'] = size
                self.session.get.side_effect = [MagicMock(), response]
                self.run_download('PVS')

                self.assertIsNone(self.bars[-1].total)
                self.assertEqual(self.bars[-1].n, len(self.payload))
                self.assertNotIn('100%', self.stderr.getvalue())

    def test_download_error_closes_progress_without_reporting_success(self):
        response = self.response()

        def interrupted_chunks(chunk_size):
            yield self.payload[:10]
            raise download_helper.requests.ConnectionError('Download interrupted')

        response.iter_content.side_effect = interrupted_chunks
        self.session.get.side_effect = [MagicMock(), response]

        with self.assertRaisesRegex(download_helper.requests.ConnectionError, 'interrupted'):
            self.run_download('PVS')

        self.assertTrue(self.bars[0].disable)
        self.assertEqual(self.bars[0].n, 10)
        self.assertNotIn('100%', self.stderr.getvalue())
        self.assertNotIn('Downloaded PVS', self.stdout.getvalue())
        self.assertFalse((self.output_dir / 'model').exists())

    def test_html_response_is_rejected_before_progress_starts(self):
        response = self.response()
        response.headers['Content-Type'] = 'text/html'
        self.session.get.side_effect = [MagicMock(), response]

        with self.assertRaisesRegex(RuntimeError, 'HTML sharing page'):
            self.run_download('PVS')

        self.assertEqual(self.bars, [])
        self.assertFalse((self.output_dir / 'T1-PVS.zip').exists())


if __name__ == '__main__':
    unittest.main()
