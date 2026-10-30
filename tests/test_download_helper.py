"""Test per-file download progress without accessing the network."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from tqdm import tqdm

from shivai.scripts import download_helper


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
