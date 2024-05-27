#!/usr/bin/env python
"""main function running the workflow, calling the parser and the shiva function"""
from shivai.utils.parsing import shivaParser, set_args_and_check
from shivai.utils.shiva_runner import shiva


def main():
    print(
        '\n'
        'SHiVAi segmentation pipeline\n'
        '\n'
        'Licences:\n'
        '- The SHiVAi segmentation pipeline (source code and Apptainer image) is distributed under'
        'the GNU General Public License - version 3 (GPLv3):\n'
        'https://www.gnu.org/licenses/gpl-3.0.fr.html#license-text \n'
        '- The SHiVAi trained AI models are distributed under and protected by the CC BY-NC-SA 4.0 license:\n'
        'https://creativecommons.org/licenses/by-nc-sa/4.0/ \n'
        '\n'
        "The segmentation models have been registered at the french 'Association de Protection des Programmes' under the numbers:\n"
        '- For PVS: IDDN.FR.001.240015.000.S.P.2022.000.31230\n'
        '- For WMH: IDDN.FR.001.240030.000.S.P.2022.000.31230\n'
        '- For CMB: IDDN.FR.001.420007.000.S.P.2023.000.31230\n'
        '\n'
    )
    parser = shivaParser()
    args = set_args_and_check(parser)
    shiva(**vars(args))


if __name__ == "__main__":
    main()
