#!/usr/bin/env python
"""main function running the workflow, calling the parser and the shiva function"""
from shivautils.utils.parsing import shivaParser, set_args_and_check
from shivautils.utils.shiva_runner import shiva


def main():

    parser = shivaParser()
    args = set_args_and_check(parser)
    shiva(**vars(args))


if __name__ == "__main__":
    main()
