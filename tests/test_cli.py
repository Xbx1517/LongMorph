"""Public command routing and package structure contracts."""

import contextlib
import io
import sys
import unittest
from longmorph.cli import COMMANDS, main


class CommandLineTests(unittest.TestCase):
    def test_public_command_help(self):
        for command in COMMANDS:
            with self.subTest(command=command), contextlib.redirect_stdout(io.StringIO()):
                previous = sys.argv[:]
                with self.assertRaises(SystemExit) as status:
                    main([*command, "--help"])
                self.assertEqual(status.exception.code, 0)
                self.assertEqual(sys.argv, previous)

    def test_root_and_group_help(self):
        for args in ([], ["--help"], ["prepare", "--help"], ["train", "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(args)
            self.assertIn(
                "prepare" if not args or args[0] == "--help" else args[0], output.getvalue()
            )

    def test_unknown_command_fails(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as status:
            main(["unknown"])
        self.assertEqual(status.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
