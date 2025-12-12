# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

reset = "\033[0m"
bold = "\033[01m"
disable = "\033[02m"
underline = "\033[04m"
reverse = "\033[07m"
strikethrough = "\033[09m"
invisible = "\033[08m"
reset = "\033[0m"

up = "\033[A"
down = "\033[B"
right = "\033[C"
left = "\033[D"
home = "\033[H"
end = "\033[F"
clearToEnd = "\033[K"
clearToBegin = "\033[1K"
clearLine = "\033[2K"


def upN(n):
    return f"\033[{n}A"


def downN(n):
    return f"\033[{n}B"


def rightN(n):
    return f"\033[{n}C"


def leftN(n):
    return f"\033[{n}D"


fgBlack = "\033[30m"
fgRed = "\033[31m"
fgGreen = "\033[32m"
fgYellow = "\033[33m"
fgBlue = "\033[34m"
fgMagenta = "\033[35m"
fgCyan = "\033[36m"
fgGrey = "\033[37m"
# B = Bright
fgBGrey = "\033[90m"
fgBRed = "\033[91m"
fgBGreen = "\033[92m"
fgBYellow = "\033[93m"
fgBBlue = "\033[94m"
fgBMagenta = "\033[95m"
fgBCyan = "\033[96m"
fgWhite = "\033[97m"

bgBlack = "\033[40m"
bgRed = "\033[41m"
bgGreen = "\033[42m"
bgYellow = "\033[43m"
bgBlue = "\033[44m"
bgMagenta = "\033[45m"
bgCyan = "\033[46m"
bgGrey = "\033[47m"

bgBBlack = "\033[100m"
bgBRed = "\033[101m"
bgBGreen = "\033[102m"
bgBYellow = "\033[103m"
bgBBlue = "\033[104m"
bgBMagenta = "\033[105m"
bgBCyan = "\033[106m"
bgWhite = "\033[107m"

bgDarkgrey = "\033[48;5;235m"
