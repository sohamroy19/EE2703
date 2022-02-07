"""
    EE2703: Applied Programming Lab - 2022
        Assignment 2: Spice - Part 2
            Soham Roy - EE20B130
"""

import sys  # for command line arguments and exiting
import re  # regular expressions for parsing the value
import numpy as np  # for matrix mathematics for MNA
import cmath, math  # complex mathematics for AC analysis

# check if the user has given the correct number of arguments
if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print(f"Usage: {sys.argv[0]} <input_file> [ground_node] [-d | --debug]")
    sys.exit()

# constants that help read the file
CIRCUIT = ".circuit"
END = ".end"
AC = ".ac"
DEBUG = sys.argv[-1] == "-d" or sys.argv[-1] == "--debug"
GND = "GND" if len(sys.argv) - DEBUG == 2 else sys.argv[2]


def parse_value(value):
    """Parse the string to find the value, supports LTSpice syntax for suffixes."""

    split = re.split("([a-z]+)", value.lower())  # regex magic

    if len(split) == 1 or split[1] == "e":  # no suffix or exponent notation
        return float(value)

    suffix = split[1].lower()
    if suffix.startswith("k"):  # kilo
        return float(split[0]) * 1e3
    elif suffix.startswith("meg"):  # mega
        return float(split[0]) * 1e6
    elif suffix.startswith("g"):  # giga
        return float(split[0]) * 1e9
    elif suffix.startswith("t"):  # tera
        return float(split[0]) * 1e12
    elif suffix.startswith("f"):  # femto
        return float(split[0]) * 1e-15
    elif suffix.startswith("p"):  # pico
        return float(split[0]) * 1e-12
    elif suffix.startswith("n"):  # nano
        return float(split[0]) * 1e-9
    elif suffix.startswith("u") or suffix.startswith("micro"):
        return float(split[0]) * 1e-6
    elif suffix.startswith("m"):  # milli
        return float(split[0]) * 1e-3


class Element:
    """Represents any element of the circuit block."""

    def __init__(self, words):
        self.name = words[0]
        self.type = self.name[0]

        self.n1 = words[1]
        self.n2 = words[2]
        self.value = parse_value(words[-1])

        if self.type == "E" or self.type == "G":  # for voltage controlled sources
            self.n3 = words[3]
            self.n4 = words[4]
        elif self.type == "F" or self.type == "H":  # for current controlled sources
            self.name2 = words[3]
        elif self.type == "V" or self.type == "I":  # for sources
            if len(words) == 6:  # V... n1 n2 ac val phase
                # for AC sources convert (r ∠φ) to (a + bj), Vp-p / 2 = r
                self.value = cmath.rect(
                    parse_value(words[-2]) / 2, math.radians(self.value)
                )

        if self.n1 not in node_idx:  # if a node is not in node_idx
            node_idx[self.n1] = len(node_idx)
        if self.n2 not in node_idx:  # then add it to node_idx
            node_idx[self.n2] = len(node_idx)

        assert self.value is not None

    def __lt__(self, other):  # for sorting
        return self.name < other.name


ac_flag = False

elements = []  # to store all the elements of the circuit
node_idx = {}  # to map the name of each node to its row in the matrix
vsrc_idx = {}  # to map the name of volt sources to rows in the matrix

try:  # try to read the file
    with open(sys.argv[1]) as file:
        started = finished = False
        for line in file:
            tokens = line.partition("#")[0].split()  # ignore comments & split the line
            if len(tokens) > 0:
                if started:
                    if tokens[0] == END:
                        finished = True
                    elif not finished:
                        elements.append(Element(tokens))
                        if elements[-1].type in ["V", "E", "H"]:  # if a voltage source
                            vsrc_idx[elements[-1].name] = len(vsrc_idx)  # assign a row
                    elif tokens[0] == AC:  # assume single frequency
                        ac_freq = 2 * math.pi * parse_value(tokens[2])
                        ac_flag = True
                elif tokens[0] == CIRCUIT:
                    started = True

    if not finished:
        print("Invalid circuit definition")
        sys.exit()

except (ValueError, TypeError, AssertionError):
    sys.exit("Invalid value in the file")

except IOError:
    sys.exit("Invalid file")

for vsrc in vsrc_idx:
    vsrc_idx[vsrc] += len(node_idx) + 1  # + 1 for ground voltage = 0 equation

dim = len(node_idx) + len(vsrc_idx) + 1
b = np.zeros((dim, 1), dtype=complex if ac_flag else float)
M = np.zeros((dim, dim), dtype=complex if ac_flag else float)

try:
    M[len(node_idx)][node_idx[GND]] = M[node_idx[GND]][len(node_idx)] = 1
except KeyError:
    sys.exit(
        f"Ground node {GND} not found, specify correct reference node\n" +
        f"Usage: {sys.argv[0]} <input_file> [ground_node] [-d | --debug]"
    )

for elem in elements:  # add respective MNA stamps to conductance matrix M (G)
    if elem.type in ["R", "L", "C"]:  # for resistors, inductors, capacitors
        if elem.type == "R":
            admittance = 1 / elem.value
        elif elem.type == "L":
            admittance = 1 / (ac_freq * elem.value * 1j)
        elif elem.type == "C":
            admittance = ac_freq * elem.value * 1j
        M[node_idx[elem.n1]][node_idx[elem.n1]] += admittance
        M[node_idx[elem.n1]][node_idx[elem.n2]] -= admittance
        M[node_idx[elem.n2]][node_idx[elem.n1]] -= admittance
        M[node_idx[elem.n2]][node_idx[elem.n2]] += admittance
    elif elem.type == "I":  # current source
        b[node_idx[elem.n1]] -= elem.value
        b[node_idx[elem.n2]] += elem.value
    elif elem.type == "G":  # voltage controlled current source
        M[node_idx[elem.n1]][node_idx[elem.n3]] += elem.value
        M[node_idx[elem.n1]][node_idx[elem.n4]] -= elem.value
        M[node_idx[elem.n2]][node_idx[elem.n3]] -= elem.value
        M[node_idx[elem.n2]][node_idx[elem.n4]] += elem.value
    elif elem.type == "F":  # current controlled current source
        M[node_idx[elem.n1]][vsrc_idx[elem.name2]] += elem.value
        M[node_idx[elem.n2]][vsrc_idx[elem.name2]] -= elem.value
    if elem.type in ["V", "E", "H"]:  # voltage sources
        M[node_idx[elem.n1]][vsrc_idx[elem.name]] += 1
        M[node_idx[elem.n2]][vsrc_idx[elem.name]] -= 1
        M[vsrc_idx[elem.name]][node_idx[elem.n1]] += 1
        M[vsrc_idx[elem.name]][node_idx[elem.n2]] -= 1
        if elem.type == "V":  # voltage source
            b[vsrc_idx[elem.name]] += elem.value
        elif elem.type == "E":  # voltage controlled voltage source
            M[vsrc_idx[elem.name]][node_idx[elem.n3]] -= elem.value
            M[vsrc_idx[elem.name]][node_idx[elem.n4]] += elem.value
        elif elem.type == "H":  # current controlled voltage source
            M[vsrc_idx[elem.name]][vsrc_idx[elem.name2]] -= elem.value

if DEBUG:
    print("Matrix M (G):\n", M, "\n\nVector b (I):\n", b, "\n")

try:
    x = np.linalg.solve(M, b)
    if DEBUG:
        print("Matrix x:\n", x, "\n")
except np.linalg.LinAlgError:
    sys.exit("Circuit unsolvable (singular matrix)")

for node, idx in sorted(node_idx.items()):
    print(f"Voltage at {node:4}   :   ", end="")
    if ac_flag:
        z = cmath.polar(x[idx][0])
        print(f"{z[0]: .4e}   ∠{math.degrees(z[1]):>7.2f}°   V")
    else:
        print(f"{x[idx][0]: .4e} V")

for vsrc, idx in sorted(vsrc_idx.items()):
    print(f"Current in {vsrc:4}   :   ", end="")
    if ac_flag:
        z = cmath.polar(x[idx][0])
        print(f"{z[0]: .4e}   ∠{math.degrees(z[1]):>7.2f}°   A")
    else:
        print(f"{x[idx][0]: .4e} A")
