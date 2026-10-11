# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   python3 ci/measure_summary.py <results> <combined.tsv>
#
# Gathers what ci/measure.sh wrote on each runner -- one directory of <results>
# a runner, with its runner.txt, tests.tsv and bench.tsv -- by the processor
# the runner had, since GitHub's runners are of a few kinds of processor and a
# run gets whichever it gets. Prints, in Markdown, for the summary of the job:
#
#  - the processors, with how many runners had each, the levels of the kernels
#    each has, and the tests that failed on it, if any;
#  - for each processor, the most FMAs it makes at each width (sdasm_bench
#    --peak), and the rate of each operation in each form, the median over its
#    runners (a runner shares its machine with others, and a median is what
#    one of them is not much moved by); how many times the rate of one form
#    another's is: each level over the one below, the kernels written by hand
#    over the C++ ones, and over ggml's; and the share of the peak of its width
#    that the kernels for AVX2 and for AVX-512 reach, and how many times that
#    of AVX-512 the kernels on AMX's tiles reach.
#
# and writes every rate of every runner to <combined.tsv>, with its processor.

import collections
import pathlib
import statistics
import sys

LEVELS = ["avx2", "avx512", "amx"]


def read_runner(directory):
    info = {"model": "?", "levels": [], "flags": ""}
    text = directory / "runner.txt"
    if text.exists():
        for line in text.read_text().splitlines():
            key, _, value = line.partition(":")
            value = value.strip()
            if key == "model":
                info["model"] = value
            elif key == "flags":
                info["flags"] = value
            elif key == "levels":
                info["levels"] = value.split()
    tests = {}
    if (directory / "tests.tsv").exists():
        for line in (directory / "tests.tsv").read_text().splitlines():
            op, _, result = line.partition("\t")
            tests[op] = result
    rates = []
    if (directory / "bench.tsv").exists():
        for line in (directory / "bench.tsv").read_text().splitlines():
            fields = line.split("\t")
            if len(fields) == 6:
                rates.append({"case": fields[0], "form": fields[1], "level": fields[2], "threads": int(fields[3]),
                              "gflops": float(fields[4]), "seconds": float(fields[5])})
    return info, tests, rates


def ratio(a, b):
    return f"{a / b:.2f}" if a and b else ""


def main():
    results = pathlib.Path(sys.argv[1])
    combined = pathlib.Path(sys.argv[2])
    runners = collections.defaultdict(list)
    with combined.open("w") as out:
        out.write("runner\tmodel\tthreads\tcase\tform\tlevel\tgflops\tseconds\n")
        for directory in sorted(p for p in results.iterdir() if p.is_dir()):
            info, tests, rates = read_runner(directory)
            runners[info["model"]].append((directory.name, info, tests, rates))
            for r in rates:
                out.write(f"{directory.name}\t{info['model']}\t{r['threads']}\t{r['case']}\t{r['form']}\t{r['level']}"
                          f"\t{r['gflops']:.2f}\t{r['seconds']:.6f}\n")

    print("## The processors of the runners\n")
    print("| Processor | Runners | Threads | Levels | Tests failed |")
    print("| --- | ---: | ---: | --- | --- |")
    for model, entries in sorted(runners.items(), key=lambda e: -len(e[1])):
        threads = sorted({r["threads"] for _, _, _, rates in entries for r in rates})
        levels = sorted({tuple(info["levels"]) for _, info, _, _ in entries})
        failed = sorted({op for _, _, tests, _ in entries for op, result in tests.items() if result != "ok"})
        print(f"| {model} | {len(entries)} | {', '.join(map(str, threads))} | "
              f"{'; '.join(' '.join(l) for l in levels)} | {', '.join(failed) if failed else 'none'} |")
    print()

    for model, entries in sorted(runners.items(), key=lambda e: -len(e[1])):
        by_case = collections.defaultdict(lambda: collections.defaultdict(list))
        cases = []
        forms = []
        peaks = collections.defaultdict(list)
        for _, _, _, rates in entries:
            for r in rates:
                if r["case"] == "fma peak":
                    peaks[r["form"]].append(r["gflops"])
                    continue
                if r["case"] not in cases:
                    cases.append(r["case"])
                if r["form"] not in forms:
                    forms.append(r["form"])
                by_case[r["case"]][r["form"]].append(r["gflops"])
        if not cases:
            continue
        # How many times one form's rate another's: each level over the one
        # below it, the best of those written by hand over the C++ kernels and
        # over ggml's.
        pairs = [(LEVELS[i], LEVELS[i - 1]) for i in range(1, len(LEVELS))
                 if LEVELS[i] in forms and LEVELS[i - 1] in forms]
        hand = [f for f in ["assembly"] + LEVELS[::-1] if f in forms]
        if hand and "cxx" in forms:
            pairs.append((hand[0], "cxx"))
        if hand and "upstream" in forms:
            pairs.append((hand[0], "upstream"))
        peak = {width: statistics.median(values) for width, values in peaks.items()}
        # The share of a width's peak: the kernels for it; AMX's, as a multiple
        # of AVX-512's peak.
        shares = [(f, f if f != "amx" else "avx512") for f in ["avx2", "avx512", "amx"] if f in forms]
        shares = [(f, w) for f, w in shares if w in peak]
        print(f"## {model} ({len(entries)} runners)\n")
        if peak:
            print("The most FMAs, GFLOP/s: " + ", ".join(f"{w} {peak[w]:.1f}" for w in sorted(peak)) + ".\n")
        print("GFLOP/s, the median over the runners; how many times one form's rate the other's; "
              "and the rate over the FMA peak of the width.\n")
        header = (["Operation"] + forms + [f"{a} / {b}" for a, b in pairs] +
                  [f"{f} / peak {w}" for f, w in shares])
        print("| " + " | ".join(header) + " |")
        print("| --- |" + " ---: |" * (len(header) - 1))
        for case in cases:
            medians = {f: statistics.median(by_case[case][f]) for f in forms if by_case[case][f]}
            cells = [case] + [f"{medians[f]:.1f}" if f in medians else "" for f in forms]
            cells += [ratio(medians.get(a), medians.get(b)) for a, b in pairs]
            cells += [ratio(medians.get(f), peak.get(w)) for f, w in shares]
            print("| " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
