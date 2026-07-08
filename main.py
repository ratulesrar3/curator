"""
curator — build long-form DJ sets from song feature analytics.

    python main.py --vibe "dark warehouse peak hour techno" --hours 4
    python main.py --vibe "sunrise melodic closing" --hours 2 --formats csv,html
    python main.py --agent --vibe "closing set at a Lisbon rooftop"
    python main.py --rebuild-library
    python main.py --rebuild-msd-library
    python main.py --library data/library_msd.csv --vibe "warehouse techno"
"""

import argparse
import sys
import time

from src import arcs, config, exporters, explain, library, report_html, viz
from src.sequencer import build_set

ALL_FORMATS = {"csv", "json", "md", "m3u8", "rekordbox", "png", "html"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sequence a DJ set from a vibe.")
    parser.add_argument("--vibe", default="", help='e.g. "dark warehouse peak hour techno"')
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--template", choices=sorted(arcs.TEMPLATES),
                        help="override the template chosen from the vibe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="output")
    parser.add_argument("--formats", default="csv,json,md,m3u8,rekordbox,png,html",
                        help=f"comma list of {sorted(ALL_FORMATS)}")
    parser.add_argument("--library", dest="library_path", metavar="PATH",
                        help="alternate library CSV (e.g. data/library_msd.csv); "
                             "adjacency is recomputed from it")
    parser.add_argument("--agent", action="store_true",
                        help="plan/critique the set with the Perplexity agent "
                             "(needs PERPLEXITY_API_KEY or secrets.md)")
    parser.add_argument("--no-search", action="store_true",
                        help="agent: skip the web-grounded vibe research step")
    parser.add_argument("--model", default=config.PPLX_MODEL,
                        help=f"agent: Perplexity model (default {config.PPLX_MODEL})")
    parser.add_argument("--rebuild-library", action="store_true")
    parser.add_argument("--rebuild-msd-library", action="store_true",
                        help="build data/library_msd.csv from data/msd/ "
                             "(see docs/msd_feasibility.md)")
    args = parser.parse_args(argv)

    if args.rebuild_library:
        lib = library.build()
        print(f"library: {len(lib)} tracks")
        print(lib["genre"].value_counts().to_string())
        print(f"bridges: {int(lib['is_bridge'].sum())}")
        print(library.load_adjacency().to_string())
        return 0

    if args.rebuild_msd_library:
        from src import msd_adapter
        lib = msd_adapter.build()
        print(f"MSD library: {len(lib)} tracks -> {msd_adapter.MSD_LIBRARY_PATH}")
        print(lib["genre"].value_counts().to_string())
        print(f"bridges: {int(lib['is_bridge'].sum())}")
        print(f"coverage report -> {msd_adapter.COVERAGE_PATH}")
        return 0

    formats = {f.strip() for f in args.formats.split(",")} & ALL_FORMATS
    if args.library_path:
        import pandas as pd
        lib = pd.read_csv(args.library_path)
        adjacency = library.adjacency_from(lib)
    else:
        lib = library.load()
        adjacency = library.load_adjacency()

    t0 = time.time()
    if args.agent:
        from src import agent
        result, notes = agent.run(
            vibe=args.vibe, hours=args.hours, lib=lib, adjacency=adjacency,
            seed=args.seed, pinned_template=args.template,
            search=not args.no_search, model=args.model,
        )
        if notes.get("fallback"):
            print(f"agent fallback: {notes['fallback']}")
        else:
            if notes.get("vibe_reading"):
                print(f"agent reading: {notes['vibe_reading']}")
            print(f"agent plan:    {notes['plan_rationale']} "
                  f"({notes['revisions']} revision(s), "
                  f"{len(notes['citations'])} source(s), model {notes['model']})")
    else:
        template, boosts = arcs.parse_vibe(args.vibe)
        if args.template:
            template = arcs.TEMPLATES[args.template]
        result = build_set(
            lib, adjacency, template, hours=args.hours,
            boosts=boosts, vibe=args.vibe, seed=args.seed,
        )
    elapsed = time.time() - t0

    print(explain.format_report(explain.set_report(result)))
    print(f"sequenced in {elapsed:.1f}s")

    written = exporters.export_all(
        result, args.out, formats,
        png_fn=viz.render_png, html_fn=report_html.render,
    )
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
