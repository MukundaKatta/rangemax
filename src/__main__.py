"""CLI for rangemax."""
import sys, json, argparse
from .core import Rangemax

def main():
    parser = argparse.ArgumentParser(description="RangeMax — EV Range Optimizer. Optimize EV driving range based on route, weather, and driving style.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Rangemax()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.process(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"rangemax v0.1.0 — RangeMax — EV Range Optimizer. Optimize EV driving range based on route, weather, and driving style.")

if __name__ == "__main__":
    main()
