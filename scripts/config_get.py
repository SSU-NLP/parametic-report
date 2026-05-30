#!/usr/bin/env python3
import argparse
import json
import os


def get_value(data, key):
    value = data
    for part in key.split("."):
        value = value[part]
    return value


def main():
    parser = argparse.ArgumentParser(description="Read a dotted key from config.json.")
    parser.add_argument("config_path")
    parser.add_argument("key")
    parser.add_argument("--join", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--path-root", default=None)
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        value = get_value(json.load(f), args.key)

    if args.path_root is not None and isinstance(value, str) and not os.path.isabs(value):
        value = os.path.abspath(os.path.join(args.path_root, value))

    if args.json:
        print(json.dumps(value))
    elif args.join is not None and isinstance(value, list):
        print(args.join.join(str(item) for item in value))
    elif isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)


if __name__ == "__main__":
    main()
