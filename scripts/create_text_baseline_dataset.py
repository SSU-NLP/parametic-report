#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

TOPICS = [
    "A city council reviewed a proposal for safer crosswalks near schools. Members compared maintenance costs, resident feedback, and the expected effect on morning traffic before scheduling a follow-up vote.",
    "The research team prepared a short memo summarizing survey results. They highlighted response rates, common concerns, and the limits of drawing conclusions from a small sample.",
    "A museum curator planned a new exhibition about coastal trade routes. The notes described artifact labels, visitor flow, lighting constraints, and how guides should introduce the historical context.",
    "The operations manager wrote an incident review after a delayed shipment. The review separated confirmed facts from assumptions and listed actions for improving vendor communication.",
    "A teacher designed a lesson about climate patterns. Students compared temperature records, rainfall charts, and local observations before discussing why averages can hide extreme events.",
    "The finance committee discussed next quarter's budget. The conversation focused on hiring needs, equipment replacement, travel limits, and the risk of postponing routine maintenance.",
    "A hospital administrator drafted an update for staff. The update explained schedule changes, patient intake procedures, and where employees should report supply shortages.",
    "The editorial team reviewed a long feature article. They checked whether each section supported the main argument and whether the examples were clear to readers without specialist knowledge.",
    "A product support lead summarized recent customer calls. The summary grouped requests by urgency, identified recurring confusion, and suggested changes to the help center structure.",
    "The planning office evaluated proposals for a public park. Reviewers considered shade, accessibility, drainage, noise, and how the space would serve families during weekends.",
    "A conference organizer prepared instructions for session chairs. The note covered timing, speaker introductions, audience questions, and what to do if a room change becomes necessary.",
    "The laboratory safety officer updated training material. The update emphasized labeling, storage practices, protective equipment, and the importance of reporting near misses promptly.",
    "A neighborhood association collected comments about evening transit service. Residents described long waits, crowded stops, and the tradeoff between express routes and wider coverage.",
    "The design team critiqued a prototype dashboard. They discussed visual hierarchy, terminology, empty states, and whether important warnings were visible without creating alarm fatigue.",
    "A library director outlined a plan for community workshops. The plan included staffing, registration limits, accessibility needs, and ways to measure whether participants found the sessions useful.",
    "The legal department circulated guidance about contract reviews. The guidance asked teams to document deadlines, unusual obligations, renewal terms, and any promises made during negotiation.",
    "A transportation analyst compared several routes for a new bus line. The analysis weighed ridership estimates, transfer convenience, walking distance, and operating cost.",
    "The human resources team prepared onboarding notes. The notes explained benefits enrollment, equipment pickup, required training, and how new employees can find internal policies.",
    "A restaurant manager reviewed feedback from diners. The comments mentioned wait times, menu clarity, portion size, and whether staff explained allergy options consistently.",
    "The facilities team planned repairs after a water leak. They listed affected rooms, inspection steps, temporary closures, and how updates would be shared with building occupants.",
]

def make_record(index: int, rng: random.Random) -> dict:
    first = TOPICS[index % len(TOPICS)]
    second = TOPICS[(index * 7 + 3) % len(TOPICS)]
    third = TOPICS[(index * 11 + 5) % len(TOPICS)]
    details = [first, second, third]
    rng.shuffle(details)
    content = " ".join(details)
    return {"content": content}

def write_split(path: Path, count: int, offset: int, rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            record = make_record(offset + i, rng)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic natural-language JSONL baseline dataset.")
    parser.add_argument("--output", type=Path, default=Path("data_preprocess/dataset/baseline-text"))
    parser.add_argument("--language", default="text")
    parser.add_argument("--train_examples", type=int, default=31)
    parser.add_argument("--test_examples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    write_split(args.output / "train" / f"{args.language}.jsonl", args.train_examples, 0, rng)
    write_split(args.output / "test" / f"{args.language}.jsonl", args.test_examples, args.train_examples, rng)
    print(f"wrote {args.train_examples} train and {args.test_examples} test records under {args.output}")

if __name__ == "__main__":
    main()
