"""Re-score saved benchmark responses after a parser-contract correction.

This module reads only JSONL response members from an experiment ZIP. It never
loads model weights and never overwrites the original archive or its reports.
The resulting report is explicitly marked post-hoc because changing a parser
after observing outputs is not the same as a preregistered benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from sycophancy_rl.evaluation.compare_runs import compare_runs
from sycophancy_rl.evaluation.io import write_json, write_records
from sycophancy_rl.evaluation.metrics import summarize_records
from sycophancy_rl.reward.reward_fn import get_reward_config, score_completion
from sycophancy_rl.utils.answer_parser import classify_answer, parse_final_answer

PARSER_REVISION = "leading-parenthesized-option-v1"
MAX_RESPONSE_MEMBER_BYTES = 50 * 1024 * 1024


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_member(archive: zipfile.ZipFile, phase: str) -> zipfile.ZipInfo:
    suffix = f"/{phase}/responses.jsonl"
    matches = [info for info in archive.infolist() if info.filename.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {phase!r} responses member ending in "
            f"{suffix!r}; found {len(matches)}."
        )
    member = matches[0]
    if member.file_size > MAX_RESPONSE_MEMBER_BYTES:
        raise ValueError(
            f"Refusing to read {member.filename!r}: uncompressed size "
            f"{member.file_size} exceeds {MAX_RESPONSE_MEMBER_BYTES} bytes."
        )
    return member


def _read_jsonl_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with archive.open(member, "r") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{member.filename}:{line_number}: invalid UTF-8 JSON object."
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{member.filename}:{line_number}: expected a JSON object."
                )
            records.append(value)
    if not records:
        raise ValueError(f"{member.filename} contains no response records.")
    return records


def rescore_record(
    record: dict[str, Any], *, reward_profile: str = "combined"
) -> dict[str, Any]:
    """Return a re-scored copy of one saved response record."""

    revised = dict(record)
    parsed = parse_final_answer(
        record.get("generated_response"),
        finish_reason=record.get("finish_reason"),
    )
    category = classify_answer(
        parsed,
        independent_option=str(record["independent_option"]),
        sycophantic_option=record.get("sycophantic_option"),
    )
    breakdown = score_completion(
        record.get("generated_response"),
        target_option=str(record["target_option"]),
        independent_option=str(record["independent_option"]),
        sycophantic_option=record.get("sycophantic_option"),
        user_preferred_option=record.get("user_preferred_option"),
        user_claim_valid=record.get("user_claim_valid"),
        behavior_target=str(
            record.get("behavior_target", "independent_reasoning")
        ),
        prompt=record.get("prompt", ""),
        finish_reason=record.get("finish_reason"),
        config=get_reward_config(reward_profile),
    )
    revised.update(
        {
            "parsed_label": parsed.label,
            "parse_status": parsed.reason,
            "category": category,
            "target_selected": parsed.valid
            and parsed.label == record.get("target_option"),
            "format_compliant": parsed.format_compliant,
            "contradictory": parsed.contradictory,
            "truncated": parsed.truncated,
            "reward": breakdown.total,
            "reward_breakdown": breakdown.to_dict(),
            "rescoring": {
                "parser_revision": PARSER_REVISION,
                "post_hoc": True,
                "reward_profile": reward_profile,
            },
        }
    )
    return revised


def rescore_archive(
    archive_path: str | Path,
    output_directory: str | Path,
    *,
    reward_profile: str = "combined",
) -> dict[str, Any]:
    """Re-score BEFORE/AFTER JSONL members and write separate derived files."""

    source = Path(archive_path).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not zipfile.is_zipfile(source):
        raise ValueError(f"{source} is not a valid ZIP archive.")

    with zipfile.ZipFile(source) as archive:
        original_before = _read_jsonl_member(
            archive, _response_member(archive, "before")
        )
        original_after = _read_jsonl_member(
            archive, _response_member(archive, "after")
        )

    revised_before = [
        rescore_record(record, reward_profile=reward_profile)
        for record in original_before
    ]
    revised_after = [
        rescore_record(record, reward_profile=reward_profile)
        for record in original_after
    ]
    revised_comparison = compare_runs(revised_before, revised_after)
    report: dict[str, Any] = {
        "post_hoc": True,
        "publishable_as_original_preregistered_result": False,
        "parser_revision": PARSER_REVISION,
        "source_archive": str(source),
        "source_archive_sha256": _archive_sha256(source),
        "reward_profile": reward_profile,
        "original": {
            "before": summarize_records(original_before),
            "after": summarize_records(original_after),
        },
        "revised": {
            "before": summarize_records(revised_before),
            "after": summarize_records(revised_after),
            "comparison": revised_comparison,
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    write_records(destination / "before.responses.rescored.jsonl", revised_before)
    write_records(destination / "after.responses.rescored.jsonl", revised_after)
    write_json(destination / "rescore_report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-profile", default="combined")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = rescore_archive(
        args.archive,
        args.output,
        reward_profile=args.reward_profile,
    )
    before = report["revised"]["before"]
    after = report["revised"]["after"]
    print(
        "Re-scored archive post-hoc: "
        f"BEFORE valid={1 - before['invalid_answer_rate']['rate']:.1%}, "
        f"AFTER valid={1 - after['invalid_answer_rate']['rate']:.1%}."
    )
    print(f"Wrote derived results to {Path(args.output).resolve()}.")


if __name__ == "__main__":
    main()
