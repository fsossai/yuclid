"""Install Yuclid's agent skills from the Python package."""

from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import tempfile

from yuclid import __version__
from yuclid.log import LogLevel, report


AGENTS = {
    "codex": Path("~/.agents/skills"),
    "claude": Path("~/.claude/skills"),
}
MANIFEST = ".yuclid-skills.json"
SOURCE = Path(__file__).with_name("agent_skills")


def target_of(args):
    if args.agent is not None:
        target = AGENTS[args.agent]
    else:
        target = Path(args.directory)
    return target.expanduser().resolve()


def bundled_skills():
    return sorted(
        path for path in SOURCE.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )


def digest(directory):
    value = sha256()
    if not directory.is_dir():
        return None
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        value.update(path.relative_to(directory).as_posix().encode())
        value.update(b"\0")
        value.update(path.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def read_manifest(target):
    path = target / MANIFEST
    if not path.is_file():
        return {"skills": {}}
    try:
        contents = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        report(LogLevel.FATAL, "cannot read skill installation", path, str(error))
    if not isinstance(contents, dict) or not isinstance(contents.get("skills"), dict):
        report(LogLevel.FATAL, "invalid skill installation manifest", path)
    return contents


def write_manifest(target, skills):
    path = target / MANIFEST
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"yuclid": __version__, "skills": skills}, indent=2) + "\n"
    )
    os.replace(temporary, path)


def install(args):
    target = target_of(args)
    target.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(target)
    recorded = manifest["skills"]
    sources = bundled_skills()

    # Check everything before changing anything, so one conflict cannot leave
    # half of the bundle updated.
    for source in sources:
        destination = target / source.name
        if not destination.exists():
            continue
        managed = recorded.get(source.name)
        if managed is None or digest(destination) != managed:
            if not args.force:
                report(
                    LogLevel.FATAL,
                    "skill already exists and is not an unmodified Yuclid install",
                    destination,
                    hint="use --force to replace it",
                )

    installed = {}
    staging = Path(tempfile.mkdtemp(prefix=".yuclid-skills-", dir=target))
    try:
        for source in sources:
            destination = target / source.name
            fresh = staging / source.name
            shutil.copytree(source, fresh)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(fresh, destination)
            installed[source.name] = digest(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    write_manifest(target, installed)
    report(
        LogLevel.INFO,
        "installed {} skill(s)".format(len(installed)),
        target,
    )


def uninstall(args):
    target = target_of(args)
    manifest = read_manifest(target)
    recorded = manifest["skills"]
    if not recorded:
        report(LogLevel.FATAL, "no Yuclid skills are installed in", target)

    for name, expected in recorded.items():
        destination = target / name
        if destination.exists() and digest(destination) != expected and not args.force:
            report(
                LogLevel.FATAL,
                "installed skill has been modified",
                destination,
                hint="use --force to remove it anyway",
            )

    removed = 0
    for name in recorded:
        destination = target / name
        if destination.is_dir():
            shutil.rmtree(destination)
            removed += 1
    try:
        (target / MANIFEST).unlink()
    except FileNotFoundError:
        pass
    report(LogLevel.INFO, "uninstalled {} skill(s)".format(removed), target)


def launch(args):
    if args.skills_action == "install":
        install(args)
    else:
        uninstall(args)
