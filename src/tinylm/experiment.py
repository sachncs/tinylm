"""TOML-configured experiments: parse, override, queue and run."""

import dataclasses
import hashlib
import json
import math
import pathlib
import tomllib
import typing

from tinylm import model, seeding, tokenizer
from tinylm.atomic import durable
from tinylm.bench import Bench
from tinylm.bootstrap import Interval
from tinylm.offline import Offline
from tinylm.provenance import Provenance
from tinylm.tokens import Token
from tinylm.train import Train

TOP_LEVEL = ("name", "data", "out", "evals", "taus", "model", "train")
SECTIONS = ("model", "train")
VOCAB = 8192
ROOT = pathlib.Path(__file__).resolve().parents[2]


class Error(Exception):
    """Raised when an experiment, override or queue file is invalid."""


if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance


def build[T: DataclassInstance](
    kind: type[T], values: object, section: str
) -> T:
    """Builds a dataclass from a TOML table, rejecting unknown keys.

    Raises:
        Error: If values is not a table, has unknown or missing keys, or
            fails the dataclass validation.
    """
    if not isinstance(values, dict):
        raise Error(f"missing [{section}] section")
    table = typing.cast("dict[str, object]", values)
    names = {f.name for f in dataclasses.fields(kind)}
    unknown = set(table) - names
    if unknown:
        raise Error(f"unknown keys in [{section}]: {sorted(unknown)}")
    try:
        return kind(**table)
    except (TypeError, ValueError) as error:
        raise Error(f"[{section}] {error}") from error


def coerce(field: dataclasses.Field[object], text: str) -> object:
    """Parses text as the type of a dataclass field.

    Raises:
        ValueError: If text is not a valid value of that type.
    """
    kind = field.type
    if typing.get_origin(kind) is tuple:
        return tuple(int(part) for part in text.split(",") if part)
    if kind is bool:
        if text not in ("true", "false"):
            raise ValueError(f"expected true or false, got {text!r}")
        return text == "true"
    if kind is int:
        return int(text)
    if kind is float:
        return float(text)
    return text


def change[T: DataclassInstance](target: T, field: str, text: str) -> T:
    """Returns target with one field replaced by the parsed text.

    Raises:
        Error: If the field does not exist or the text is not a valid value.
    """
    fields = {f.name: f for f in dataclasses.fields(target)}
    if field not in fields:
        raise Error(f"unknown field {field!r}")
    try:
        return dataclasses.replace(
            target, **{field: coerce(fields[field], text)}
        )
    except (ValueError, TypeError) as error:
        raise Error(f"{field}: {error}") from error


@dataclasses.dataclass(frozen=True, slots=True)
class Summary:
    """Precision and coverage of confident answers on one benchmark."""

    name: str
    tau: float
    precision: float | None
    coverage: float


@dataclasses.dataclass(frozen=True, slots=True)
class Metrics:
    """Results of a run, written to metrics.json.

    Attributes:
        parameters: Trainable parameters.
        dev_loss: Dev cross-entropy after each epoch.
        epochs_used: Epochs consumed according to the ledger.
        tokens_per_second: Training throughput.
        train_tokens: Tokens in the training stream.
        benchmarks: Mean paradigm accuracy per evaluation folder.
        intervals: 95% bootstrap interval of each benchmark mean.
        bits_per_char: Final dev bits per character.
        confidence: Confident-correct summaries per folder and threshold.
        seconds: Device-synchronised training time (evaluation excluded).
        provenance: Code revision, environment and input fingerprints.
    """

    parameters: int
    dev_loss: list[float]
    epochs_used: float
    tokens_per_second: float
    train_tokens: int
    benchmarks: dict[str, float]
    intervals: dict[str, Interval]
    bits_per_char: float
    confidence: tuple[Summary, ...]
    seconds: float
    provenance: Provenance


@dataclasses.dataclass(frozen=True, slots=True)
class Experiment:
    """A complete, validated run description.

    Attributes:
        name: Run name.
        data: Folder written by Prepare.
        out: Folder for checkpoint and metrics.
        model: Model configuration (vocab is taken from the data).
        train: Training configuration.
        evals: Minimal-pair folders scored after training.
        taus: Margin thresholds for confident-correct summaries.
    """

    name: str
    data: pathlib.Path
    out: pathlib.Path
    model: model.Config
    train: Train
    evals: tuple[pathlib.Path, ...] = ()
    taus: tuple[float, ...] = ()

    @classmethod
    def load(cls, path: pathlib.Path, vocab: int = VOCAB) -> Experiment:
        """Loads and validates an experiment file.

        Args:
            path: TOML file.
            vocab: Vocabulary used for the parameter-budget check.

        Raises:
            FileNotFoundError: If path does not exist.
            Error: If the file is invalid or the model exceeds the budget.
        """
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            raw = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise Error(f"invalid toml: {error}") from error
        unknown = set(raw) - set(TOP_LEVEL)
        if unknown:
            raise Error(f"unknown top-level keys: {sorted(unknown)}")
        try:
            name, data, out = raw["name"], raw["data"], raw["out"]
        except KeyError as error:
            raise Error(f"missing key {error}") from error
        if not isinstance(raw.get("model"), dict):
            raise Error("missing [model] section")
        values = dict(raw["model"])
        values.setdefault("vocab_size", vocab)
        loaded = cls(
            name=name,
            data=pathlib.Path(data),
            out=pathlib.Path(out),
            model=build(model.Config, values, "model"),
            train=build(Train, raw.get("train"), "train"),
            evals=tuple(pathlib.Path(p) for p in raw.get("evals", [])),
            taus=tuple(float(t) for t in raw.get("taus", [])),
        )
        loaded.check(vocab)
        return loaded

    def fingerprint(self) -> str:
        """Hash of everything that defines this experiment (not its name)."""
        settings = dataclasses.asdict(self)
        for key in ("name", "out"):
            del settings[key]
        text = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()

    def check(self, vocab: int) -> None:
        """Raises Error if the model exceeds the parameter budget."""
        sized = dataclasses.replace(self.model, vocab_size=vocab)
        try:
            model.check(model.TinyLM(sized))
        except ValueError as error:
            raise Error(str(error)) from error

    def override(
        self, assignments: list[str], name: str, vocab: int = VOCAB
    ) -> Experiment:
        """Returns a copy with "section.field=value" overrides applied.

        The copy gets the given name and an output folder next to the
        original. "data=<path>" is the only top-level key that can change.

        Raises:
            Error: If an assignment is malformed, names an unknown field,
                has a wrong-typed value, fails validation or pushes the
                model over the parameter budget.
        """
        net, train = self.model, self.train
        data = self.data
        for assignment in assignments:
            key, separator, text = assignment.partition("=")
            if key == "data" and separator:
                data = pathlib.Path(text)
                continue
            section, dot, field = key.partition(".")
            if not separator or not dot or section not in SECTIONS:
                raise Error(f"bad override {assignment!r}")
            try:
                if section == "model":
                    net = change(net, field, text)
                else:
                    train = change(train, field, text)
            except Error as error:
                raise Error(f"{key}: {error}") from error
        updated = dataclasses.replace(
            self,
            name=name,
            data=data,
            out=self.out.parent / name,
            model=net,
            train=train,
        )
        updated.check(vocab)
        return updated

    def run(self) -> Metrics:
        """Trains from the prepared data with no network and writes metrics.

        Raises:
            FileNotFoundError: If the prepared data or an evaluation folder
                is missing (checked before any training).
        """
        for folder in self.evals:
            if not folder.is_dir():
                raise FileNotFoundError(folder)
        with Offline():
            return self.execute()

    def execute(self) -> Metrics:
        """Does the work of run() once the inputs are known to exist."""
        tok = tokenizer.load(self.data / "tokenizer.json")
        train = Token.load(self.data / "train")
        dev = Token.load(self.data / "dev")
        stats = json.loads((self.data / "stats.json").read_text())
        net_config = dataclasses.replace(self.model, vocab_size=train.vocab)
        seeding.set_seed(self.train.seed)
        net = model.TinyLM(net_config)
        model.check(net)
        losses: list[float] = []

        def after(epoch: int) -> None:
            losses.append(self.train.evaluate(net, dev))
            print(f"epoch {epoch} dev_loss {losses[-1]:.4f}", flush=True)
            net.train()

        self.out.mkdir(parents=True, exist_ok=True)
        result = self.train.run(net, train, self.out, after=after)
        bench = Bench(net, tok, self.train.device)
        benchmarks: dict[str, float] = {}
        intervals: dict[str, Interval] = {}
        summaries: list[Summary] = []
        for folder in self.evals:
            report = bench.evaluate(folder, self.taus)
            benchmarks[folder.name] = report.mean
            intervals[folder.name] = report.interval
            text = json.dumps(report.outcomes, sort_keys=True)
            durable(
                self.out / f"outcomes-{folder.name}.json",
                lambda h, t=text: h.write(t.encode()),
            )
            summaries.extend(
                Summary(folder.name, c.tau, c.precision, c.coverage)
                for c in report.confidence.values()
            )
        trace = Provenance.capture(
            ROOT, self.train.device, self.fingerprint(), train.fingerprint()
        )
        metrics = Metrics(
            parameters=model.count(net),
            dev_loss=losses,
            epochs_used=result.budget.epochs_used,
            tokens_per_second=result.budget.consumed / result.seconds,
            train_tokens=len(train),
            benchmarks=benchmarks,
            intervals=intervals,
            bits_per_char=losses[-1]
            * len(dev)
            / math.log(2)
            / stats["dev_chars"],
            confidence=tuple(summaries),
            seconds=result.seconds,
            provenance=trace,
        )
        text = json.dumps(dataclasses.asdict(metrics), indent=2) + "\n"
        durable(self.out / "metrics.json", lambda h: h.write(text.encode()))
        return metrics


@dataclasses.dataclass(frozen=True, slots=True)
class Item:
    """One queue entry: a run name and its overrides."""

    name: str
    sets: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Queue:
    """A base experiment file plus variants to run in order."""

    base: pathlib.Path
    runs: tuple[Item, ...]

    @classmethod
    def load(cls, path: pathlib.Path) -> Queue:
        """Reads a queue file.

        Raises:
            FileNotFoundError: If path does not exist.
            Error: If the file is malformed, a run has no name, names repeat,
                or there is no base or no run.
        """
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            raw = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise Error(f"invalid toml: {error}") from error
        if "base" not in raw:
            raise Error("queue lacks base")
        entries = raw.get("run")
        if not entries:
            raise Error("queue has no [[run]] entries")
        runs: list[Item] = []
        for entry in entries:
            if "name" not in entry:
                raise Error("run lacks name")
            runs.append(Item(entry["name"], tuple(entry.get("set", []))))
        names = [r.name for r in runs]
        if len(set(names)) != len(names):
            raise Error("duplicate run names")
        return cls(pathlib.Path(raw["base"]), tuple(runs))

    def pending(self, runs: pathlib.Path) -> list[Item]:
        """Entries that have no metrics.json under runs yet."""
        return [
            r
            for r in self.runs
            if not (runs / r.name / "metrics.json").exists()
        ]
