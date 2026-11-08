"""python submit.py phase1/configs/x.yaml   validate submissions/<ID_name>/prediction.json, write submission.zip + meta.json"""
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from config import args, path, plain, subdir, wrap
from data import filefp, loadtest, writejson


def dupcheck(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys in prediction.json")
    return dict(pairs)


def validate(pred, prompts, words=128):
    """Return a list of problems; empty means the prediction file is valid."""
    errs = []
    if not isinstance(pred, dict):
        return ["prediction must be a JSON object"]
    missing = set(prompts) - set(pred)
    extra = set(pred) - set(prompts)
    if missing:
        errs.append(f"{len(missing)} missing ids, e.g. {sorted(missing)[:5]}")
    if extra:
        errs.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:5]}")
    for i in sorted(set(pred) & set(prompts), key=str):
        rec = pred[i]
        if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
            errs.append(f"{i}: record must be {{'text': str}}")
            continue
        text = rec["text"]
        w = text.split()
        if not text.strip():
            errs.append(f"{i}: empty text")
        elif len(w) != words:
            errs.append(f"{i}: {len(w)} words (need {words})")
        p = prompts[i].split()
        head = " ".join(p[:16])
        if head and (" ".join(w[:16]) == head or " ".join(p) in " ".join(w)):
            errs.append(f"{i}: prompt appears in the prediction")
        if set(rec) - {"text"}:
            errs.append(f"{i}: extra fields {sorted(set(rec) - {'text'})}")
    return errs


def zipit(pred_path, zpath):
    tmp = zpath.with_name(zpath.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    with zipfile.ZipFile(tmp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        json.loads(z.read("prediction.json"))
    tmp.replace(zpath)


def main():
    cfg, modes = args()
    out = subdir(cfg)
    pp = out / "prediction.json"
    if not pp.exists():
        raise SystemExit(f"missing {pp}  (run: python generate.py {cfg.cfgpath} test)")
    with open(pp, encoding="utf-8") as f:
        pred = json.load(f, object_pairs_hook=dupcheck)
    prompts = loadtest(cfg, "test")
    errs = validate(pred, prompts, int(cfg.generation.words))
    if errs:
        print(f"validation FAILED ({len(errs)} problems); submission.zip not written")
        for e in errs[:20]:
            print("  " + e)
        sys.exit(1)
    zipit(pp, out / "submission.zip")
    gl = out / "genlog.json"
    genlog = json.load(open(gl)) if gl.exists() else {}
    m, g = cfg.model, cfg.generation
    reg = {}
    try:
        import yaml
        for e in yaml.safe_load(open(path("phase1/experiments.yaml")))["experiments"]:
            if e["id"] == cfg.id:
                reg = e
    except Exception:
        pass
    from tokenizer import gettok
    tok = gettok(cfg)
    params = reg.get("params") if isinstance(reg.get("params"), int) else None
    if params is None:
        import torch
        from models import build, countparams
        with torch.device("meta"):
            params = countparams(build(cfg.model))["total"]
            if cfg.get("pool"):
                params = 0
                cks = {json.load(open(path(f"runs/pool/{t}.json")))["meta"]["checkpoint"] for t in cfg.pool.tags}
                for ck in sorted(cks):
                    em = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]["model"]
                    params += countparams(build(wrap(em)))["total"]
            for r in cfg.generation.get("ensemble") or []:
                from checkpoint import ckptpath
                em = torch.load(ckptpath(path(r), cfg.generation.get("ensckpt", "last")), map_location="cpu",
                                weights_only=False)["cfg"]["model"]
                params += countparams(build(wrap(em)))["total"]
    meta = {
        "id": cfg.id, "name": cfg.get("subname", cfg.name), "category": reg.get("category"),
        "hypothesis": reg.get("hypothesis"), "architecture": m.name, "params": params,
        "initialization": f"random N(0,{m.get('init', 0.02)}), residual projections /sqrt(2L); no pretrained weights",
        "tokenizer": cfg.tokenizer.source, "tokenizer_fp": tok.fp, "run": str(cfg.get("run", f"runs/{cfg.name}")),
        "checkpoint": genlog.get("checkpoint"), "checkpoint_step": genlog.get("step"),
        "training_tokens": genlog.get("tokens"), "context": m.get("seq"),
        "position": m.get("pos", "rope" if m.name != "starter" else "learned"),
        "attention": {"qheads": m.get("qheads"), "kvheads": m.get("kvheads", m.get("qheads")),
                      "window": m.get("window"), "globalevery": m.get("globalevery"), "attnevery": m.get("attnevery")},
        "data_mixture": {"sources": cfg.data.sources, "mixing": cfg.data.mixing, "alpha": cfg.data.get("alpha")},
        "decoding": g.mode, "temperature": g.get("temperature"), "topk": g.get("topk"), "topp": g.get("topp"),
        "repetition": g.get("repetition"), "norepeat": g.get("norepeat"), "seed": g.seed,
        "test_fp": filefp(path(cfg.data.root) / cfg.data.test),
        "prediction_sha256": hashlib.sha256(pp.read_bytes()).hexdigest(),
        "generation": genlog,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "compliance": {"from_scratch": True, "pretrained_weights": False, "external_data": False,
                       "param_range_ok": 50e6 <= params <= 300e6},
    }
    writejson(out / "meta.json", meta, 1)
    print(f"ok: {len(pred)} predictions, {cfg.generation.words} words each -> {out / 'submission.zip'}")


if __name__ == "__main__":
    main()
