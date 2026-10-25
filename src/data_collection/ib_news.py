from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from data_collection.config import Settings, ensure_project_dirs
from data_collection.ib_connector import _import_ib_async, connect_ib
from data_collection.symbols import canonical_symbol, ib_contract_symbol


DEFAULT_API_NEWS_PROVIDERS = ("BRFG", "BRFUPDN", "DJNL")
_HEADLINE_TAG_RE = re.compile(r"^\{[^}]*\}")


def clean_headline(text: str | None) -> str:
    return _HEADLINE_TAG_RE.sub("", text or "").strip()


def _provider_attr(provider, *names: str):
    for name in names:
        value = getattr(provider, name, None)
        if value is not None:
            return value
    return None


def provider_to_dict(provider) -> dict:
    return {
        "code": _provider_attr(provider, "code", "providerCode", "ProviderCode"),
        "name": _provider_attr(provider, "name", "providerName", "ProviderName"),
        "repr": repr(provider),
    }


def headline_to_dict(item) -> dict:
    when = getattr(item, "time", None)
    return {
        "time": when.isoformat() if when is not None else None,
        "providerCode": getattr(item, "providerCode", None),
        "articleId": getattr(item, "articleId", None),
        "headline_raw": getattr(item, "headline", None),
        "headline": clean_headline(getattr(item, "headline", None)),
        "extraData": getattr(item, "extraData", None),
    }


def _article_text(article) -> str:
    for name in ("articleText", "text"):
        value = getattr(article, name, None)
        if value is not None:
            return str(value)
    return str(article)


def _provider_query(provider_codes: Sequence[str]) -> str:
    clean = [str(code).strip().upper() for code in provider_codes if str(code).strip()]
    if not clean:
        raise ValueError("At least one provider code is required.")
    return "+".join(clean)


def _parse_provider_codes(raw: str) -> list[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def _parse_provider_groups(raw: str) -> list[list[str]]:
    groups = []
    for group in raw.split(";"):
        codes = _parse_provider_codes(group.replace("+", ","))
        if codes:
            groups.append(codes)
    if not groups:
        raise ValueError("At least one provider group is required.")
    return groups


def _default_end() -> str:
    return datetime.now().strftime("%Y%m%d %H:%M:%S")


def _start_from_days(days: int) -> str:
    return (datetime.now() - timedelta(days=int(days))).strftime("%Y%m%d %H:%M:%S")


def _resolve_window(args: argparse.Namespace) -> tuple[str, str]:
    end = args.end or _default_end()
    start = args.start or _start_from_days(args.days)
    return start, end


def _qualify_stock(ib, symbol: str):
    _, Stock, _ = _import_ib_async()
    clean_symbol = canonical_symbol(symbol)
    contract = Stock(ib_contract_symbol(clean_symbol), "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"Could not qualify IB stock contract for {clean_symbol}.")
    return clean_symbol, qualified[0]


def list_providers_payload(ib) -> dict:
    providers = [provider_to_dict(provider) for provider in ib.reqNewsProviders()]
    provider_codes = [provider["code"] for provider in providers]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "providers_count": len(providers),
        "providers": providers,
        "default_provider_codes_expected_from_ib_docs": list(DEFAULT_API_NEWS_PROVIDERS),
        "default_provider_codes_present": [
            code for code in DEFAULT_API_NEWS_PROVIDERS if code in provider_codes
        ],
    }


def fetch_headlines_payload(
    ib,
    *,
    symbol: str,
    provider_codes: Sequence[str],
    start: str,
    end: str,
    limit: int,
) -> dict:
    clean_symbol, contract = _qualify_stock(ib, symbol)
    provider_query = _provider_query(provider_codes)
    items = ib.reqHistoricalNews(contract.conId, provider_query, start, end, int(limit), [])
    # A timed-out request comes back as None rather than raising, and iterating it threw
    # TypeError on three of twenty stocks on 2026-08-04. Reporting the failure is the
    # point: an empty list would read as "this stock has no news", and the news seat
    # would then abstain for a transport reason the record could not distinguish from a
    # genuinely quiet stock.
    request_failed = items is None
    serialized = [headline_to_dict(item) for item in (items or [])]
    return {
        "request_failed": request_failed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": clean_symbol,
        "contract": {
            "conId": getattr(contract, "conId", None),
            "exchange": getattr(contract, "exchange", None),
            "primaryExchange": getattr(contract, "primaryExchange", None),
        },
        "provider_codes": [code.strip().upper() for code in provider_codes],
        "provider_query": provider_query,
        "start": start,
        "end": end,
        "limit": int(limit),
        "count": len(serialized),
        "items": serialized,
    }


def fetch_article_payload(
    ib,
    *,
    provider_code: str,
    article_id: str,
    include_body: bool = False,
) -> dict:
    article = ib.reqNewsArticle(provider_code.strip().upper(), article_id, [])
    text = _article_text(article)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "providerCode": provider_code.strip().upper(),
        "articleId": article_id,
        "article_fetch_ok": True,
        "article_text_length": len(text),
    }
    if include_body:
        payload["article_text"] = text
    return payload


def coverage_payload(
    ib,
    *,
    symbol: str,
    provider_groups: Sequence[Sequence[str]],
    start: str,
    end: str,
    limit: int,
) -> dict:
    clean_symbol = canonical_symbol(symbol)
    coverage = {}
    for group in provider_groups:
        query = _provider_query(group)
        try:
            payload = fetch_headlines_payload(
                ib,
                symbol=clean_symbol,
                provider_codes=group,
                start=start,
                end=end,
                limit=limit,
            )
            times = [
                datetime.fromisoformat(item["time"])
                for item in payload["items"]
                if item.get("time")
            ]
            providers = sorted({
                item["providerCode"]
                for item in payload["items"]
                if item.get("providerCode")
            })
            coverage[query] = {
                "ok": True,
                "count": payload["count"],
                "earliest": min(times).isoformat() if times else None,
                "latest": max(times).isoformat() if times else None,
                "providers_returned": providers,
            }
        except Exception as exc:
            coverage[query] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": clean_symbol,
        "start": start,
        "end": end,
        "limit": int(limit),
        "coverage": coverage,
    }


def _write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    return Settings(
        ib_host=args.host or settings.ib_host,
        ib_port=int(args.port) if args.port is not None else settings.ib_port,
        ib_client_id=(
            int(args.client_id)
            if args.client_id is not None
            else settings.ib_client_id
        ),
        data_dir=settings.data_dir,
        reports_dir=settings.reports_dir,
        rapidapi_key=settings.rapidapi_key,
        rapidapi_host=settings.rapidapi_host,
    )


def _with_ib(func, args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    ensure_project_dirs(settings)
    ib = connect_ib(settings)
    try:
        payload = func(ib, args)
        payload["gateway"] = {
            "host": settings.ib_host,
            "port": settings.ib_port,
            "client_id": settings.ib_client_id,
        }
    finally:
        ib.disconnect()
    _write_json(args.output, payload)
    print(f"WROTE {args.output}", flush=True)
    return 0


def providers_command(args: argparse.Namespace) -> int:
    return _with_ib(lambda ib, _args: list_providers_payload(ib), args)


def headlines_command(args: argparse.Namespace) -> int:
    start, end = _resolve_window(args)
    provider_codes = _parse_provider_codes(args.providers)
    return _with_ib(
        lambda ib, _args: fetch_headlines_payload(
            ib,
            symbol=args.symbol,
            provider_codes=provider_codes,
            start=start,
            end=end,
            limit=args.limit,
        ),
        args,
    )


def article_command(args: argparse.Namespace) -> int:
    return _with_ib(
        lambda ib, _args: fetch_article_payload(
            ib,
            provider_code=args.provider,
            article_id=args.article_id,
            include_body=args.include_body,
        ),
        args,
    )


def coverage_command(args: argparse.Namespace) -> int:
    start, end = _resolve_window(args)
    provider_groups = _parse_provider_groups(args.providers)
    return _with_ib(
        lambda ib, _args: coverage_payload(
            ib,
            symbol=args.symbol,
            provider_groups=provider_groups,
            start=start,
            end=end,
            limit=args.limit,
        ),
        args,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Brokers API news tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_gateway_args(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--host", default=None)
        command_parser.add_argument("--port", type=int, default=None)
        command_parser.add_argument("--client-id", type=int, default=None)

    providers = subparsers.add_parser("providers", help="List subscribed IB API news providers")
    add_gateway_args(providers)
    providers.add_argument("--output", default="reports/ib_news_providers.json")
    providers.set_defaults(func=providers_command)

    headlines = subparsers.add_parser("headlines", help="Fetch historical news headlines")
    add_gateway_args(headlines)
    headlines.add_argument("--symbol", required=True)
    headlines.add_argument("--providers", default="BRFG,BRFUPDN,DJNL")
    headlines.add_argument("--days", type=int, default=30)
    headlines.add_argument("--start")
    headlines.add_argument("--end")
    headlines.add_argument("--limit", type=int, default=100)
    headlines.add_argument("--output", default="reports/ib_news_headlines.json")
    headlines.set_defaults(func=headlines_command)

    article = subparsers.add_parser("article", help="Fetch one news article by provider/article id")
    add_gateway_args(article)
    article.add_argument("--provider", required=True)
    article.add_argument("--article-id", required=True)
    article.add_argument(
        "--include-body",
        action="store_true",
        help="Store article body in JSON. By default only stores fetch status and length.",
    )
    article.add_argument("--output", default="reports/ib_news_article.json")
    article.set_defaults(func=article_command)

    coverage = subparsers.add_parser("coverage", help="Probe headline coverage by provider group")
    add_gateway_args(coverage)
    coverage.add_argument("--symbol", required=True)
    coverage.add_argument(
        "--providers",
        default="BRFG;BRFUPDN;DJNL;BRFG,BRFUPDN,DJNL",
        help="Semicolon-separated provider groups; commas or plus signs combine providers.",
    )
    coverage.add_argument("--days", type=int, default=3650)
    coverage.add_argument("--start")
    coverage.add_argument("--end")
    coverage.add_argument("--limit", type=int, default=1000)
    coverage.add_argument("--output", default="reports/ib_news_coverage.json")
    coverage.set_defaults(func=coverage_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
