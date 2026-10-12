#!/usr/bin/env python3
"""Generate LspCpp protocol types and handler registrations from the LSP metaModel."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_lsp_metamodel_coverage as coverage  # noqa: E402

GENERATED_HEADER = "include/LibLsp/lsp/generated/lsp_generated_protocol.h"
HANDLER_FILE = "src/lsp/ProtocolJsonHandler.cpp"
GENERATED_INCLUDE = "LibLsp/lsp/generated/lsp_generated_protocol.h"

BEGIN_HEADER = "// BEGIN LSPGEN generated protocol types"
END_HEADER = "// END LSPGEN generated protocol types"
BEGIN_CLIENT_REQ = "    // BEGIN LSPGEN client request registrations"
END_CLIENT_REQ = "    // END LSPGEN client request registrations"
BEGIN_SERVER_REQ = "    // BEGIN LSPGEN server-initiated request registrations"
END_SERVER_REQ = "    // END LSPGEN server-initiated request registrations"
BEGIN_RSP = "    // BEGIN LSPGEN response registrations"
END_RSP = "    // END LSPGEN response registrations"
BEGIN_NTF = "    // BEGIN LSPGEN notification registrations"
END_NTF = "    // END LSPGEN notification registrations"

BASE_TYPE_MAP = {
    "string": "std::string",
    "boolean": "bool",
    "integer": "int32_t",
    "uinteger": "uint32_t",
    "decimal": "double",
    "null": "JsonNull",
    "URI": "lsDocumentUri",
    "DocumentUri": "lsDocumentUri",
    "RegExp": "std::string",
}

KNOWN_TYPE_ALIASES = {
    "TextDocumentIdentifier": "lsTextDocumentIdentifier",
    "VersionedTextDocumentIdentifier": "lsVersionedTextDocumentIdentifier",
    "TextDocumentItem": "lsTextDocumentItem",
    "TextDocumentPositionParams": "lsTextDocumentPositionParams",
    "Range": "lsRange",
    "Position": "lsPosition",
    "Diagnostic": "lsDiagnostic",
    "TextEdit": "lsTextEdit",
    "Location": "lsLocation",
    "LocationLink": "lsLocationLink",
    "WorkspaceEdit": "lsWorkspaceEdit",
    "Command": "lsCommandWithAny",
    "MarkupContent": "MarkupContent",
    "MarkedString": "lsMarkedString",
    "SymbolInformation": "lsSymbolInformation",
    "DocumentHighlight": "lsDocumentHighlight",
    "CompletionItem": "lsCompletionItem",
    "CompletionList": "CompletionList",
    "CodeAction": "CodeAction",
    "DocumentSymbol": "lsDocumentSymbol",
    "FoldingRange": "lsFoldingRange",
    "SemanticTokens": "SemanticTokens",
    "InlayHint": "lsInlayHint",
    "CallHierarchyItem": "CallHierarchyItem",
    "TypeHierarchyItem": "TypeHierarchyItem",
    "LSPAny": "lsp::Any",
    "LSPObject": "lsp::Any",
    "LSPArray": "lsp::Any",
}

HEADER_INCLUDES = [
    "LibLsp/JsonRpc/NotificationInMessage.h",
    "LibLsp/JsonRpc/RequestInMessage.h",
    "LibLsp/JsonRpc/lsResponseMessage.h",
    "LibLsp/lsp/lsAny.h",
    "LibLsp/lsp/lsDocumentUri.h",
    "LibLsp/lsp/lsRange.h",
    "LibLsp/lsp/lsPosition.h",
    "LibLsp/lsp/lsTextDocumentIdentifier.h",
    "LibLsp/lsp/symbol.h",
]


@dataclass
class MethodTarget:
    method: str
    kind: str  # request | notification
    symbol: str
    params_cpp: str
    result_cpp: Optional[str]
    message_direction: str
    needs_type_gen: bool
    needs_request_parser: bool
    needs_response_parser: bool
    needs_notification_parser: bool


@dataclass
class GenerationPlan:
    methods: List[MethodTarget] = field(default_factory=list)
    type_names: Set[str] = field(default_factory=set)
    client_request_blocks: List[str] = field(default_factory=list)
    server_request_blocks: List[str] = field(default_factory=list)
    response_blocks: List[str] = field(default_factory=list)
    notification_blocks: List[str] = field(default_factory=list)
    header_body: str = ""
    collisions: List[str] = field(default_factory=list)


class MetaModelIndex:
    def __init__(self, metamodel_path: Path):
        payload = coverage.load_json(metamodel_path)
        self.version = payload.get("metaData", {}).get("version", "unknown")
        self.requests = {entry["method"]: entry for entry in payload.get("requests", [])}
        self.notifications = {entry["method"]: entry for entry in payload.get("notifications", [])}
        self.structures = {entry["name"]: entry for entry in payload.get("structures", [])}
        self.enumerations = {entry["name"]: entry for entry in payload.get("enumerations", [])}
        self.type_aliases = {entry["name"]: entry for entry in payload.get("typeAliases", [])}

    def entry_for(self, method: str) -> Tuple[str, dict]:
        if method in self.requests:
            return "request", self.requests[method]
        if method in self.notifications:
            return "notification", self.notifications[method]
        raise KeyError(f"Unknown LSP method: {method}")


class ExistingTypesIndex:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.structs: Set[str] = set()
        self.enums: Set[str] = set()
        self.symbols: Set[str] = set()
        self.generated_symbols: Set[str] = set()
        self._scan()

    def _scan(self) -> None:
        generated_path = self.repo_root / GENERATED_HEADER
        generated_text = generated_path.read_text(encoding="utf-8") if generated_path.exists() else ""
        for macro in ("DEFINE_REQUEST_RESPONSE_TYPE", "DEFINE_NOTIFICATION_TYPE"):
            for symbol, _method in coverage.extract_macros(generated_text, macro):
                self.generated_symbols.add(symbol)

        for pattern in coverage.HEADER_GLOBS:
            for path in self.repo_root.glob(pattern):
                if path.resolve() == generated_path.resolve():
                    continue
                text = path.read_text(encoding="utf-8")
                self.structs.update(re.findall(r"\bstruct\s+(\w+)", text))
                self.enums.update(re.findall(r"\benum class\s+(\w+)", text))
                for macro in ("DEFINE_REQUEST_RESPONSE_TYPE", "DEFINE_NOTIFICATION_TYPE"):
                    for symbol, _method in coverage.extract_macros(text, macro):
                        self.symbols.add(symbol)

        for relative_path in coverage.EXTRA_HEADER_FILES:
            text = (self.repo_root / relative_path).read_text(encoding="utf-8")
            for macro in ("DEFINE_REQUEST_RESPONSE_TYPE", "DEFINE_NOTIFICATION_TYPE"):
                for symbol, _method in coverage.extract_macros(text, macro):
                    self.symbols.add(symbol)

    def has_type(self, name: str) -> bool:
        return name in self.structs or name in self.enums or name in KNOWN_TYPE_ALIASES

    def has_symbol(self, name: str) -> bool:
        if name in self.generated_symbols:
            return False
        return name in self.symbols


class TypeResolver:
    def __init__(self, index: MetaModelIndex, existing: ExistingTypesIndex):
        self.index = index
        self.existing = existing
        self.pending_structs: OrderedDict[str, dict] = OrderedDict()
        self.pending_enums: OrderedDict[str, dict] = OrderedDict()
        self.pending_aliases: OrderedDict[str, dict] = OrderedDict()
        self.generated_structs: Set[str] = set()
        self.generated_enums: Set[str] = set()
        self.generated_aliases: Set[str] = set()
        self.generation_order: List[Tuple[str, str]] = []
        self.required_includes: Set[str] = set()

    def collect_dependencies(self, type_node: dict) -> None:
        kind = type_node.get("kind")
        if kind == "reference":
            self._collect_reference(type_node["name"])
        elif kind == "array":
            self.collect_dependencies(type_node["element"])
        elif kind == "map":
            self.collect_dependencies(type_node["value"])
        elif kind == "or":
            for item in type_node.get("items", []):
                self.collect_dependencies(item)
        elif kind == "tuple":
            for item in type_node.get("items", []):
                self.collect_dependencies(item)

    def _collect_reference(self, name: str) -> None:
        if self.existing.has_type(name) or name in KNOWN_TYPE_ALIASES:
            return
        if name in self.pending_structs or name in self.pending_enums or name in self.pending_aliases:
            return
        if name in self.index.structures:
            self.pending_structs[name] = self.index.structures[name]
            entry = self.index.structures[name]
            for prop in entry.get("properties", []):
                self.collect_dependencies(prop["type"])
            for mixin in entry.get("mixins", []):
                self.collect_dependencies(mixin)
        elif name in self.index.enumerations:
            self.pending_enums[name] = self.index.enumerations[name]
        elif name in self.index.type_aliases:
            self.pending_aliases[name] = self.index.type_aliases[name]
            self.collect_dependencies(self.index.type_aliases[name]["type"])

    def resolve(self, type_node: dict, *, nullable_union: bool = False) -> str:
        kind = type_node.get("kind")
        if kind == "base":
            mapped = BASE_TYPE_MAP.get(type_node["name"], type_node["name"])
            if mapped == type_node["name"] and mapped not in BASE_TYPE_MAP.values():
                return "lsp::Any"
            return mapped
        if kind == "reference":
            name = type_node["name"]
            if name in KNOWN_TYPE_ALIASES:
                return KNOWN_TYPE_ALIASES[name]
            if self.existing.has_type(name):
                return name
            if name in self.generated_structs or name in self.pending_structs:
                return name
            if name in self.generated_enums or name in self.pending_enums:
                return name
            if name in self.generated_aliases or name in self.pending_aliases:
                return name
            self._collect_reference(name)
            return name
        if kind == "array":
            element = self.resolve(type_node["element"])
            return f"std::vector<{element}>"
        if kind == "map":
            value = self.resolve(type_node["value"])
            return f"std::map<std::string, {value}>"
        if kind == "literal":
            return "lsp::Any"
        if kind == "stringLiteral":
            return "std::string"
        if kind == "or":
            return self._resolve_or(type_node, nullable_union=nullable_union)
        if kind == "tuple":
            parts = [self.resolve(item) for item in type_node.get("items", [])]
            if len(parts) == 2:
                return f"std::pair<{parts[0]}, {parts[1]}>"
            return "lsp::Any"
        return "lsp::Any"

    def _resolve_or(self, type_node: dict, *, nullable_union: bool) -> str:
        items = type_node.get("items", [])
        non_null = [item for item in items if not (item.get("kind") == "base" and item.get("name") == "null")]
        if len(non_null) == 1 and len(items) == 2:
            inner = self.resolve(non_null[0])
            if nullable_union:
                return f"Nullable<{inner}>"
            return f"optional<{inner}>"
        resolved = [self.resolve(item) for item in non_null]
        if not resolved:
            return "JsonNull"
        if len(resolved) == 1:
            return resolved[0]
        if len(resolved) == 2:
            return f"std::pair<optional<{resolved[0]}>, optional<{resolved[1]}>>"
        pair = resolved[-1]
        for item in reversed(resolved[:-1]):
            pair = f"std::pair<optional<{item}>, optional<{pair}>>"
        return pair

    def finalize_pending(self) -> None:
        while self.pending_structs or self.pending_enums or self.pending_aliases:
            if self.pending_enums:
                name = next(iter(self.pending_enums))
                self.pending_enums.pop(name)
                self.generated_enums.add(name)
                self.generation_order.append(("enum", name))
                continue
            if self.pending_aliases:
                name = next(iter(self.pending_aliases))
                entry = self.pending_aliases.pop(name)
                self.collect_dependencies(entry["type"])
                self.generated_aliases.add(name)
                self.generation_order.append(("alias", name))
                continue
            name = next(iter(self.pending_structs))
            entry = self.pending_structs.pop(name)
            for prop in entry.get("properties", []):
                self.collect_dependencies(prop["type"])
            for mixin in entry.get("mixins", []):
                self.collect_dependencies(mixin)
            self.generated_structs.add(name)
            self.generation_order.append(("struct", name))


def type_name_to_symbol(type_name: str) -> str:
    for suffix in ("Request", "Notification"):
        if type_name.endswith(suffix):
            base = type_name[: -len(suffix)]
            break
    else:
        base = type_name
    if not base:
        base = type_name
    return f"gen_{base[0].lower()}{base[1:]}"


def method_to_symbol(method: str) -> str:
    cleaned = method.replace("$/", "$").replace("/", "_").replace("-", "_")
    return f"gen_{cleaned}"


def choose_symbol(entry: dict, method: str) -> str:
    type_name = entry.get("typeName")
    if type_name:
        symbol = type_name_to_symbol(type_name)
        return symbol
    return method_to_symbol(method)


def build_targets(
    report: coverage.CoverageReport,
    index: MetaModelIndex,
    forced_methods: Set[str],
) -> List[MethodTarget]:
    targets: Dict[str, MethodTarget] = {}

    def add_method(method: str, *, needs_type: bool, req: bool, rsp: bool, ntf: bool) -> None:
        kind, entry = index.entry_for(method)
        symbol = choose_symbol(entry, method)
        resolver = TypeResolver(index, ExistingTypesIndex(report.repo_root))
        if kind == "request":
            params_cpp = resolver.resolve(entry["params"])
            result_cpp = resolver.resolve(entry["result"], nullable_union=True)
        else:
            params_cpp = resolver.resolve(entry["params"])
            result_cpp = None
        targets[method] = MethodTarget(
            method=method,
            kind=kind,
            symbol=symbol,
            params_cpp=params_cpp,
            result_cpp=result_cpp,
            message_direction=entry.get("messageDirection", "clientToServer"),
            needs_type_gen=needs_type,
            needs_request_parser=req,
            needs_response_parser=rsp,
            needs_notification_parser=ntf,
        )

    if forced_methods:
        for method in sorted(forced_methods):
            kind, _entry = index.entry_for(method)
            add_method(
                method,
                needs_type=True,
                req=kind == "request",
                rsp=kind == "request",
                ntf=kind == "notification",
            )
        return list(targets.values())

    for method in report.remaining.get("missing_type_declaration", []):
        kind, _entry = index.entry_for(method)
        add_method(
            method,
            needs_type=True,
            req=kind == "request",
            rsp=kind == "request",
            ntf=kind == "notification",
        )

    for method in report.remaining.get("missing_request_parser", []):
        if method in targets:
            targets[method].needs_request_parser = True
        else:
            add_method(method, needs_type=False, req=True, rsp=False, ntf=False)

    for method in report.remaining.get("missing_response_parser", []):
        if method in targets:
            targets[method].needs_response_parser = True
        else:
            add_method(method, needs_type=False, req=False, rsp=True, ntf=False)

    for method in report.remaining.get("missing_notification_parser", []):
        if method in targets:
            targets[method].needs_notification_parser = True
        else:
            add_method(method, needs_type=False, req=False, rsp=False, ntf=True)

    return list(targets.values())


def render_enum(name: str, entry: dict) -> str:
    lines = [f"enum class {name}", "{"]
    for item in entry.get("values", []):
        value = item.get("value")
        item_name = item["name"]
        if isinstance(value, int):
            lines.append(f"    {item_name} = {value},")
        else:
            lines.append(f"    {item_name},")
    lines.append("};")
    lines.append(f"MAKE_REFLECT_TYPE_PROXY({name})")
    return "\n".join(lines)


def render_alias(name: str, cpp_type: str) -> str:
    return f"using {name} = {cpp_type};"


def flatten_properties(entry: dict, index: MetaModelIndex) -> List[dict]:
    props: List[dict] = []
    seen: Set[str] = set()
    for mixin in entry.get("mixins", []):
        if mixin.get("kind") != "reference":
            continue
        mixin_entry = index.structures.get(mixin["name"])
        if not mixin_entry:
            continue
        for prop in flatten_properties(mixin_entry, index):
            if prop["name"] not in seen:
                props.append(prop)
                seen.add(prop["name"])
    for prop in entry.get("properties", []):
        if prop["name"] not in seen:
            props.append(prop)
            seen.add(prop["name"])
    return props


def render_struct(name: str, entry: dict, index: MetaModelIndex, resolver: TypeResolver) -> str:
    props = flatten_properties(entry, index)
    fields: List[str] = []
    reflect_fields: List[str] = []
    swap_fields: List[str] = []
    for prop in props:
        cpp_type = resolver.resolve(prop["type"])
        if prop.get("optional"):
            cpp_type = f"optional<{cpp_type}>"
        fields.append(f"    {cpp_type} {prop['name']};")
        reflect_fields.append(prop["name"])
        swap_fields.append(prop["name"])

    lines = [f"struct {name}", "{"]
    if fields:
        lines.extend(fields)
    else:
        lines.append("    // generated empty struct")
    if swap_fields:
        lines.append(f"    MAKE_SWAP_METHOD({name}, {', '.join(swap_fields)})")
    lines.append("};")
    lines.append(f"MAKE_REFLECT_STRUCT({name}, {', '.join(reflect_fields) if reflect_fields else ''})")
    return "\n".join(line for line in lines if line is not None)


def render_method_macro(target: MethodTarget) -> str:
    if target.kind == "notification":
        return (
            f'DEFINE_NOTIFICATION_TYPE({target.symbol}, {target.params_cpp}, "{target.method}")'
        )
    return (
        f"DEFINE_REQUEST_RESPONSE_TYPE("
        f"{target.symbol}, {target.params_cpp}, {target.result_cpp}, \"{target.method}\")"
    )


def render_request_registration(symbol: str) -> str:
    return (
        f"    handler.method2request[{symbol}::request::kMethodInfo] = [](Reader& visitor)\n"
        f"    {{ return {symbol}::request::ReflectReader(visitor); }};"
    )


def render_response_registration(symbol: str) -> str:
    return (
        f"    handler.method2response[{symbol}::request::kMethodInfo] = [](Reader& visitor)\n"
        f"    {{\n"
        f"        if (visitor.HasMember(\"error\"))\n"
        f"        {{\n"
        f"            return Rsp_Error::ReflectReader(visitor);\n"
        f"        }}\n"
        f"        return {symbol}::response::ReflectReader(visitor);\n"
        f"    }};"
    )


def render_notification_registration(symbol: str) -> str:
    return (
        f"    handler.method2notification[{symbol}::notify::kMethodInfo] = [](Reader& visitor)\n"
        f"    {{ return {symbol}::notify::ReflectReader(visitor); }};"
    )


def generate_plan(
    report: coverage.CoverageReport,
    index: MetaModelIndex,
    forced_methods: Set[str],
    *,
    force: bool,
) -> GenerationPlan:
    plan = GenerationPlan()
    existing = ExistingTypesIndex(report.repo_root)
    targets = build_targets(report, index, forced_methods)
    plan.methods = targets
    if not targets:
        return plan

    resolver = TypeResolver(index, existing)
    for target in targets:
        if not target.needs_type_gen:
            continue
        kind, entry = index.entry_for(target.method)
        resolver.collect_dependencies(entry["params"])
        if kind == "request":
            resolver.collect_dependencies(entry["result"])
        plan.type_names.update(resolver.pending_structs)
        plan.type_names.update(resolver.pending_enums)
        plan.type_names.update(resolver.pending_aliases)

    # Merge all dependencies from all targets
    merged_resolver = TypeResolver(index, existing)
    for target in targets:
        if not target.needs_type_gen:
            continue
        kind, entry = index.entry_for(target.method)
        merged_resolver.collect_dependencies(entry["params"])
        if kind == "request":
            merged_resolver.collect_dependencies(entry["result"])
    merged_resolver.finalize_pending()

    for name in merged_resolver.generated_structs | merged_resolver.generated_enums | merged_resolver.generated_aliases:
        if existing.has_type(name) and name not in existing.generated_symbols and not force:
            plan.collisions.append(f"type {name}")

    for target in targets:
        if existing.has_symbol(target.symbol) and not force:
            plan.collisions.append(f"symbol {target.symbol}")

    body_parts: List[str] = [BEGIN_HEADER, ""]
    for kind, name in merged_resolver.generation_order:
        if kind == "enum":
            body_parts.append(render_enum(name, index.enumerations[name]))
        elif kind == "alias":
            alias_cpp = merged_resolver.resolve(index.type_aliases[name]["type"])
            body_parts.append(render_alias(name, alias_cpp))
        elif kind == "struct":
            body_parts.append(render_struct(name, index.structures[name], index, merged_resolver))
        body_parts.append("")

    for target in targets:
        if target.needs_type_gen:
            body_parts.append(render_method_macro(target))
            body_parts.append("")

    body_parts.append(END_HEADER)
    plan.header_body = "\n".join(body_parts).rstrip() + "\n"

    for target in targets:
        if target.needs_request_parser:
            block = render_request_registration(target.symbol)
            if target.message_direction == "serverToClient":
                plan.server_request_blocks.append(block)
            else:
                plan.client_request_blocks.append(block)
        if target.needs_response_parser:
            plan.response_blocks.append(render_response_registration(target.symbol))
        if target.needs_notification_parser:
            plan.notification_blocks.append(render_notification_registration(target.symbol))

    return plan


def wrap_block(begin: str, end: str, lines: List[str]) -> str:
    if not lines:
        return ""
    return begin + "\n" + "\n".join(lines) + "\n" + end


def render_generated_header(plan: GenerationPlan) -> str:
    includes = "\n".join(f'#include "{inc}"' for inc in HEADER_INCLUDES)
    return (
        "#pragma once\n\n"
        f"{includes}\n\n"
        f"{plan.header_body}"
    )


def write_generated_header(path: Path, plan: GenerationPlan) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        pattern = re.compile(re.escape(BEGIN_HEADER) + r".*?" + re.escape(END_HEADER), re.DOTALL)
        if pattern.search(existing):
            header_text = pattern.sub(plan.header_body.rstrip(), existing)
        else:
            header_text = existing.rstrip() + "\n\n" + plan.header_body
    else:
        header_text = render_generated_header(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header_text if header_text.endswith("\n") else header_text + "\n", encoding="utf-8")


def insert_or_replace_block(content: str, function_name: str, begin: str, end: str, lines: List[str]) -> str:
    if not lines:
        return content
    wrapped = wrap_block(begin, end, lines)
    pattern = re.compile(
        rf"(void {re.escape(function_name)}\(MessageJsonHandler& handler\)\s*\{{)(.*?)(\n\}})",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        raise RuntimeError(f"Could not locate function {function_name} in {HANDLER_FILE}")

    body = match.group(2)
    sentinel_pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if sentinel_pattern.search(body):
        body = sentinel_pattern.sub(wrapped, body)
    else:
        body = body.rstrip() + "\n\n" + wrapped
    return content[: match.start()] + match.group(1) + body + match.group(3) + content[match.end() :]


LEGACY_SENTINEL_PAIRS = (
    ("    // BEGIN LSPGEN request registrations", "    // END LSPGEN request registrations"),
)


def remove_legacy_sentinels(content: str) -> str:
    for begin, end in LEGACY_SENTINEL_PAIRS:
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        content = pattern.sub("", content)
    return content


def patch_handler_content(content: str, plan: GenerationPlan) -> str:
    content = remove_legacy_sentinels(content)
    include_line = f'#include "{GENERATED_INCLUDE}"'
    if include_line not in content:
        marker = '#include "LibLsp/lsp/workspace/configuration.h"'
        if marker in content:
            content = content.replace(marker, marker + "\n" + include_line)
        else:
            content = include_line + "\n" + content

    content = insert_or_replace_block(
        content,
        "AddStadardResponseJsonRpcMethod",
        BEGIN_RSP,
        END_RSP,
        plan.response_blocks,
    )
    content = insert_or_replace_block(
        content,
        "AddNotifyJsonRpcMethod",
        BEGIN_NTF,
        END_NTF,
        plan.notification_blocks,
    )
    content = insert_or_replace_block(
        content,
        "AddStadardResponseJsonRpcMethod",
        BEGIN_SERVER_REQ,
        END_SERVER_REQ,
        plan.server_request_blocks,
    )
    content = insert_or_replace_block(
        content,
        "AddStandardRequestJsonRpcMethod",
        BEGIN_CLIENT_REQ,
        END_CLIENT_REQ,
        plan.client_request_blocks,
    )
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LspCpp protocol types from the LSP metaModel.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--metamodel", type=Path, default=None)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--method", action="append", default=[], help="Generate for specific method(s), bypassing allowlist.")
    parser.add_argument("--write", action="store_true", help="Write generated files (default is dry-run).")
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing generated symbols/types.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = coverage.analyze_coverage(args.repo_root, args.metamodel, args.allowlist)
    index = MetaModelIndex(report.metamodel_path)
    forced = set(args.method)
    plan = generate_plan(report, index, forced, force=args.force)

    print("LspCpp lspgen plan")
    print(f"  metaModel: {report.metamodel_path} (version {index.version})")
    print(f"  methods: {len(plan.methods)}")
    if plan.collisions:
        print(f"  collisions: {len(plan.collisions)}")
        for item in plan.collisions:
            print(f"    - {item}")
        if not args.force:
            print("Refusing to write due to collisions (use --force to override).")
            return 1

    if not plan.methods:
        print("Nothing to generate.")
        return 0

    print("Targets:")
    for target in plan.methods:
        flags = []
        if target.needs_type_gen:
            flags.append("types")
        if target.needs_request_parser:
            flags.append("request")
        if target.needs_response_parser:
            flags.append("response")
        if target.needs_notification_parser:
            flags.append("notification")
        print(f"  - {target.method} ({target.symbol}) [{', '.join(flags)}]")

    header_path = report.repo_root / GENERATED_HEADER
    handler_path = report.repo_root / HANDLER_FILE
    header_text = render_generated_header(plan)

    print()
    print(f"Generated header preview ({header_path}):")
    print("-" * 40)
    preview = header_text if len(header_text) <= 4000 else header_text[:4000] + "\n... (truncated)"
    print(preview)
    print("-" * 40)

    if plan.client_request_blocks or plan.server_request_blocks or plan.response_blocks or plan.notification_blocks:
        print("Handler registration blocks:")
        for block in (
            plan.client_request_blocks
            + plan.server_request_blocks
            + plan.response_blocks
            + plan.notification_blocks
        ):
            print(block)
            print()

    if not args.write:
        print("Dry-run only. Re-run with --write to apply changes.")
        return 0

    header_path.parent.mkdir(parents=True, exist_ok=True)
    write_generated_header(header_path, plan)
    handler_text = handler_path.read_text(encoding="utf-8")
    handler_path.write_text(patch_handler_content(handler_text, plan), encoding="utf-8")
    print(f"Wrote {header_path}")
    print(f"Updated {handler_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
