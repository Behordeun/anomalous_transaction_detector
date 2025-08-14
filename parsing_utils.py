import logging

import re
from typing import Any, Dict, Optional
from datetime import datetime
import pandas as pd
from dateutil import parser as date_parser

def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse a date/time string into a :class:`datetime` object.

    The raw timestamps in the logs come in two primary formats:

    - ISO style "YYYY-MM-DD HH:MM:SS" (e.g. "2025-07-05 19:18:10")
    - European style "DD/MM/YYYY HH:MM:SS" (e.g. "24/07/2025 22:47:06")

    Using ``dayfirst=True`` unconditionally can lead to mis-parsed
    values for ISO strings (interpreting ``2025-06-12`` as
    ``2025-12-06``).  To mitigate this, we infer the correct
    ``dayfirst`` flag based on the delimiter present.  If the
    timestamp contains a forward slash ``/`` we assume day comes
    first; otherwise we assume the canonical ISO ordering.

    Parameters
    ----------
    dt_str : str
        Date/time string extracted from a log.

    Returns
    -------
    datetime or None
        Parsed datetime or ``None`` if parsing fails.
    """
    if pd.isna(dt_str):
        return None
    dt_str = dt_str.strip()
    if not dt_str:
        return None
    # Determine the appropriate day/month ordering
    dayfirst = "/" in dt_str
    try:
        return date_parser.parse(dt_str, dayfirst=dayfirst)
    except (ValueError, TypeError):
        return None

# Setup logging for production
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


###############################################################################
# Parsing functions
###############################################################################


def parse_triple_colon(log: str) -> Optional[Dict[str, Any]]:
    """
    Parse logs with triple-colon format and some compound patterns.

    This function attempts to extract transaction details from logs that use a triple-colon ':::' delimiter or match a specific compound regex pattern.
    It returns a dictionary of parsed fields if successful, otherwise None.

    Parameters
    ----------
    log : str
        Raw log string.

    Returns
    -------
    dict or None
        Parsed components or None if no pattern matches.
    """

    m = re.match(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):::(?P<user>\d+):::(?P<type>[\w-]+):::(?P<amount>[€£$]?\d[\d,.]*):::(?P<location>[^:]+):::(?P<device>.+)$",
        log,
    )
    if m:
        gd = m.groupdict()
    # use local parse_datetime

        ts = parse_datetime(gd["timestamp"])
        iso_ts = ts.isoformat(sep=" ") if ts else gd["timestamp"]
        location = gd["location"].strip()
        location_norm = location.lower()
        return {
            "timestamp": iso_ts,
            "user": f"user{gd['user']}",
            "type": gd["type"].lower(),
            "amount": str(gd["amount"]),
            "location": location,
            "location_norm": location_norm,
            "device": gd["device"].strip(),
        }
    # Compound pattern fallback
    m = re.match(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|user:(?P<user>\d+)\|type:(?P<type>[\w-]+)\|amt:(?P<amount>[€£$]?\d[\d,.]*)\|location:(?P<location>[^|]+)\|device:(?P<device>.+)$",
        log,
    )
    if m:
        gd = m.groupdict()
    # use local parse_datetime

        ts = parse_datetime(gd["timestamp"])
        iso_ts = ts.isoformat(sep=" ") if ts else gd["timestamp"]
        return {
            "timestamp": iso_ts,
            "user": f"user{gd['user']}",
            "type": gd["type"].lower(),
            "amount": str(gd["amount"]),
            "location": gd["location"].strip(),
            "device": gd["device"].strip(),
        }
    return None


def parse_simple_colon(log: str) -> Optional[Dict[str, Any]]:
    """
    Parse logs with double-colon format and simple colon patterns.

    This function extracts transaction details from logs using double-colon '::' delimiters or simple colon-separated fields.
    Returns a dictionary of parsed fields if successful, otherwise None.

    Parameters
    ----------
    log : str
        Raw log string.

    Returns
    -------
    dict or None
        Parsed components or None if no pattern matches.
    """

    m = re.match(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})::(?P<user>\d+)::(?P<type>[\w-]+)::(?P<amount>[€£$]?\d[\d,.]*)::(?P<location>[^:]+)::(?P<device>.+)$",
        log,
    )
    if m:
        gd = m.groupdict()
    # use local parse_datetime

        ts = parse_datetime(gd["timestamp"])
        iso_ts = ts.isoformat(sep=" ") if ts else gd["timestamp"]
        location = gd["location"].strip()
        location_norm = location.lower()
        return {
            "timestamp": iso_ts,
            "user": f"user{gd['user']}",
            "type": gd["type"].lower(),
            "amount": str(gd["amount"]),
            "location": location,
            "location_norm": location_norm,
            "device": gd["device"].strip(),
        }
    # Double-colon format
    if log.count("::") >= 5:
        parts = log.split("::")
        timestamp = parts[0]
        user = parts[1]
        type_ = parts[2].lower()
        amount = parts[3]
        location = parts[4]
        device = "::".join(parts[5:])
        location_norm = location.strip().lower()
        return {
            "timestamp": timestamp,
            "user": user,
            "type": type_,
            "amount": amount,
            "location": location,
            "location_norm": location_norm,
            "device": device,
        }
    return None


def parse_compound_patterns(log: str) -> Optional[Dict[str, Any]]:
    """
    Handle compound log patterns with fallback heuristics.

    Attempts to match the log string against several complex regex patterns inspired by observed log formats.
    Returns a dictionary of parsed fields if a pattern matches, otherwise None.

    Parameters
    ----------
    log : str
        Raw log string.

    Returns
    -------
    dict or None
        Parsed components or None if no pattern matches.
    """

    pattern_usr = re.compile(
        r"^usr:(?P<user>user\d+)\|(?P<type>[\w-]+)\|(?P<amount>[€£$]?\d[\d,.]*)"
        r"\|(?P<location>[^|]+)\|(?P<timestamp>[^|]+)\|(?P<device>.+)$",
        flags=re.IGNORECASE,
    )
    pattern_bracket = re.compile(
        r"^(?P<timestamp>.+?)>>\s*\[(?P<user>user\d+)\]\s*did\s*(?P<type>[\w-]+)\s*-\s*"
        r"amt=(?P<amount>[€£$]?\d[\d,.]*)\s*-\s*(?P<location>[^/]+)//\s*dev:(?P<device>.+)$",
        flags=re.IGNORECASE,
    )
    pattern_bar = re.compile(
        r"^(?P<timestamp>.+?)\|\s*user:\s*(?P<user>user\d+)\s*\|\s*"
        r"txn:\s*(?P<type>[\w-]+)\s+of\s*(?P<amount>[€£$]?\d[\d,.]*)\s+from\s*(?P<location>[^|]+)"
        r"\|\s*device:\s*(?P<device>.+)$",
        flags=re.IGNORECASE,
    )
    pattern_dash = re.compile(
        r"^(?P<timestamp>.+?)\s*-\s*user=(?P<user>user\d+)\s*-\s*"
        r"action=(?P<type>[\w-]+)\s*(?P<amount>[€£$]?\d[\d,.]*)\s*-\s*ATM:\s*(?P<location>.+?)"
        r"\s*-\s*device=(?P<device>.+)$",
        flags=re.IGNORECASE,
    )
    pattern_space = re.compile(
        r"^(?P<user>user\d+)\s+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s+(?P<type>[\w-]+)\s+(?P<amount>[€£$]?\d[\d,.]*)\s+(?P<location>\w+|None)\s+(?P<device>.+)$",
        flags=re.IGNORECASE,
    )
    for pattern in [
        pattern_usr,
        pattern_bracket,
        pattern_bar,
        pattern_dash,
        pattern_space,
    ]:
        m = pattern.match(log)
        if m:
            gd = m.groupdict()
            gd["type"] = gd["type"].lower()
            location = gd.get("location", "").strip()
            gd["location_norm"] = location.lower()
            return gd
    return None


def _fallback_parse_log(log: str) -> Optional[Dict[str, Any]]:
    """
    Fallback: try to extract partial info if all parsers fail.

    Uses regex heuristics to extract any available fields (timestamp, user, type, amount, location, device)
    from logs that do not match any known patterns. Returns a dictionary of found fields, or None if nothing is found.

    Parameters
    ----------
    log : str
        Raw log string.

    Returns
    -------
    dict or None
        Dictionary of partially extracted fields, or None if no fields found.
    """

    partial = {}
    found_fields = []
    ts_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", log)
    if not ts_match:
        ts_match = re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", log)
    if ts_match:
    # use local parse_datetime

        ts = parse_datetime(ts_match.group(0))
        partial["timestamp"] = ts.isoformat(sep=" ") if ts else ts_match.group(0)
        found_fields.append("timestamp")
    user_match = re.search(r"user[:\s]*(\d+)", log, re.IGNORECASE)
    if user_match:
        partial["user"] = f"user{user_match.group(1)}"
        found_fields.append("user")
    type_match = re.search(
        r"\b(withdrawal|deposit|transfer|purchase|cashout|top-up|debit|refund)\b",
        log,
        re.IGNORECASE,
    )
    if type_match:
        partial["type"] = type_match.group(1).lower()
        found_fields.append("type")
    amount_match = re.search(r"([€£$])([\d,.]+)", log)
    if amount_match:
        partial["amount"] = f"{amount_match.group(1)}{amount_match.group(2)}"
        found_fields.append("amount")
    else:
        amount_match = re.search(r"\b([\d,.]+)\b", log)
        if amount_match:
            partial["amount"] = amount_match.group(1)
            found_fields.append("amount")
    loc_match = re.search(r"location[:\s]*([^|]+)", log, re.IGNORECASE)
    if loc_match:
        location = loc_match.group(1).strip()
        partial["location"] = location
        partial["location_norm"] = location.lower()
    dev_match = re.search(r"device[:\s]*([^|]+)", log, re.IGNORECASE)
    if dev_match:
        partial["device"] = dev_match.group(1).strip()
        found_fields.append("device")
    if found_fields:
        partial["fields_found"] = found_fields
        logging.warning(f"Partial parse for log: {str(log)[:50]}... Parsed: {partial}")
        return partial
    logging.warning(f"Failed to parse log: {str(log)[:100]}")
    return None


def parse_log(log: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single log entry using multiple parsing strategies.

    Attempts to parse a transaction log using several strategies in order. Returns a dictionary of parsed fields if successful, otherwise None.

    Parameters
    ----------
    log : str
        Raw log string.

    Returns
    -------
    dict or None
        Parsed components or None if no pattern matches.
    """
    import pandas as pd

    if pd.isna(log) or not isinstance(log, str):
        return None
    log = log.strip()
    if not log or log.upper() == "MALFORMED_LOG":
        return None
    # Try triple-colon pattern
    if ":::" in log:
        res = parse_triple_colon(log)
        if res:
            return res
    # Try compound patterns
    res = parse_compound_patterns(log)
    if res:
        return res
    # Try simple colon pattern
    if "::" in log:
        res = parse_simple_colon(log)
        if res:
            return res
    # Fallback parsing
    return _fallback_parse_log(log)
