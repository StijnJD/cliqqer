import requests
import json
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_URL = "hier webhook van 112hook.nl"  
OUTPUT_FILE = Path("p2000_meldingen.json")
POLL_INTERVAL = 30  # seconden


def fetch_meldingen() -> list[dict]:
    """Haal live P2000 meldingen op via 112hook.nl."""
    response = requests.get(API_URL, timeout=10, headers={"Accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    # Normaliseer naar lijst als API een dict teruggeeft
    if isinstance(data, dict):
        return data.get("meldingen", data.get("data", data.get("results", [data])))
    return data


def parse_melding(raw: dict) -> dict:
    """Extraheer relevante velden uit een ruwe melding."""
    return {
        "timestamp": raw.get("timestamp") or raw.get("time") or raw.get("date") or datetime.utcnow().isoformat(),
        "type_dienst": raw.get("dienst") or raw.get("type") or raw.get("service") or "Onbekend",
        "locatie": raw.get("locatie") or raw.get("location") or raw.get("address") or "Onbekend",
        "raw": raw,
    }


def laad_bestaande(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Kon bestaand bestand niet lezen: %s", e)
    return []


def sla_op(meldingen: list[dict], path: Path) -> None:
    path.write_text(json.dumps(meldingen, ensure_ascii=False, indent=2), encoding="utf-8")


def run_eenmalig() -> None:
    """Haal meldingen op en sla op — eenmalige uitvoering."""
    logger.info("Ophalen van P2000 meldingen via %s", API_URL)
    raws = fetch_meldingen()
    parsed = [parse_melding(r) for r in raws]
    bestaand = laad_bestaande(OUTPUT_FILE)
    # Deduplicatie op basis van timestamp + locatie
    bestaand_keys = {(m["timestamp"], m["locatie"]) for m in bestaand}
    nieuw = [m for m in parsed if (m["timestamp"], m["locatie"]) not in bestaand_keys]
    if nieuw:
        gecombineerd = bestaand + nieuw
        sla_op(gecombineerd, OUTPUT_FILE)
        logger.info("%d nieuwe melding(en) opgeslagen in %s", len(nieuw), OUTPUT_FILE)
    else:
        logger.info("Geen nieuwe meldingen.")


def run_continu() -> None:
    """Poll continu met POLL_INTERVAL seconden tussenpauze."""
    logger.info("Continu polling gestart (interval: %ds). Ctrl+C om te stoppen.", POLL_INTERVAL)
    while True:
        try:
            run_eenmalig()
        except requests.exceptions.ConnectionError as e:
            logger.error("Verbindingsfout: %s", e)
        except requests.exceptions.Timeout:
            logger.error("Verzoek timeout na 10 seconden.")
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP-fout: %s", e)
        except requests.exceptions.RequestException as e:
            logger.error("Onverwachte netwerk­fout: %s", e)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Fout bij verwerken van data: %s", e)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    import sys

    if "--continu" in sys.argv or "-c" in sys.argv:
        run_continu()
    else:
        try:
            run_eenmalig()
        except requests.exceptions.ConnectionError as e:
            logger.error("Verbindingsfout: %s", e)
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP-fout: %s", e)
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            logger.error("Netwerkfout: %s", e)
            sys.exit(1)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Datafout: %s", e)
            sys.exit(1)
